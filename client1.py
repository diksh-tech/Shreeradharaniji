# client.py
import os
import json
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import AzureOpenAI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tool_registry import TOOLS

# Load environment variables
load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000").rstrip("/")

# Azure OpenAI configuration
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_KEY:
    raise RuntimeError("❌ AZURE_OPENAI_KEY not set in environment")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("FlightOps.MCPClient")

# Initialize Azure OpenAI client
client_azure = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ---------------------------------------------------------------------
#  SYSTEM PROMPTS
# ---------------------------------------------------------------------
def _build_tool_prompt() -> str:
    """Convert TOOLS dict into compact text to feed the LLM."""
    lines = []
    for name, meta in TOOLS.items():
        arg_str = ", ".join(meta["args"])
        lines.append(f"- {name}({arg_str}): {meta['desc']}")
    return "\n".join(lines)


SYSTEM_PROMPT_PLAN = f"""
You are an assistant that converts user questions into MCP tool calls.

Available tools:
{_build_tool_prompt()}

### IMPORTANT: ROUTE SELECTION IS AUTOMATIC
- Route selection is handled automatically by the system - you don't need to manage it
- Always include `session_id` parameter in flight-related tools
- If multiple routes exist, the system will automatically prompt the user for selection
- Continue with normal tool execution - no special handling needed
- The system uses standardized response format with "requires_route_selection" flag

### TIME CONVERSION RULES:
1. **ALWAYS convert UTC times to local time (IST)** when showing to users
2. Use `convert_utc_to_local_time` for:
   - Flight departure/arrival times
   - Scheduled times  
   - Any UTC timestamp from flight data

3. **Example usage:**
   - Input: "2024-06-23 13:25" (UTC)
   - Output: Shows both UTC and IST (13:25 UTC → 18:55 IST)

4. **Never show only UTC** - always convert to local time

### Tool selection logic

1. **Use `run_aggregated_query`** for statistical queries (counts, averages, sums, etc.)
2. **Use `raw_mongodb_query`** for filtered listings of flights
3. **Use specific flight tools** (get_flight_basic_info, etc.) for single-flight queries

### Session Management:
- Always include `session_id` in flight-related tool calls
- Use the same session_id for related queries in the same conversation

### General rules
1. Always return valid JSON with a top-level "plan" key.
2. Use the correct tool type based on query intent.
3. Never invent field names — use schema fields only.
4. Never return "_id" in projections.
5. For numerical summaries → use run_aggregated_query.
6. For filtered listings → use raw_mongodb_query.
7. For time conversions → use convert_utc_to_local_time
8. **Route selection is automatic** - just include session_id
"""


SYSTEM_PROMPT_SUMMARIZE = """
You are an assistant that summarizes tool outputs into a concise, readable answer.
When summarizing time conversions, clearly mention the UTC time, the +5:30 hour offset applied,
and the resulting local (IST) time. Focus on clarity and readability.
Be factual, short, and helpful.
"""

# ---------------------------------------------------------------------
#  FLIGHTOPS MCP CLIENT CLASS
# ---------------------------------------------------------------------
class FlightOpsMCPClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or MCP_SERVER_URL).rstrip("/")
        self.session: ClientSession = None
        self._client_context = None

    # -------------------- CONNECTION HANDLERS -------------------------
    async def connect(self):
        try:
            logger.info(f"Connecting to MCP server at {self.base_url}")
            self._client_context = streamablehttp_client(self.base_url)
            read_stream, write_stream, _ = await self._client_context.__aenter__()
            self.session = ClientSession(read_stream, write_stream)
            await self.session.__aenter__()
            await self.session.initialize()
            logger.info("✅ Connected to MCP server successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

    async def disconnect(self):
        try:
            if self.session:
                await self.session.__aexit__(None, None, None)
            if self._client_context:
                await self._client_context.__aexit__(None, None, None)
            logger.info("Disconnected from MCP server")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    # -------------------- AZURE OPENAI WRAPPER -------------------------
    def _call_azure_openai(self, messages: list, temperature: float = 0.2, max_tokens: int = 2048) -> dict:
        try:
            completion = client_azure.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage_obj = getattr(completion, "usage", None)
            usage = None
            if usage_obj is not None:
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
                }
            return {
                "content": completion.choices[0].message.content,
                "usage": usage
            }
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            return {"content": json.dumps({"error": str(e)}), "usage": None}

    # -------------------- MCP TOOL CALLS -------------------------
    async def list_tools(self) -> dict:
        try:
            if not self.session:
                await self.connect()
            tools_list = await self.session.list_tools()
            tools_dict = {tool.name: {"description": tool.description, "inputSchema": tool.inputSchema} for tool in tools_list.tools}
            return {"tools": tools_dict}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {"error": str(e)}

    async def invoke_tool(self, tool_name: str, args: dict) -> dict:
        try:
            if not self.session:
                await self.connect()
            logger.info(f"Calling tool: {tool_name} with args: {args}")
            result = await self.session.call_tool(tool_name, args)

            if result.content:
                content_items = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        try:
                            content_items.append(json.loads(item.text))
                        except json.JSONDecodeError:
                            content_items.append(item.text)
                if len(content_items) == 1:
                    return content_items[0]
                return {"results": content_items}

            return {"error": "No content in response"}
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {e}")
            return {"error": str(e)}

    # -------------------- LLM PLANNING & SUMMARIZATION -------------------------
    def plan_tools(self, user_query: str) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PLAN},
            {"role": "user", "content": user_query},
        ]

        res = self._call_azure_openai(messages, temperature=0.1)
        content = res.get("content")
        plan_usage = res.get("usage")
        
        if not content:
            logger.warning("⚠️ LLM returned empty response during plan generation.")
            return {"plan": [], "llm_usage": plan_usage}

        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            cleaned = cleaned.replace("```", "").strip()

        if cleaned != content:
            logger.debug(f"🔍 Cleaned LLM plan output:\n{cleaned}")

        try:
            plan = json.loads(cleaned)
            if isinstance(plan, dict) and "plan" in plan:
                return {"plan": plan["plan"], "llm_usage": plan_usage}
            else:
                logger.warning("⚠️ LLM output did not contain 'plan' key.")
                return {"plan": [], "llm_usage": plan_usage}
        except json.JSONDecodeError:
            logger.warning(f"❌ Could not parse LLM plan output after cleaning:\n{cleaned}")
            return {"plan": [], "llm_usage": plan_usage}

    def summarize_results(self, user_query: str, plan: list, results: list) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARIZE},
            {"role": "user", "content": f"Question:\n{user_query}"},
            {"role": "assistant", "content": f"Plan:\n{json.dumps(plan, indent=2)}"},
            {"role": "assistant", "content": f"Results:\n{json.dumps(results, indent=2)}"},
        ]
        res = self._call_azure_openai(messages, temperature=0.3)
        return {"summary": res.get("content"), "llm_usage": res.get("usage")}

    async def run_query(self, user_query: str) -> dict:
        """
        Full flow with proper token tracking for cost monitoring
        """
        try:
            logger.info(f"User query: {user_query}")
            plan_data = self.plan_tools(user_query)
            plan = plan_data.get("plan", [])
            planning_usage = plan_data.get("llm_usage")
            
            if not plan:
                return {
                    "error": "LLM did not produce a valid tool plan.",
                    "token_usage": {
                        "planning": planning_usage,
                        "summarization": None,
                        "total": planning_usage
                    }
                }

            # Execute the plan
            results = []
            for step in plan:
                tool = step.get("tool")
                args = step.get("arguments", {})
                
                # Clean up bad args
                args = {k: v for k, v in args.items() if v and str(v).strip().lower() != "unknown"}

                if tool == "raw_mongodb_query":
                    query_json = args.get("query_json", "")
                    if not query_json:
                        results.append({"raw_mongodb_query": {"error": "Empty query_json"}})
                        continue
                    args["limit"] = int(args.get("limit", 50))
                    logger.info(f"Executing raw MongoDB query: {query_json}")

                resp = await self.invoke_tool(tool, args)
                results.append({tool: resp})

            # Summarize results
            summary_data = self.summarize_results(user_query, plan, results)
            summarization_usage = summary_data.get("llm_usage")

            # Calculate total tokens for cost tracking
            def safe_int(value):
                try:
                    return int(value) if value is not None else 0
                except (ValueError, TypeError):
                    return 0

            total_tokens = {
                "prompt_tokens": safe_int(planning_usage and planning_usage.get("prompt_tokens")) + 
                                safe_int(summarization_usage and summarization_usage.get("prompt_tokens")),
                "completion_tokens": safe_int(planning_usage and planning_usage.get("completion_tokens")) + 
                                    safe_int(summarization_usage and summarization_usage.get("completion_tokens")),
                "total_tokens": safe_int(planning_usage and planning_usage.get("total_tokens")) + 
                               safe_int(summarization_usage and summarization_usage.get("total_tokens"))
            }

            return {
                "plan": plan,
                "results": results,
                "summary": {"summary": summary_data.get("summary")},
                "token_usage": {
                    "planning": planning_usage,
                    "summarization": summarization_usage,
                    "total": total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error in run_query: {e}")
            return {"error": str(e)}

    # -------------------- TEST FUNCTION FOR TIME CONVERSION -------------------------
    async def test_time_conversion(self):
        """
        Test converting a UTC time to local time
        """
        await self.connect()
        
        result = await self.invoke_tool("convert_utc_to_local_time", {
            "utc_time_str": "2024-06-23 13:25",
            "timezone_str": "Asia/Kolkata"
        })
        
        print("🕐 Time conversion test result:")
        print(json.dumps(result, indent=2))
        await self.disconnect()
