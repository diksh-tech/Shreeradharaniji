# ag_ui_adapter.py
import os
import json
import asyncio
import time
import uuid
from typing import AsyncGenerator, List, Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from client import FlightOpsMCPClient

from redis import Redis
from redis_entraid.cred_provider import create_from_service_principal
import logging

app = FastAPI(title="FlightOps — AG-UI Adapter")

# CORS (adjust origins for your Vite origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mcp_client = FlightOpsMCPClient()

# ------------------------------------------------------------------------------------
# REDIS CONFIGURATION
# ------------------------------------------------------------------------------------
REDIS_HOST = "occh-uamr01.centralindia.redis.azure.net"
REDIS_PORT = 10000
REDIS_CLIENT_ID = os.getenv("REDIS_CLIENT_ID")
REDIS_CLIENT_SECRET = os.getenv("REDIS_CLIENT_SECRET")
REDIS_TENANT_ID = os.getenv("REDIS_TENANT_ID")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis credential provider
try:
    redis_credential_provider = create_from_service_principal(
        REDIS_CLIENT_ID,
        REDIS_CLIENT_SECRET,
        REDIS_TENANT_ID,
    )
except Exception as e:
    logger.error("Failed to create redis credential provider: %s", e)
    redis_credential_provider = None

# DEV: create SSL flags that disable verification (ONLY FOR DEV)
_dev_disable_ssl_verify = os.getenv("DISABLE_REDIS_SSL_VERIFY", "1") == "1"
if _dev_disable_ssl_verify:
    logger.warning("Redis SSL verification DISABLED (development)")

redis_client = None

try:
    if redis_credential_provider:
        redis_client = Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            ssl=True,
            credential_provider=redis_credential_provider,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            ssl_cert_reqs=None if _dev_disable_ssl_verify else "required",
            ssl_check_hostname=False if _dev_disable_ssl_verify else True,
        )
        # quick non-fatal ping to surface connectivity issues early
        try:
            ok = redis_client.ping()
            logger.info("Redis ping -> %s", ok)
        except Exception as e:
            logger.warning("Redis ping failed (will continue): %s", e)
    else:
        redis_client = None
except TypeError as te:
    logger.error("Redis client init TypeError (unsupported arg): %s", te)
    redis_client = None
except Exception as e:
    logger.error("Failed to create redis_client: %s", e)
    redis_client = None

# Redis namespace configuration
NAMESPACE = "non-prod"
PROJECT = "occhub"
MODULE = "flight_mcp"
HISTORY_TTL_SECONDS = 60 * 60 * 24  # 1 day
MAX_HISTORY_MESSAGES = 20  # last 20 messages (user+assistant)

def make_history_key(session_id: str) -> str:
    """Create Redis key for session history"""
    return f"{NAMESPACE}:{PROJECT}:{MODULE}:history:{session_id}"

def make_session_key(session_id: str) -> str:
    """Create Redis key for session metadata"""
    return f"{NAMESPACE}:{PROJECT}:{MODULE}:session:{session_id}"

def make_route_selection_key(session_id: str) -> str:
    """Create Redis key for route selection state"""
    return f"{NAMESPACE}:{PROJECT}:{MODULE}:route_selection:{session_id}"

async def append_to_history(session_id: str, role: str, content: str, metadata: dict = None) -> None:
    """Append a message to session history"""
    if not redis_client:
        logger.debug("append_to_history: redis_client is not available, skipping")
        return

    key = make_history_key(session_id)
    try:
        message_data = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        # Use pipeline for atomic operations
        pipe = redis_client.pipeline()
        pipe.rpush(key, json.dumps(message_data, ensure_ascii=False))
        pipe.ltrim(key, -MAX_HISTORY_MESSAGES, -1)  # Keep only last N messages
        pipe.expire(key, HISTORY_TTL_SECONDS)
        pipe.execute()

        logger.debug(f"💾 Saved message to history for session: {session_id}")
    except Exception as e:
        logger.warning("append_to_history failed for key=%s: %s", key, e)

async def load_history(session_id: str, max_messages: int = MAX_HISTORY_MESSAGES) -> List[dict]:
    """Load session history from Redis"""
    if not redis_client:
        logger.debug("load_history: redis_client is not available")
        return []

    key = make_history_key(session_id)
    try:
        length = redis_client.llen(key)
    except Exception as e:
        logger.warning("Redis llen failed for key=%s: %s", key, e)
        return []

    if not length:
        return []

    start = max(0, length - max_messages)
    try:
        raw_msgs = redis_client.lrange(key, start, -1)
    except Exception as e:
        logger.warning("Redis lrange failed for key=%s: %s", key, e)
        return []

    messages = []
    for raw in raw_msgs:
        try:
            messages.append(json.loads(raw))
        except Exception:
            continue
    return messages

async def create_or_update_session(session_id: str, user_query: str = None) -> None:
    """Create or update session metadata"""
    if not redis_client:
        return

    try:
        key = make_session_key(session_id)

        # Generate title based on user query
        title = f"Flight Query: {user_query[:50]}..." if user_query else f"Session {session_id[-8:]}"

        session_data = {
            "session_id": session_id,
            "title": title,
            "created_at": time.time(),
            "last_activity": time.time(),
            "message_count": 0,
            "last_query": user_query or ""
        }

        # Update existing or create new
        existing = redis_client.get(key)
        if existing:
            try:
                existing_data = json.loads(existing)
                session_data["created_at"] = existing_data.get("created_at", time.time())
                session_data["message_count"] = existing_data.get("message_count", 0) + 1
                # Update title if it's a new query
                if user_query and len(user_query) > 10:
                    session_data["title"] = title
                else:
                    session_data["title"] = existing_data.get("title", title)
            except json.JSONDecodeError:
                pass

        redis_client.setex(key, HISTORY_TTL_SECONDS, json.dumps(session_data, ensure_ascii=False))
        logger.debug(f"💾 Updated session metadata: {session_id}")
    except Exception as e:
        logger.warning(f"Failed to update session {session_id}: {e}")

async def save_route_selection_state(session_id: str, state: dict) -> None:
    """Save route selection state to Redis"""
    if not redis_client:
        return

    try:
        key = make_route_selection_key(session_id)
        redis_client.setex(key, HISTORY_TTL_SECONDS, json.dumps(state, ensure_ascii=False))
        logger.debug(f"💾 Saved route selection state for session: {session_id}")
    except Exception as e:
        logger.warning(f"Failed to save route selection state for {session_id}: {e}")

async def load_route_selection_state(session_id: str) -> dict:
    """Load route selection state from Redis"""
    if not redis_client:
        return {}

    try:
        key = make_route_selection_key(session_id)
        state_json = redis_client.get(key)
        if state_json:
            return json.loads(state_json)
    except Exception as e:
        logger.warning(f"Failed to load route selection state for {session_id}: {e}")
    
    return {}

async def clear_route_selection_state(session_id: str) -> None:
    """Clear route selection state from Redis"""
    if not redis_client:
        return

    try:
        key = make_route_selection_key(session_id)
        redis_client.delete(key)
        logger.debug(f"🧹 Cleared route selection state for session: {session_id}")
    except Exception as e:
        logger.warning(f"Failed to clear route selection state for {session_id}: {e}")

# ------------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------------
def sse_event(data: dict) -> str:
    """Encode one SSE event (JSON payload)"""
    return f"data: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"

async def ensure_mcp_connected():
    if not mcp_client.session:
        await mcp_client.connect()

@app.on_event("startup")
async def startup_event():
    try:
        await ensure_mcp_connected()
    except Exception:
        pass

@app.get("/")
async def root():
    return {"message": "FlightOps AG-UI Adapter running", "status": "ok"}

@app.get("/health")
async def health():
    try:
        await ensure_mcp_connected()
        redis_status = "connected" if redis_client and redis_client.ping() else "disconnected"
        return {
            "status": "healthy",
            "mcp_connected": True,
            "redis_connected": redis_status
        }
    except Exception as e:
        return {"status": "unhealthy", "mcp_connected": False, "error": str(e)}

def chunk_text(txt: str, max_len: int = 200) -> List[str]:
    txt = txt or ""
    parts: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf:
            parts.append(buf)
            buf = ""

    for ch in txt:
        buf += ch
        if ch in ".!?\n" and len(buf) >= max_len // 2:
            flush()
        elif len(buf) >= max_len:
            flush()
    flush()
    return parts

# ✅ FIXED: Consistent route selection detection
def _detect_route_selection(tool_result: Any) -> tuple[bool, Optional[dict]]:
    """
    Detect if tool result indicates route selection is needed.
    Returns (needs_selection, route_data)
    """
    try:
        if isinstance(tool_result, dict):
            # Case 1: Standardized format with explicit flag
            if tool_result.get("ok") and tool_result.get("requires_route_selection"):
                data_field = tool_result.get("data")
                if isinstance(data_field, dict) and "available_routes" in data_field:
                    return True, data_field
            
            # Case 2: Direct available_routes in data field (JSON string)
            if tool_result.get("ok"):
                data_field = tool_result.get("data")
                if isinstance(data_field, str):
                    try:
                        parsed_data = json.loads(data_field)
                        if isinstance(parsed_data, dict) and "available_routes" in parsed_data:
                            return True, parsed_data
                    except json.JSONDecodeError:
                        pass
                elif isinstance(data_field, dict) and "available_routes" in data_field:
                    return True, data_field
            
            # Case 3: Direct available_routes in tool result
            if "available_routes" in tool_result:
                return True, tool_result
        
        elif isinstance(tool_result, str):
            # Handle string JSON responses
            try:
                parsed = json.loads(tool_result)
                if isinstance(parsed, dict):
                    if parsed.get("ok") and parsed.get("requires_route_selection"):
                        data_field = parsed.get("data")
                        if isinstance(data_field, dict) and "available_routes" in data_field:
                            return True, data_field
                    if "available_routes" in parsed:
                        return True, parsed
            except:
                pass
        
        return False, None
        
    except Exception as e:
        logger.error(f"Route selection detection error: {e}")
        return False, None

# ✅ UPDATED: LLM Summarization after route selection - returns dict
async def _summarize_after_route_selection(user_query: str, tool_name: str, tool_result: dict, selected_route: dict) -> dict:
    """
    Use LLM to summarize the tool result after route selection
    Returns dict with 'summary' and 'llm_usage'
    """
    try:
        loop = asyncio.get_event_loop()
        
        # Prepare the plan and results in the same format as original agent flow
        plan_step = {
            "tool": tool_name,
            "arguments": {
                "carrier": selected_route.get("carrier", ""),
                "flight_number": str(selected_route.get("flightNumber", "")),
                "date_of_origin": selected_route.get("dateOfOrigin", ""),
                "session_id": "route_selection_context"
            }
        }
        
        results_data = [{tool_name: tool_result}]
        
        # Use the same summarization logic as main agent
        summary_data = await loop.run_in_executor(
            None, 
            mcp_client.summarize_results, 
            user_query, 
            [plan_step], 
            results_data
        )
        
        summary_text = ""
        llm_usage = None
        
        if isinstance(summary_data, dict):
            summary_text = summary_data.get("summary", "")
            llm_usage = summary_data.get("llm_usage")
        else:
            summary_text = str(summary_data)
        
        # Add route context to the summary
        route_context = f"✅ **Route Selected**: {selected_route['startStation']} → {selected_route['endStation']}\n\n"
        
        return {
            "summary": route_context + summary_text,
            "llm_usage": llm_usage
        }
        
    except Exception as e:
        logger.error(f"Route selection summarization failed: {e}")
        # Fallback to basic summary
        return {
            "summary": f"✅ **Route Selected**: {selected_route['startStation']} → {selected_route['endStation']}\n\n📊 Data retrieved successfully. {tool_name.replace('get_', '').replace('_', ' ').title()} information is now available.",
            "llm_usage": None
        }

# ------------------------------------------------------------------------------------
# AG-UI /agent endpoint (LLM-based only)
# ------------------------------------------------------------------------------------
@app.post("/agent", response_class=StreamingResponse)
async def run_agent(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    thread_id = body.get("thread_id") or f"thread-{uuid.uuid4().hex[:8]}"
    run_id = body.get("run_id") or f"run-{uuid.uuid4().hex[:8]}"
    messages = body.get("messages", [])

    # Use thread_id as session_id for history tracking
    session_id = thread_id

    # last user message as query
    user_query = ""
    if messages:
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") == "user":
            user_query = last.get("content", "") or ""
        elif isinstance(last, str):
            user_query = last

    if not user_query.strip():
        raise HTTPException(status_code=400, detail="No user query found")

    # Save user message to history and update session (Redis integration)
    await append_to_history(session_id, "user", user_query, {"run_id": run_id})
    await create_or_update_session(session_id, user_query)

    # Always use LLM-based handling
    return await handle_llm_based_query(user_query, thread_id, run_id, session_id, request)

# ------------------------------------------------------------------------------------
# LLM-based query streaming handler with DROPDOWN route selection - FIXED VERSION
# ------------------------------------------------------------------------------------
async def handle_llm_based_query(user_query: str, thread_id: str, run_id: str, session_id: str, request: Request):
    """
    Handle all queries using LLM planning with DROPDOWN route selection
    """
    async def event_stream() -> AsyncGenerator[str, None]:
        last_heartbeat = time.time()

        # --- RUN STARTED
        yield sse_event({"type": "RUN_STARTED", "thread_id": thread_id, "run_id": run_id})

        # Load conversation history for context
        history_messages = await load_history(session_id)

        # THINKING: Initial analysis
        yield sse_event({
            "type": "STATE_UPDATE",
            "state": {
                "phase": "thinking",
                "progress_pct": 5,
                "message": "🧠 Analyzing your query..."
            }
        })

        # ensure MCP
        try:
            await ensure_mcp_connected()
        except Exception as e:
            yield sse_event({"type": "RUN_ERROR", "error": f"MCP connect failed: {e}"})
            return

        loop = asyncio.get_event_loop()

        # --- PLAN (THINKING phase)
        yield sse_event({
            "type": "STATE_UPDATE",
            "state": {
                "phase": "thinking",
                "progress_pct": 15,
                "message": "📋 Planning which tools to use..."
            }
        })

        try:
            plan_data = await loop.run_in_executor(None, mcp_client.plan_tools, user_query)
        except Exception as e:
            yield sse_event({"type": "RUN_ERROR", "error": f"Planner error: {e}"})
            return

        plan = plan_data.get("plan", []) if isinstance(plan_data, dict) else []
        planning_usage = plan_data.get("llm_usage", {})

        if planning_usage:
            yield sse_event({
                "type": "TOKEN_USAGE",
                "phase": "planning",
                "usage": planning_usage
            })

        yield sse_event({"type": "STATE_SNAPSHOT", "snapshot": {"plan": plan}})

        if not plan:
            error_msg = "I couldn't generate a valid plan for your query. Please try rephrasing."
            yield sse_event({
                "type": "TEXT_MESSAGE_CONTENT",
                "message": {
                    "id": f"msg-{uuid.uuid4().hex[:8]}",
                    "role": "assistant",
                    "content": error_msg
                }
            })
            await append_to_history(session_id, "assistant", error_msg, {"run_id": run_id, "error": True})
            yield sse_event({"type": "STATE_UPDATE", "state": {"phase": "finished", "progress_pct": 100}})
            yield sse_event({"type": "RUN_FINISHED", "run_id": run_id})
            return

        # --- PROCESSING: Tool execution with DROPDOWN route selection handling
        yield sse_event({
            "type": "STATE_UPDATE",
            "state": {
                "phase": "processing",
                "progress_pct": 20,
                "message": f"🛠️ Executing {len(plan)} tools..."
            }
        })

        results = []
        num_steps = max(1, len(plan))
        per_step = 60.0 / num_steps
        current_progress = 20.0

        # Track route selection state
        route_selection_required = False
        route_selection_data = None
        pending_tool_after_route_selection = None
        selection_step_index = None  # ✅ NEW: Track which step triggered route selection

        for step_index, step in enumerate(plan):
            # Check for disconnection
            try:
                if await request.is_disconnected():
                    return
            except:
                pass

            # If we're in route selection mode, skip normal tool execution
            if route_selection_required:
                pending_tool_after_route_selection = step
                selection_step_index = step_index
                break

            tool_name = step.get("tool")
            args = step.get("arguments", {}) or {}

            # Add session_id to all flight-related tools
            if tool_name in ["get_flight_basic_info", "get_operation_times", "get_equipment_info", 
                           "get_delay_summary", "get_fuel_summary", "get_passenger_info", "get_crew_info"]:
                args["session_id"] = session_id

            yield sse_event({
                "type": "STATE_UPDATE",
                "state": {
                    "phase": "processing",
                    "progress_pct": round(current_progress),
                    "message": f"🔧 Running {tool_name}..."
                }
            })

            tool_call_id = f"toolcall-{uuid.uuid4().hex[:8]}"

            # AG-UI compatible tool call events
            yield sse_event({
                "type": "TOOL_CALL_START",
                "toolCallId": tool_call_id,
                "toolCallName": tool_name,
                "parentMessageId": None
            })

            yield sse_event({
                "type": "TOOL_CALL_ARGS",
                "toolCallId": tool_call_id,
                "delta": json.dumps(args, ensure_ascii=False)
            })

            try:
                tool_result = await mcp_client.invoke_tool(tool_name, args)
            except Exception as exc:
                tool_result = {"error": str(exc)}
                logger.error(f"Tool execution failed: {exc}")

            # Result event
            yield sse_event({
                "type": "TOOL_CALL_RESULT",
                "message": {
                    "id": f"msg-{uuid.uuid4().hex[:8]}",
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                    "tool_call_id": tool_call_id,
                }
            })
            results.append({tool_name: tool_result})

            yield sse_event({
                "type": "TOOL_CALL_END",
                "toolCallId": tool_call_id
            })

            # Progress update
            current_progress = min(80.0, 20.0 + per_step * (step_index + 1))

            # ✅✅✅ FIXED: Route-selection detection AFTER result emit
            needs_selection, selection_data = _detect_route_selection(tool_result)
            if needs_selection:
                logger.info(
                    f"🚀 Route selection detected! "
                    f"{len(selection_data.get('available_routes', []))} routes found"
                )
                route_selection_required = True
                route_selection_data = selection_data
                pending_tool_after_route_selection = step
                selection_step_index = step_index
                break  # ✅ Stop further tool execution

            # Heartbeat
            if time.time() - last_heartbeat > 15:
                yield sse_event({"type": "HEARTBEAT", "ts": time.time()})
                last_heartbeat = time.time()

        # --- Handle Route Selection if needed
        if route_selection_required and route_selection_data:
            # Send DROPDOWN route selection event
            available_routes = route_selection_data.get("available_routes", [])
            yield sse_event({
                "type": "ROUTE_SELECTION_OPTIONS",
                "message": f"Found {len(available_routes)} routes for your flight. Please select one:",
                "routes": available_routes,
                "session_id": session_id,
                "run_id": run_id,
                "thread_id": thread_id
            })
            
            # ✅ FIXED: Save the route selection state with proper remaining plan
            route_selection_state = {
                "run_id": run_id,
                "thread_id": thread_id,
                "user_query": user_query,  # ✅ CRITICAL: Save for summarization
                "route_data": route_selection_data,
                "pending_tool": pending_tool_after_route_selection,
                "completed_results": results,
                "remaining_plan": (
                    plan[selection_step_index + 1:]
                    if selection_step_index is not None
                    and selection_step_index + 1 < len(plan)
                    else []
                ),
            }
            
            await save_route_selection_state(session_id, route_selection_state)
            
            yield sse_event({
                "type": "STATE_UPDATE",
                "state": {
                    "phase": "waiting_for_input",
                    "progress_pct": 90,
                    "message": "⏳ Waiting for route selection..."
                }
            })
            
            # Don't proceed to summarization - wait for user input
            yield sse_event({"type": "RUN_FINISHED", "run_id": run_id})
            return

        # --- TYPING: Result generation (only if no route selection needed)
        yield sse_event({
            "type": "STATE_UPDATE",
            "state": {
                "phase": "typing",
                "progress_pct": 85,
                "message": "✍️ Generating your analysis..."
            }
        })

        try:
            summary_data = await loop.run_in_executor(None, mcp_client.summarize_results, user_query, plan, results)
            assistant_text = summary_data.get("summary", "") if isinstance(summary_data, dict) else str(summary_data)
            summarization_usage = summary_data.get("llm_usage", {})

            if summarization_usage:
                yield sse_event({
                    "type": "TOKEN_USAGE",
                    "phase": "summarization",
                    "usage": summarization_usage
                })

        except Exception as e:
            assistant_text = f"❌ Failed to summarize results: {e}"

        # Save assistant response to history (Redis integration)
        await append_to_history(session_id, "assistant", assistant_text, {
            "run_id": run_id,
            "tools_used": [step.get("tool") for step in plan],
            "results_count": len(results)
        })

        # Stream summary as chunks - AG-UI compatible format
        msg_id = f"msg-{uuid.uuid4().hex[:8]}"

        # Start with empty message
        yield sse_event({
            "type": "TEXT_MESSAGE_CONTENT",
            "message": {
                "id": msg_id,
                "role": "assistant",
                "content": ""
            }
        })

        # Stream content chunks
        chunks = chunk_text(assistant_text, max_len=150)
        for i, chunk in enumerate(chunks):
            yield sse_event({
                "type": "TEXT_MESSAGE_CONTENT",
                "message": {
                    "id": msg_id,
                    "role": "assistant",
                    "delta": chunk
                }
            })

            typing_progress = 85 + (i / len(chunks)) * 15
            yield sse_event({
                "type": "STATE_UPDATE",
                "state": {
                    "phase": "typing",
                    "progress_pct": round(typing_progress),
                    "message": "✍️ Generating your analysis..."
                }
            })
            await asyncio.sleep(0.03)

        # Final state with history info (Redis integration)
        yield sse_event({
            "type": "STATE_SNAPSHOT",
            "snapshot": {
                "plan": plan,
                "results": results,
                "session_id": session_id,
                "history_count": len(history_messages) + 2
            }
        })
        yield sse_event({
            "type": "STATE_UPDATE",
            "state": {
                "phase": "finished",
                "progress_pct": 100,
                "message": f"✅ Analysis complete (Session: {session_id})"
            }
        })
        yield sse_event({"type": "RUN_FINISHED", "run_id": run_id})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ------------------------------------------------------------------------------------
# Route Selection Handler Endpoint with LLM Summarization (ENHANCED)
# ------------------------------------------------------------------------------------
@app.post("/sessions/{session_id}/select-route")
async def handle_route_selection(session_id: str, request: Request):
    """
    Enhanced route selection with LLM summarization
    """
    try:
        body = await request.json()
        selected_route_index = body.get("route_index")
        
        if selected_route_index is None:
            raise HTTPException(status_code=400, detail="No route index provided")
        
        # Get the route selection state from Redis
        route_state = await load_route_selection_state(session_id)
        if not route_state:
            raise HTTPException(status_code=404, detail="No pending route selection found")
        
        route_data = route_state["route_data"]
        pending_tool = route_state["pending_tool"]
        user_query = route_state.get("user_query", "")
        completed_results = route_state.get("completed_results", [])
        routes = route_data.get("available_routes", [])
        
        # Validate route index
        try:
            route_index = int(selected_route_index)
            if route_index < 0 or route_index >= len(routes):
                raise HTTPException(status_code=400, detail="Invalid route index")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid route index format")
        
        selected_route = routes[route_index]
        
        # Extract parameters from the selected route
        carrier = selected_route.get("carrier", "")
        flight_number = str(selected_route.get("flightNumber", ""))
        date_of_origin = selected_route.get("dateOfOrigin", "")
        
        # Step 1: Call select_route tool to set the route context
        select_result = await mcp_client.invoke_tool("select_route", {
            "session_id": session_id,
            "start_station": selected_route["startStation"],
            "end_station": selected_route["endStation"],
            "carrier": carrier,
            "flight_number": flight_number,
            "date_of_origin": date_of_origin
        })
        
        if not select_result.get("ok"):
            raise HTTPException(status_code=500, detail=f"Route selection failed: {select_result}")
        
        # Step 2: Execute the pending tool that was waiting for route selection
        final_tool_result = None
        llm_summary = None
        summarization_usage = None
        
        if pending_tool:
            tool_name = pending_tool.get("tool")
            tool_args = pending_tool.get("arguments", {}) or {}
            tool_args["session_id"] = session_id
            
            # Execute the tool that was blocked by route selection
            tool_result = await mcp_client.invoke_tool(tool_name, tool_args)
            final_tool_result = tool_result
            
            # ✅ CRITICAL: Step 3 - Generate LLM summary using the tool result
            llm_summary_data = await _summarize_after_route_selection(
                user_query, 
                tool_name, 
                tool_result, 
                selected_route
            )
            
            # ✅ FIXED: Now llm_summary_data is always a dict
            llm_summary = llm_summary_data.get("summary", "")
            summarization_usage = llm_summary_data.get("llm_usage")
        
        # Step 4: Clear the route selection state
        await clear_route_selection_state(session_id)
        
        # Step 5: Save to history
        selection_message = f"✅ Selected route: {selected_route['startStation']} → {selected_route['endStation']}"
        await append_to_history(session_id, "user", selection_message, {
            "route_selected": True,
            "selected_route": selected_route
        })
        
        if llm_summary:
            await append_to_history(session_id, "assistant", llm_summary, {
                "route_summary": True,
                "tool_used": pending_tool.get("tool") if pending_tool else None,
                "summarization_usage": summarization_usage
            })
        
        return {
            "status": "success",
            "selected_route": {
                "startStation": selected_route["startStation"],
                "endStation": selected_route["endStation"],
                "display": f"{selected_route['startStation']} → {selected_route['endStation']}"
            },
            "tool_result": final_tool_result,
            "llm_summary": llm_summary,
            "summarization_usage": summarization_usage,
            "message": "Route selected and data summarized successfully"
        }
            
    except Exception as e:
        logger.error(f"Route selection handling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------------
# Get Route Selection State Endpoint
# ------------------------------------------------------------------------------------
@app.get("/sessions/{session_id}/route-selection-state")
async def get_route_selection_state(session_id: str):
    """
    Get current route selection state for a session
    """
    route_state = await load_route_selection_state(session_id)
    if not route_state:
        raise HTTPException(status_code=404, detail="No route selection state found")
    
    return {
        "session_id": session_id,
        "route_state": route_state
    }

# ------------------------------------------------------------------------------------
# Session / history management endpoints (unchanged)
# ------------------------------------------------------------------------------------
@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 20):
    """Get conversation history for a session"""
    history = await load_history(session_id, limit)
    return {
        "session_id": session_id,
        "history": history,
        "count": len(history)
    }

@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get session metadata"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        key = make_session_key(session_id)
        session_data = redis_client.get(key)
        if session_data:
            return json.loads(session_data)
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving session: {e}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        history_key = make_history_key(session_id)
        session_key = make_session_key(session_id)
        route_key = make_route_selection_key(session_id)

        pipe = redis_client.pipeline()
        pipe.delete(history_key)
        pipe.delete(session_key)
        pipe.delete(route_key)
        result = pipe.execute()

        deleted_count = sum(result)
        return {
            "session_id": session_id,
            "deleted": deleted_count > 0,
            "items_removed": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting session: {e}")

@app.get("/sessions")
async def list_sessions(limit: int = 50):
    """List all active sessions"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        pattern = make_session_key("*")
        keys = redis_client.keys(pattern)
        sessions = []

        for key in keys[:limit]:
            session_data = redis_client.get(key)
            if session_data:
                sessions.append(json.loads(session_data))

        return {
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {e}")

@app.get("/chat-history")
async def get_chat_history():
    """Get complete chat history for sidebar - COMPATIBLE WITH FRONTEND"""
    if not redis_client:
        return {"sessions": []}

    try:
        # Get all sessions
        pattern = make_session_key("*")
        session_keys = redis_client.keys(pattern)
        session_keys.sort(reverse=True)  # Most recent first

        sessions_data = []

        for key in session_keys[:50]:  # Limit to 50 most recent sessions
            session_json = redis_client.get(key)
            if session_json:
                session_data = json.loads(session_json)

                # Get message count and last message
                history_key = make_history_key(session_data["session_id"])
                message_count = redis_client.llen(history_key)

                # Get last message for preview
                last_message = ""
                if message_count > 0:
                    last_msg_json = redis_client.lindex(history_key, -1)  # Get last message
                    if last_msg_json:
                        try:
                            last_msg = json.loads(last_msg_json)
                            text = last_msg.get("content", "")
                            last_message = text[:80] + "..." if len(text) > 80 else text
                        except Exception:
                            pass

                sessions_data.append({
                    "id": session_data["session_id"],
                    "title": session_data.get("title", f"Session {session_data['session_id'][-8:]}"),
                    "created_at": session_data["created_at"],
                    "last_activity": session_data["last_activity"],
                    "message_count": message_count,
                    "last_message": last_message,
                    "user_query": session_data.get("last_query", "")
                })

        return {"sessions": sessions_data}

    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return {"sessions": []}

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 100):
    """Get all messages for a specific session - COMPATIBLE WITH FRONTEND"""
    history_messages = await load_history(session_id, limit)

    # Format messages for frontend
    formatted_messages = []
    for msg in history_messages:
        formatted_messages.append({
            "id": f"msg-{uuid.uuid4().hex[:8]}",  # Generate unique ID for frontend
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg["timestamp"],
            "metadata": msg.get("metadata", {})
        })

    return {
        "session_id": session_id,
        "messages": formatted_messages,
        "count": len(formatted_messages)
    }
