# server.py
import os
import logging
import json
from typing import Optional, Any, Dict, List
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv() 
from bson import ObjectId

from mcp.server.fastmcp import FastMCP

HOST = os.getenv("MCP_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_PORT", "8000"))
TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")

MONGODB_URL = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("MONGO_DB")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("flightops.mcp.server")

mcp = FastMCP("FlightOps MCP Server")

_mongo_client: Optional[AsyncIOMotorClient] = None
_db = None
_col = None

# Global storage for route selection (in-memory, for production consider Redis)
_route_selections = {}
_multiple_routes_cache = {}

async def get_mongodb_client():
    """Initialize and return the global Motor client, DB and collection."""
    global _mongo_client, _db, _col
    if _mongo_client is None:
        logger.info("Connecting to MongoDB: %s", MONGODB_URL)
        _mongo_client = AsyncIOMotorClient(MONGODB_URL)
        _db = _mongo_client[DATABASE_NAME]
        _col = _db[COLLECTION_NAME]
    return _mongo_client, _db, _col

def normalize_flight_number(flight_number: Any) -> Optional[int]:
    """Convert flight_number to int. MongoDB stores it as int."""
    if flight_number is None or flight_number == "":
        return None
    if isinstance(flight_number, int):
        return flight_number
    try:
        return int(str(flight_number).strip())
    except (ValueError, TypeError):
        logger.warning(f"Could not normalize flight_number: {flight_number}")
        return None

def validate_date(date_str: str) -> Optional[str]:
    """
    Validate date_of_origin string. Accepts common formats.
    Returns normalized ISO date string YYYY-MM-DD if valid, else None.
    """
    if not date_str or date_str == "":
        return None
    
    # Handle common date formats
    formats = [
        "%Y-%m-%d",      # 2024-06-23
        "%d-%m-%Y",      # 23-06-2024
        "%Y/%m/%d",      # 2024/06/23
        "%d/%m/%Y",      # 23/06/2024
        "%B %d, %Y",     # June 23, 2024
        "%d %B %Y",      # 23 June 2024
        "%b %d, %Y",     # Jun 23, 2024
        "%d %b %Y"       # 23 Jun 2024
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    logger.warning(f"Could not parse date: {date_str}")
    return None

def make_query(carrier: str, flight_number: Optional[int], date_of_origin: str, start_station: Optional[str] = None, end_station: Optional[str] = None) -> Dict:
    """
    Build MongoDB query matching the actual database schema.
    """
    query = {}
    
    # Add carrier if provided
    if carrier:
        query["flightLegState.carrier"] = carrier
    
    # Add flight number as integer (as stored in DB)
    if flight_number is not None:
        query["flightLegState.flightNumber"] = flight_number
    
    # Add date if provided
    if date_of_origin:
        query["flightLegState.dateOfOrigin"] = date_of_origin
    
    # Add route selection if provided
    if start_station:
        query["flightLegState.startStation"] = start_station
    if end_station:
        query["flightLegState.endStation"] = end_station
    
    logger.info(f"Built query: {json.dumps(query)}")
    return query

def response_ok(data: Any) -> str:
    """Return JSON string for successful response."""
    return json.dumps({"ok": True, "data": data}, indent=2, default=str)

def response_error(msg: str, code: int = 400, **extra) -> str:
    """Return JSON string for error response."""
    error_data = {"message": msg, "code": code}
    error_data.update(extra)
    return json.dumps({"ok": False, "error": error_data}, indent=2)

# ✅ NEW: Standardized route selection response format
def response_route_selection_required(route_data: dict) -> str:
    """✅ STANDARDIZED: Route selection response format for SSE pipeline"""
    return json.dumps({
        "ok": True,
        "requires_route_selection": True,  # ✅ Explicit flag
        "data": route_data
    }, indent=2, default=str)

async def _fetch_multiple_routes(query: dict) -> Dict:
    """
    Check for multiple routes and return available options with FULL document details.
    Returns dict with count, routes, and full documents.
    """
    try:
        _, _, col = await get_mongodb_client()
        
        # ✅ FIXED: Use FULL projection to get complete document for all tools
        full_projection = {
            "flightLegState.carrier": 1,
            "flightLegState.flightNumber": 1,
            "flightLegState.dateOfOrigin": 1,
            "flightLegState.startStation": 1,
            "flightLegState.endStation": 1,
            "flightLegState.scheduledStartTime": 1,
            "flightLegState.scheduledEndTime": 1,
            # Include fields needed for all tools
            "flightLegState.operation": 1,
            "flightLegState.delays": 1,
            "flightLegState.equipment": 1,
            "flightLegState.pax": 1,
            "flightLegState.crewConnections": 1,
            "flightLegState.flightStatus": 1,
            "flightLegState.operationalStatus": 1
        }
        
        cursor = col.find(query, full_projection)
        documents = await cursor.to_list(length=20)  # Limit to 20 for safety
        
        if not documents:
            return {"count": 0, "routes": [], "documents": []}
        
        # Extract unique routes with document references
        routes = []
        seen_routes = set()
        
        for doc in documents:
            flight_data = doc.get("flightLegState", {})
            route_key = (
                flight_data.get("startStation"), 
                flight_data.get("endStation")
            )
            
            if route_key not in seen_routes:
                seen_routes.add(route_key)
                routes.append({
                    "startStation": flight_data.get("startStation"),
                    "endStation": flight_data.get("endStation"),
                    "carrier": flight_data.get("carrier"),
                    "flightNumber": flight_data.get("flightNumber"),
                    "dateOfOrigin": flight_data.get("dateOfOrigin"),
                    "scheduledStartTime": flight_data.get("scheduledStartTime"),
                    "scheduledEndTime": flight_data.get("scheduledEndTime"),
                    "document_index": len(routes)  # Index to reference the full document
                })
        
        return {
            "count": len(documents), 
            "routes": routes, 
            "documents": documents,
            "query": query
        }
        
    except Exception as exc:
        logger.exception("Route detection failed")
        return {"count": 0, "routes": [], "documents": [], "error": str(exc)}

async def _check_route_selection(session_id: str, carrier: str, flight_number: Optional[int], date_of_origin: str) -> Dict:
    """
    Check route selection status and return appropriate action.
    
    Returns:
        Dict with:
        - action: "proceed" | "route_selection_required" | "error"
        - data: route_info or error details
    """
    try:
        # Build base query without route
        base_query = make_query(carrier, flight_number, date_of_origin)
        
        # Check for multiple routes
        route_info = await _fetch_multiple_routes(base_query)
        
        if route_info["count"] == 0:
            return {
                "action": "error",
                "data": response_error(
                    f"No flights found for carrier={carrier}, flight_number={flight_number}, date={date_of_origin}",
                    code=404
                )
            }
        
        elif route_info["count"] == 1:
            # Single route found - store it and proceed
            single_route = route_info["routes"][0]
            _route_selections[session_id] = {
                "startStation": single_route["startStation"],
                "endStation": single_route["endStation"],
                "carrier": carrier,
                "flightNumber": flight_number,
                "dateOfOrigin": date_of_origin,
                "selected_document": route_info["documents"][0]  # ✅ Now contains FULL document
            }
            
            return {
                "action": "proceed",
                "data": {
                    "startStation": single_route["startStation"],
                    "endStation": single_route["endStation"]
                }
            }
            
        else:
            # Multiple routes found - cache them and require selection
            cache_key = f"{session_id}_{carrier}_{flight_number}_{date_of_origin}"
            _multiple_routes_cache[cache_key] = route_info
            
            return {
                "action": "route_selection_required",
                "data": {
                    "message": f"Found {route_info['count']} routes for {carrier} {flight_number} on {date_of_origin}. Please select a route:",
                    "available_routes": route_info["routes"],
                    "session_id": session_id,
                    "cache_key": cache_key,
                    "query": base_query
                }
            }
            
    except Exception as exc:
        logger.exception("Route selection check failed")
        return {
            "action": "error", 
            "data": response_error(f"Route selection failed: {str(exc)}", code=500)
        }

async def _fetch_one_async(query: dict, projection: dict) -> str:
    """
    Consistent async DB fetch and error handling.
    Returns JSON string response.
    """
    try:
        _, _, col = await get_mongodb_client()
        logger.info(f"Executing query: {json.dumps(query)}")
        
        result = await col.find_one(query, projection)
        
        if not result:
            logger.warning(f"No document found for query: {json.dumps(query)}")
            return response_error("No matching document found.", code=404)
        
        # Remove _id and _class to keep output clean
        if "_id" in result:
            result.pop("_id")
        if "_class" in result:
            result.pop("_class")
        
        logger.info(f"Query successful")
        return response_ok(result)
    except Exception as exc:
        logger.exception("DB query failed")
        return response_error(f"DB query failed: {str(exc)}", code=500)

def _extract_basic_info_from_document(doc: dict) -> dict:
    """Extract basic flight information from document"""
    flight_data = doc.get("flightLegState", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "suffix": flight_data.get("suffix"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "seqNumber": flight_data.get("seqNumber"),
        "startStation": flight_data.get("startStation"),
        "endStation": flight_data.get("endStation"),
        "startStationICAO": flight_data.get("startStationICAO"),
        "endStationICAO": flight_data.get("endStationICAO"),
        "scheduledStartTime": flight_data.get("scheduledStartTime"),
        "scheduledEndTime": flight_data.get("scheduledEndTime"),
        "flightStatus": flight_data.get("flightStatus"),
        "operationalStatus": flight_data.get("operationalStatus"),
        "flightType": flight_data.get("flightType"),
        "blockTimeSch": flight_data.get("blockTimeSch"),
        "blockTimeActual": flight_data.get("blockTimeActual"),
        "flightHoursActual": flight_data.get("flightHoursActual"),
        "isOTPFlight": flight_data.get("isOTPFlight"),
        "isOTPAchieved": flight_data.get("isOTPAchieved"),
        "isOTPConsidered": flight_data.get("isOTPConsidered"),
        "isOTTFlight": flight_data.get("isOTTFlight"),
        "isOTTAchievedFlight": flight_data.get("isOTTAchievedFlight"),
        "turnTimeFlightBeforeActual": flight_data.get("turnTimeFlightBeforeActual"),
        "turnTimeFlightBeforeSch": flight_data.get("turnTimeFlightBeforeSch")
    }

def _extract_operation_times_from_document(doc: dict) -> dict:
    """Extract operation times from document"""
    flight_data = doc.get("flightLegState", {})
    operation_data = flight_data.get("operation", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "startStation": flight_data.get("startStation"),
        "endStation": flight_data.get("endStation"),
        "scheduledStartTime": flight_data.get("scheduledStartTime"),
        "scheduledEndTime": flight_data.get("scheduledEndTime"),
        "startTimeOffset": flight_data.get("startTimeOffset"),
        "endTimeOffset": flight_data.get("endTimeOffset"),
        "operation": {
            "estimatedTimes": operation_data.get("estimatedTimes", {}),
            "actualTimes": operation_data.get("actualTimes", {})
        },
        "taxiOutTime": flight_data.get("taxiOutTime"),
        "taxiInTime": flight_data.get("taxiInTime"),
        "blockTimeSch": flight_data.get("blockTimeSch"),
        "blockTimeActual": flight_data.get("blockTimeActual"),
        "flightHoursActual": flight_data.get("flightHoursActual")
    }

def _extract_equipment_info_from_document(doc: dict) -> dict:
    """Extract equipment information from document"""
    flight_data = doc.get("flightLegState", {})
    equipment_data = flight_data.get("equipment", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "equipment": {
            "plannedAircraftType": equipment_data.get("plannedAircraftType"),
            "aircraft": equipment_data.get("aircraft"),
            "aircraftConfiguration": equipment_data.get("aircraftConfiguration"),
            "aircraftRegistration": equipment_data.get("aircraftRegistration"),
            "assignedAircraftTypeIATA": equipment_data.get("assignedAircraftTypeIATA"),
            "assignedAircraftTypeICAO": equipment_data.get("assignedAircraftTypeICAO"),
            "assignedAircraftTypeIndigo": equipment_data.get("assignedAircraftTypeIndigo"),
            "assignedAircraftConfiguration": equipment_data.get("assignedAircraftConfiguration"),
            "tailLock": equipment_data.get("tailLock"),
            "onwardFlight": equipment_data.get("onwardFlight"),
            "actualOnwardFlight": equipment_data.get("actualOnwardFlight")
        }
    }

def _extract_delay_summary_from_document(doc: dict) -> dict:
    """Extract delay summary from document"""
    flight_data = doc.get("flightLegState", {})
    operation_data = flight_data.get("operation", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "startStation": flight_data.get("startStation"),
        "endStation": flight_data.get("endStation"),
        "scheduledStartTime": flight_data.get("scheduledStartTime"),
        "operation": {
            "actualTimes": operation_data.get("actualTimes", {})  # ✅ Now includes offBlock
        },
        "delays": flight_data.get("delays")  # ✅ Now includes delays array
    }

def _extract_fuel_summary_from_document(doc: dict) -> dict:
    """Extract fuel summary from document"""
    flight_data = doc.get("flightLegState", {})
    operation_data = flight_data.get("operation", {})
    flight_plan_data = operation_data.get("flightPlan", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "startStation": flight_data.get("startStation"),
        "endStation": flight_data.get("endStation"),
        "operation": {
            "fuel": operation_data.get("fuel"),  # ✅ Now includes fuel data
            "flightPlan": {
                "offBlockFuel": flight_plan_data.get("offBlockFuel"),
                "takeoffFuel": flight_plan_data.get("takeoffFuel"),
                "landingFuel": flight_plan_data.get("landingFuel"),
                "holdFuel": flight_plan_data.get("holdFuel")
            }
        }
    }

def _extract_passenger_info_from_document(doc: dict) -> dict:
    """Extract passenger information from document"""
    flight_data = doc.get("flightLegState", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "pax": flight_data.get("pax")  # ✅ Now includes passenger data
    }

def _extract_crew_info_from_document(doc: dict) -> dict:
    """Extract crew information from document"""
    flight_data = doc.get("flightLegState", {})
    return {
        "carrier": flight_data.get("carrier"),
        "flightNumber": flight_data.get("flightNumber"),
        "dateOfOrigin": flight_data.get("dateOfOrigin"),
        "crewConnections": flight_data.get("crewConnections")  # ✅ Now includes crew data
    }

# --- MCP Tools ---

@mcp.tool()
async def health_check() -> str:
    """
    Simple health check for orchestrators and clients.
    Attempts a cheap DB ping.
    """
    try:
        _, _, col = await get_mongodb_client()
        doc = await col.find_one({}, {"_id": 1})
        return response_ok({"status": "ok", "db_connected": doc is not None})
    except Exception as e:
        logger.exception("Health check DB ping failed")
        return response_error("DB unreachable", code=503)

@mcp.tool()
async def select_route(session_id: str, start_station: str, end_station: str, carrier: str = "", flight_number: str = "", date_of_origin: str = "") -> str:
    """
    Select a specific route when multiple options are available.
    
    Args:
        session_id: Unique session identifier from the route selection prompt
        start_station: Selected departure station (e.g., "DEL")
        end_station: Selected arrival station (e.g., "BOM") 
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
    """
    try:
        # Normalize inputs
        fn = normalize_flight_number(flight_number) if flight_number else None
        dob = validate_date(date_of_origin) if date_of_origin else None
        
        # Find the cache key
        cache_key = f"{session_id}_{carrier}_{fn}_{dob}"
        route_info = _multiple_routes_cache.get(cache_key)
        
        if not route_info:
            return response_error("Route selection data not found. Please run the original query first.", code=404)
        
        # Find the selected route and corresponding document
        selected_route = None
        selected_document = None
        
        for i, route in enumerate(route_info["routes"]):
            if route["startStation"] == start_station and route["endStation"] == end_station:
                selected_route = route
                # Find the first document that matches this route
                for doc in route_info["documents"]:
                    flight_data = doc.get("flightLegState", {})
                    if (flight_data.get("startStation") == start_station and 
                        flight_data.get("endStation") == end_station):
                        selected_document = doc
                        break
                break
        
        if not selected_route:
            return response_error(f"Route {start_station} → {end_station} not found in available routes.", code=404)
        
        # Store the selected route and document for future queries
        _route_selections[session_id] = {
            "startStation": start_station,
            "endStation": end_station,
            "carrier": carrier,
            "flightNumber": fn,
            "dateOfOrigin": dob,
            "selected_document": selected_document  # ✅ Now contains FULL document
        }
        
        # Clean up cache
        if cache_key in _multiple_routes_cache:
            del _multiple_routes_cache[cache_key]
        
        return response_ok({
            "message": f"Route selected: {start_station} → {end_station}",
            "session_id": session_id,
            "selected_route": selected_route,
            "document_available": selected_document is not None
        })
        
    except Exception as exc:
        logger.exception("Route selection failed")
        return response_error(f"Route selection failed: {str(exc)}", code=500)

@mcp.tool()
async def get_flight_basic_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Fetch basic flight information including carrier, flight number, date, stations, times, and status.
    
    Args:
        carrier: Airline carrier code (e.g., "6E", "AI")
        flight_number: Flight number as string (e.g., "215")
        date_of_origin: Date in YYYY-MM-DD format (e.g., "2024-06-23")
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_flight_basic_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    # Normalize inputs
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date_of_origin format. Expected YYYY-MM-DD or common date formats", 400)
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract basic info from it
            basic_info = _extract_basic_info_from_document(route["selected_document"])
            return response_ok(basic_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                basic_info = _extract_basic_info_from_document(route["selected_document"])
                return response_ok(basic_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    # Project basic flight information
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.suffix": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.seqNumber": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.startStationICAO": 1,
        "flightLegState.endStationICAO": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.scheduledEndTime": 1,
        "flightLegState.flightStatus": 1,
        "flightLegState.operationalStatus": 1,
        "flightLegState.flightType": 1,
        "flightLegState.blockTimeSch": 1,
        "flightLegState.blockTimeActual": 1,
        "flightLegState.flightHoursActual": 1,
        "flightLegState.isOTPFlight": 1,
        "flightLegState.isOTPAchieved": 1,
        "flightLegState.isOTPConsidered": 1,
        "flightLegState.isOTTFlight": 1,
        "flightLegState.isOTTAchievedFlight": 1,
        "flightLegState.turnTimeFlightBeforeActual": 1,
        "flightLegState.turnTimeFlightBeforeSch": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_operation_times(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Return estimated and actual operation times for a flight including takeoff, landing, block times,StartTimeOffset, EndTimeOffset.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_operation_times: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    if date_of_origin and not dob:
        return response_error("Invalid date format.", 400)
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract operation times from it
            operation_info = _extract_operation_times_from_document(route["selected_document"])
            return response_ok(operation_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                operation_info = _extract_operation_times_from_document(route["selected_document"])
                return response_ok(operation_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.scheduledEndTime": 1,
        "flightLegState.startTimeOffset": 1,
        "flightLegState.endTimeOffset": 1,
        "flightLegState.operation.estimatedTimes": 1,
        "flightLegState.operation.actualTimes": 1,
        "flightLegState.taxiOutTime": 1,
        "flightLegState.taxiInTime": 1,
        "flightLegState.blockTimeSch": 1,
        "flightLegState.blockTimeActual": 1,
        "flightLegState.flightHoursActual": 1,
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_equipment_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Get aircraft equipment details including aircraft type, registration (tail number), and configuration.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_equipment_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract equipment info from it
            equipment_info = _extract_equipment_info_from_document(route["selected_document"])
            return response_ok(equipment_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                equipment_info = _extract_equipment_info_from_document(route["selected_document"])
                return response_ok(equipment_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.equipment.plannedAircraftType": 1,
        "flightLegState.equipment.aircraft": 1,
        "flightLegState.equipment.aircraftConfiguration": 1,
        "flightLegState.equipment.aircraftRegistration": 1,
        "flightLegState.equipment.assignedAircraftTypeIATA": 1,
        "flightLegState.equipment.assignedAircraftTypeICAO": 1,
        "flightLegState.equipment.assignedAircraftTypeIndigo": 1,
        "flightLegState.equipment.assignedAircraftConfiguration": 1,
        "flightLegState.equipment.tailLock": 1,
        "flightLegState.equipment.onwardFlight": 1,
        "flightLegState.equipment.actualOnwardFlight": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_delay_summary(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Summarize delay reasons, durations, and total delay time for a specific flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_delay_summary: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract delay summary from it
            delay_info = _extract_delay_summary_from_document(route["selected_document"])
            return response_ok(delay_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                delay_info = _extract_delay_summary_from_document(route["selected_document"])
                return response_ok(delay_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.scheduledStartTime": 1,
        "flightLegState.operation.actualTimes.offBlock": 1,
        "flightLegState.delays": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_fuel_summary(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Retrieve fuel summary including planned vs actual fuel for takeoff, landing, and total consumption.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_fuel_summary: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract fuel summary from it
            fuel_info = _extract_fuel_summary_from_document(route["selected_document"])
            return response_ok(fuel_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                fuel_info = _extract_fuel_summary_from_document(route["selected_document"])
                return response_ok(fuel_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.startStation": 1,
        "flightLegState.endStation": 1,
        "flightLegState.operation.fuel": 1,
        "flightLegState.operation.flightPlan.offBlockFuel": 1,
        "flightLegState.operation.flightPlan.takeoffFuel": 1,
        "flightLegState.operation.flightPlan.landingFuel": 1,
        "flightLegState.operation.flightPlan.holdFuel": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_passenger_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Get passenger count and connection information for the flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_passenger_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract passenger info from it
            passenger_info = _extract_passenger_info_from_document(route["selected_document"])
            return response_ok(passenger_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                passenger_info = _extract_passenger_info_from_document(route["selected_document"])
                return response_ok(passenger_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.pax": 1
    }
    
    return await _fetch_one_async(query, projection)

@mcp.tool()
async def get_crew_info(carrier: str = "", flight_number: str = "", date_of_origin: str = "", session_id: str = "default") -> str:
    """
    Get crew connections and details for the flight.
    
    Args:
        carrier: Airline carrier code
        flight_number: Flight number as string
        date_of_origin: Date in YYYY-MM-DD format
        session_id: Unique session identifier for route selection tracking
    """
    logger.info(f"get_crew_info: carrier={carrier}, flight_number={flight_number}, date={date_of_origin}, session_id={session_id}")
    
    fn = normalize_flight_number(flight_number) if flight_number else None
    dob = validate_date(date_of_origin) if date_of_origin else None
    
    # Check if we already have a route selection for this session
    if session_id in _route_selections:
        route = _route_selections[session_id]
        # Use the stored document if available, otherwise query with route
        if "selected_document" in route and route["selected_document"]:
            # We have the actual document - extract crew info from it
            crew_info = _extract_crew_info_from_document(route["selected_document"])
            return response_ok(crew_info)
        else:
            # Fallback: query with route selection
            query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    else:
        # Check route selection
        route_check = await _check_route_selection(session_id, carrier, fn, dob)
        
        if route_check["action"] == "error":
            return route_check["data"]
        elif route_check["action"] == "route_selection_required":
            # ✅ UPDATED: Use standardized route selection response
            return response_route_selection_required(route_check["data"])
        elif route_check["action"] == "proceed":
            # Single route found - use the stored selection
            route = _route_selections[session_id]
            if "selected_document" in route and route["selected_document"]:
                crew_info = _extract_crew_info_from_document(route["selected_document"])
                return response_ok(crew_info)
            else:
                query = make_query(carrier, fn, dob, route["startStation"], route["endStation"])
    
    projection = {
        "flightLegState.carrier": 1,
        "flightLegState.flightNumber": 1,
        "flightLegState.dateOfOrigin": 1,
        "flightLegState.crewConnections": 1
    }
    
    return await _fetch_one_async(query, projection)

# The following tools DON'T use route selection logic:

@mcp.tool()
async def raw_mongodb_query(query_json: str, projection: str = "", limit: int = 10) -> str:
    """
    Execute a raw MongoDB query (stringified JSON) with optional projection.

    Supports intelligent LLM-decided projections to reduce payload size based on query intent.

    Args:
        query_json: The MongoDB query (as stringified JSON).
        projection: Optional projection (as stringified JSON) for selecting fields.
        limit: Max number of documents to return (default 10, capped at 50).
    """

    def _safe_json_loads(text: str) -> dict:
        """Safely parse JSON, handling single quotes and formatting errors."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                fixed = text.replace("'", '"')
                return json.loads(fixed)
            except Exception as e:
                raise ValueError(f"Invalid JSON: {e}")

    try:
        _, _, col = await get_mongodb_client()

        # --- Parse Query ---
        try:
            query = _safe_json_loads(query_json)
        except ValueError as e:
            return response_error(f"❌ Invalid query_json: {str(e)}", 400)

        # --- Parse Projection (optional) ---
        projection_dict = None
        if projection:
            try:
                projection_dict = _safe_json_loads(projection)
            except ValueError as e:
                return response_error(f"❌ Invalid projection JSON: {str(e)}", 400)

        # --- Validate types ---
        if not isinstance(query, dict):
            return response_error("❌ query_json must be a JSON object.", 400)
        if projection_dict and not isinstance(projection_dict, dict):
            return response_error("❌ projection must be a JSON object.", 400)

        # --- Safety guard ---
        forbidden_ops = ["$where", "$out", "$merge", "$accumulator", "$function"]
        for key in query.keys():
            if key in forbidden_ops or key.startswith("$"):
                return response_error(f"❌ Operator '{key}' is not allowed.", 400)

        limit = min(max(1, int(limit)), 50)

        # --- Fallback projection ---
        # If the LLM forgets to include projection, return a minimal safe set.
        if not projection_dict:
            projection_dict = {
                "_id": 0,
                "flightLegState.carrier": 1,
                "flightLegState.flightNumber": 1,
                "flightLegState.dateOfOrigin": 1
            }
            
        logger.info(f"Executing MongoDB query: {query} | projection={projection_dict} | limit={limit}")

        # --- Run query ---
        cursor = col.find(query, projection_dict).sort("flightLegState.dateOfOrigin", -1).limit(limit)
        docs = []
        async for doc in cursor:
            doc.pop("_id", None)
            doc.pop("_class", None)
            docs.append(doc)

        if not docs:
            return response_error("No documents found for the given query.", 404)

        return response_ok({
            "count": len(docs),
            "query": query,
            "projection": projection_dict,
            "documents": docs
        })

    except Exception as exc:
        logger.exception("❌ raw_mongodb_query failed")
        return response_error(f"Raw MongoDB query failed: {str(exc)}", 500)

@mcp.tool()
async def run_aggregated_query(
    query_type: str = "",
    carrier: str = "",
    field: str = "",
    start_date: str = "",
    end_date: str = "",
    filter_json: str = ""
) -> str:
    """
    Run statistical or comparative MongoDB aggregation queries.

    Args:
        query_type: "average", "sum", "min", "max", "count".
        carrier: Optional carrier filter.
        field: Field to aggregate, e.g. "flightLegState.pax.passengerCount.count".
        start_date: Optional start date (YYYY-MM-DD).
        end_date: Optional end date (YYYY-MM-DD).
        filter_json: Optional filter query (as JSON string).
    """

    _, _, col = await get_mongodb_client()

    match_stage = {}

    # --- Optional filters ---
    if filter_json:
        try:
            match_stage.update(json.loads(filter_json.replace("'", '"')))
        except Exception as e:
            return response_error(f"Invalid filter_json: {e}", 400)

    if carrier:
        match_stage["flightLegState.carrier"] = carrier
    if start_date and end_date:
        match_stage["flightLegState.dateOfOrigin"] = {"$gte": start_date, "$lte": end_date}

    agg_map = {
        "average": {"$avg": f"${field}"},
        "sum": {"$sum": f"${field}"},
        "min": {"$min": f"${field}"},
        "max": {"$max": f"${field}"},
        "count": {"$sum": 1},
    }

    if query_type not in agg_map:
        return response_error(f"Unsupported query_type '{query_type}'. Use one of: average, sum, min, max, count", 400)

    pipeline = [{"$match": match_stage}, {"$group": {"_id": None, "value": agg_map[query_type]}}]

    try:
        logger.info(f"Running aggregation pipeline: {pipeline}")
        docs = await col.aggregate(pipeline).to_list(length=10)
        return response_ok({"pipeline": pipeline, "results": docs})
    except Exception as e:
        logger.exception("Aggregation query failed")
        return response_error(f"Aggregation failed: {str(e)}", 500)

@mcp.tool()
async def convert_utc_to_local_time(utc_time_str: str, timezone_str: str = "Asia/Kolkata") -> str:
    """
    Convert UTC datetime to local time (default: IST = UTC+5:30).
    
    Args:
        utc_time_str: UTC time in 'YYYY-MM-DD HH:MM' or ISO format
        timezone_str: Timezone (default: Asia/Kolkata for IST)
    
    Returns:
        JSON with UTC time, local time, and applied offset
    """
    try:
        # Parse UTC time
        formats = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        utc_dt = None
        for fmt in formats:
            try:
                utc_dt = datetime.strptime(utc_time_str, fmt)
                break
            except ValueError:
                continue
        
        if not utc_dt:
            return response_error("Invalid UTC time format. Use 'YYYY-MM-DD HH:MM' or ISO format.")

        # Manual conversion for IST (+5:30)
        if timezone_str == "Asia/Kolkata":
            local_dt = utc_dt + timedelta(hours=5, minutes=30)
            offset = "+05:30"
        else:
            # For other timezones, you can add more logic
            local_dt = utc_dt
            offset = "+00:00"

        return response_ok({
            "utc_time": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "local_time": local_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "offset_applied": offset,
            "timezone": timezone_str
        })

    except Exception as e:
        logger.exception("Time conversion failed")
        return response_error(f"Time conversion failed: {str(e)}")

# --- Run MCP Server ---
if __name__ == "__main__":
    logger.info("Starting FlightOps MCP Server on %s:%s (transport=%s)", HOST, PORT, TRANSPORT)
    logger.info("MongoDB URL: %s, Database: %s, Collection: %s", MONGODB_URL, DATABASE_NAME, COLLECTION_NAME)
    mcp.run(transport="streamable-http")
