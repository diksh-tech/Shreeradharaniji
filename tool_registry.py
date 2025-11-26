# tool_registry.py

TOOLS = {
    "health_check": {
        "args": [],
        "desc": "Simple health check for orchestrators and clients. Attempts a cheap DB ping.",
    },
    "select_route": {
        "args": ["session_id", "start_station", "end_station", "carrier", "flight_number", "date_of_origin"],
        "desc": "Select a specific route when multiple options are available. Used internally by the system.",
    },
    "get_flight_basic_info": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Fetch basic flight information. Automatically handles route selection when multiple flights match.",
    },
    "get_operation_times": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Return estimated and actual operation times. Automatically handles route selection.",
    },
    "get_equipment_info": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Get aircraft equipment details. Automatically handles route selection.",
    },
    "get_delay_summary": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Summarize delay reasons and durations. Automatically handles route selection.",
    },
    "get_fuel_summary": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Retrieve fuel summary information. Automatically handles route selection.",
    },
    "get_passenger_info": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Get passenger count and connection information. Automatically handles route selection.",
    },
    "get_crew_info": {
        "args": ["carrier", "flight_number", "date_of_origin", "session_id"],
        "desc": "Get crew connections and details. Automatically handles route selection.",
    },
    "raw_mongodb_query": {
        "args": ["query_json", "projection", "limit"],
        "desc": "Execute a raw MongoDB query (stringified JSON) with optional projection.",
    },
    "run_aggregated_query": {
        "args": ["query_type", "carrier", "field", "start_date", "end_date", "filter_json"],
        "desc": "Run statistical or comparative MongoDB aggregation queries.",
    },
    "convert_utc_to_local_time": {
        "args": ["utc_time_str", "timezone_str"],
        "desc": "Convert UTC time to local time. Defaults to IST (UTC+5:30).",
    },
}
