import os
from datetime import datetime, timedelta
import dateutil.parser
from typing import Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state.trip_state import GraphState
from prompts.agent_prompts import TRANSPORT_ORCHESTRATOR_PROMPT
from core.tools.airport_tools import resolve_multi_halt_airports
from core.tools.bus_tools import resolve_bus_route
from core.tools.train_tools import resolve_multi_halt_train_stations

class TransportOrchestrationPlan(BaseModel):
    selected_agent: str = Field(description="'flight_agent', 'train_agent', 'bus_agent', or 'car_agent'")
    special_requirements: str = Field(description="Summary of NEW constraints like non-stop, departure times, extra baggage. Or 'None'.")
    manual_origin_override: Optional[str] = Field(
        default=None, 
        description="Populate ONLY if the user explicitly demands a specific AIRPORT or RAILWAY STATION (e.g., 'search from Secunderabad'). DO NOT populate for neighborhoods like IDPL."
    )
    manual_destination_override: Optional[str] = Field(
        default=None, 
        description="Populate ONLY if the user explicitly demands a specific AIRPORT or RAILWAY STATION to arrive at. Otherwise leave null."
    )
    reasoning: str = Field(description="Explanation of routing and logistics plan.")

def calculate_return_date(start_date_str: str, duration_days: int) -> str:
    if not start_date_str or start_date_str.upper() == "TBD":
        return "TBD"
    try:
        dt = dateutil.parser.parse(start_date_str)
        return (dt + timedelta(days=duration_days)).strftime("%Y-%m-%d")
    except Exception:
        return "TBD"

def run_transport_orchestrator(state: GraphState) -> dict:
    print("🧠 Transport Orchestrator: Analyzing logistics and routing...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(TransportOrchestrationPlan)

    trip_data = state["trip_data"]
    state_updates = {}

    mode = (trip_data.transport_mode or "flight").lower()

    if trip_data.start_date and (not trip_data.return_date or trip_data.return_date == "TBD"):
        days = trip_data.duration_days or 3
        state_updates["return_date"] = calculate_return_date(trip_data.start_date, days)

    halt_names = [h.halt_name if hasattr(h, "halt_name") else str(h) for h in trip_data.halts] if trip_data.halts else [trip_data.destination]
    entry_halt_str = halt_names[0]
    exit_halt_str = halt_names[-1]
    
    state_updates["entry_halt"] = entry_halt_str
    state_updates["exit_halt"] = exit_halt_str

    recent_messages = state["messages"][-4:]
    context_str = "\n".join([getattr(m, 'content', str(m)) for m in recent_messages if not str(m).startswith("SYSTEM_NOTE:")])

    temp_state_dict = {**trip_data.dict(), **state_updates}
    state_summary = f"Trip Data: {temp_state_dict}"
    prompt = TRANSPORT_ORCHESTRATOR_PROMPT.format(trip_state=state_summary)
    
    plan = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Recent Conversation:\n{context_str}")
    ])

    new_notes = plan.special_requirements.strip()
    if new_notes.lower() not in ["none", "n/a", "", "no special requirements"]:
        current_notes = trip_data.special_transport_notes or ""
        state_updates["special_transport_notes"] = f"{current_notes} | {new_notes}" if current_notes else new_notes

    # --- MANUAL OVERRIDE INTERCEPTION ---
    if plan.manual_origin_override:
        override_src = plan.manual_origin_override.strip().upper()
        print(f"   [Transport Orchestrator] Applying user manual origin override: {override_src}")
        state_updates["origin_iata"] = override_src

    if plan.manual_destination_override:
        override_dst = plan.manual_destination_override.strip().upper()
        print(f"   [Transport Orchestrator] Applying user manual destination override: {override_dst}")
        state_updates["arrival_iata"] = override_dst
        state_updates["return_departure_iata"] = override_dst

    # --- DYNAMIC HUB RESOLUTION ---
    current_origin_code = state_updates.get("origin_iata", trip_data.origin_iata)
    current_arrival_code = state_updates.get("arrival_iata", trip_data.arrival_iata)

    if mode == "flight" and trip_data.origin_city:
        if not current_origin_code or not current_arrival_code:
            route_info = resolve_multi_halt_airports(trip_data.origin_city, entry_halt_str, exit_halt_str)
            state_updates.setdefault("origin_iata", route_info.origin_iata)
            state_updates.setdefault("arrival_iata", route_info.arrival_iata)
            state_updates["arrival_airport_name"] = route_info.arrival_airport_name
            state_updates["outbound_last_mile_note"] = route_info.outbound_last_mile_note
            state_updates.setdefault("return_departure_iata", route_info.return_departure_iata)
            state_updates["return_departure_airport_name"] = route_info.return_departure_airport_name
            state_updates["return_last_mile_note"] = route_info.return_last_mile_note
            state_updates["is_direct_flight_available"] = route_info.is_direct_flight_available

    elif mode == "bus" and trip_data.origin_city:
        if not current_origin_code or not current_arrival_code:
            print(f"   [Transport Orchestrator] Resolving bus hubs: Halt 1 ({entry_halt_str}) & Final Halt ({exit_halt_str})...")
            out_route = resolve_bus_route(trip_data.origin_city, entry_halt_str)
            state_updates["destination_drop_area"] = out_route.destination_drop_area
            state_updates["outbound_last_mile_note"] = out_route.last_mile_note
            ret_route = resolve_bus_route(exit_halt_str, trip_data.origin_city)
            state_updates["return_last_mile_note"] = ret_route.last_mile_note

    elif mode == "train" and trip_data.origin_city:
        if not current_origin_code or not current_arrival_code:
            doorstep_origin = f"{trip_data.origin_boarding_area}, {trip_data.origin_city}" if trip_data.origin_boarding_area else trip_data.origin_city
            print(f"   [Transport Orchestrator] Resolving train hubs for {doorstep_origin}...")
            route_info = resolve_multi_halt_train_stations(doorstep_origin, entry_halt_str, exit_halt_str)
            state_updates.setdefault("origin_iata", route_info.origin_station_code)
            state_updates.setdefault("arrival_iata", route_info.arrival_station_code)
            state_updates["outbound_last_mile_note"] = route_info.outbound_last_mile_note
            state_updates.setdefault("return_departure_iata", route_info.return_departure_station_code)
            state_updates["return_last_mile_note"] = route_info.return_last_mile_note

    print(f"🔀 Commute Router: Delegating to {plan.selected_agent}.")
    
    updated_trip_data = trip_data.model_copy(update=state_updates)
    
    return {
        "messages": [f"SYSTEM_NOTE: Routing commute to {plan.selected_agent}"],
        "trip_data": updated_trip_data
    }