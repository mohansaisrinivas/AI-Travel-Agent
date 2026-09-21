from datetime import datetime, timedelta
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
    reasoning: str = Field(description="Explanation of routing and logistics plan.")

def calculate_return_date(start_date_str: str, duration_days: int) -> str:
    if not start_date_str:
        return "TBD"
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(start_date_str.strip(), fmt)
            return (dt + timedelta(days=duration_days)).strftime("%Y-%m-%d")
        except ValueError:
            continue
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    res = llm.invoke(f"Calculate the return date given Start Date: '{start_date_str}' and Duration: {duration_days} days. Return ONLY the date string (e.g. 'YYYY-MM-DD').")
    return res.content.strip()

def run_transport_orchestrator(state: GraphState) -> dict:
    print("🧠 Transport Orchestrator: Analyzing logistics and routing...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(TransportOrchestrationPlan)

    trip_data = state["trip_data"]
    mode = (trip_data.transport_mode or "flight").lower()

    if trip_data.start_date and (not trip_data.return_date or trip_data.return_date == "TBD"):
        days = trip_data.duration_days or 3
        trip_data.return_date = calculate_return_date(trip_data.start_date, days)

    # 1. Routing for FLIGHTS
    if mode == "flight" and trip_data.origin_city:
        halts = trip_data.halts or [trip_data.destination]
        trip_data.entry_halt = halts[0]
        trip_data.exit_halt = halts[-1]

        route_info = resolve_multi_halt_airports(trip_data.origin_city, trip_data.entry_halt, trip_data.exit_halt)

        trip_data.origin_iata = route_info.origin_iata
        trip_data.arrival_iata = route_info.arrival_iata
        trip_data.arrival_airport_name = route_info.arrival_airport_name
        trip_data.outbound_last_mile_note = route_info.outbound_last_mile_note
        trip_data.return_departure_iata = route_info.return_departure_iata
        trip_data.return_departure_airport_name = route_info.return_departure_airport_name
        trip_data.return_last_mile_note = route_info.return_last_mile_note
        trip_data.is_direct_flight_available = route_info.is_direct_flight_available

    # 2. Routing for BUSES
    elif mode == "bus" and trip_data.origin_city:
        halts = trip_data.halts or [trip_data.destination]
        trip_data.entry_halt = halts[0]
        trip_data.exit_halt = halts[-1]

        print(f"   [Transport Orchestrator] Resolving bus hubs: Halt 1 ({trip_data.entry_halt}) & Final Halt ({trip_data.exit_halt})...")
        
        out_route = resolve_bus_route(trip_data.origin_city, trip_data.entry_halt)
        trip_data.destination_drop_area = out_route.destination_drop_area
        trip_data.outbound_last_mile_note = out_route.last_mile_note
        
        ret_route = resolve_bus_route(trip_data.exit_halt, trip_data.origin_city)
        trip_data.return_last_mile_note = ret_route.last_mile_note

    # 3. Routing for TRAINS (Fully Upgraded with Doorstep Micro-Routing)
    elif mode == "train" and trip_data.origin_city:
        halts = trip_data.halts or [trip_data.destination]
        trip_data.entry_halt = halts[0]
        trip_data.exit_halt = halts[-1]
        
        # STITCH THE DOORSTEP LOCATION TOGETHER: e.g., "IDPL, Hyderabad"
        doorstep_origin = trip_data.origin_city
        if trip_data.origin_boarding_area:
            doorstep_origin = f"{trip_data.origin_boarding_area}, {trip_data.origin_city}"

        print(f"   [Transport Orchestrator] Resolving train hubs for {doorstep_origin}...")
        
        route_info = resolve_multi_halt_train_stations(doorstep_origin, trip_data.entry_halt, trip_data.exit_halt)

        # We reuse the flight IATA fields to pass the IRCTC station codes cleanly to the Train Agent
        trip_data.origin_iata = route_info.origin_station_code
        trip_data.arrival_iata = route_info.arrival_station_code
        trip_data.outbound_last_mile_note = route_info.outbound_last_mile_note
        trip_data.return_departure_iata = route_info.return_departure_station_code
        trip_data.return_last_mile_note = route_info.return_last_mile_note

    # Synthesize Context & Delegate to Specific Agent
    recent_messages = state["messages"][-4:]
    context_str = "\n".join([getattr(m, 'content', str(m)) for m in recent_messages if not str(m).startswith("SYSTEM_NOTE:")])

    state_summary = f"Trip Data: {trip_data.dict()}"
    prompt = TRANSPORT_ORCHESTRATOR_PROMPT.format(trip_state=state_summary)
    
    plan = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Recent Conversation:\n{context_str}")
    ])

    new_notes = plan.special_requirements.strip()
    if new_notes.lower() not in ["none", "n/a", "", "no special requirements"]:
        current_notes = trip_data.special_transport_notes or ""
        if new_notes.lower() not in current_notes.lower():
            if current_notes:
                trip_data.special_transport_notes = f"{current_notes} | {new_notes}"
            else:
                trip_data.special_transport_notes = new_notes

    print(f"🔀 Commute Router: Delegating to {plan.selected_agent}.")
    return {
        "messages": [f"SYSTEM_NOTE: Routing commute to {plan.selected_agent}"],
        "trip_data": trip_data
    }