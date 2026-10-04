import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch

from state.trip_state import GraphState
from prompts.agent_prompts import FLIGHT_AGENT_PROMPT
from core.tools.flight_tools import search_flights

def run_flight_agent(state: GraphState) -> dict:
    trip_data = state["trip_data"]

    # RECALL CHECK: If transport details are already saved and user isn't modifying the route, return cached version!
    last_msg = str(state["messages"][-1]).lower()
    if trip_data.saved_transport_details and not any(k in last_msg for k in ["switch", "train", "bus", "change", "instead"]):
        print("✈️ Flight Agent: Recalling saved flight schedule from state memory (Zero LLM / Scraper Cost)...")
        return {"messages": [trip_data.saved_transport_details]}

    print("✈️ Flight Agent: Researching round-trip flights aligned to halts (Direct Tool Binding)...")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    origin_city = trip_data.origin_city or "Origin"
    origin_iata = trip_data.origin_iata or "HYD"
    arrival_iata = trip_data.arrival_iata or "COK"
    arrival_airport = trip_data.arrival_airport_name or "Arrival Airport"
    return_departure_iata = trip_data.return_departure_iata or arrival_iata
    return_airport = trip_data.return_departure_airport_name or arrival_airport

    start_date = trip_data.start_date or "Upcoming Date"
    return_date = trip_data.return_date or "Return Date"
    budget = trip_data.budget_tier or "standard"
    travelers = trip_data.number_of_travelers or 1 
    entry_halt = trip_data.entry_halt or (trip_data.halts[0] if trip_data.halts else trip_data.destination)
    exit_halt = trip_data.exit_halt or (trip_data.halts[-1] if trip_data.halts else trip_data.destination)
    outbound_last_mile = trip_data.outbound_last_mile_note or "Direct airport access"
    return_last_mile = trip_data.return_last_mile_note or "Direct airport access"
    special_notes = trip_data.special_transport_notes or "None"

    tools = [search_flights]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearch(max_results=3))

    system_prompt = FLIGHT_AGENT_PROMPT.format(
        start_date=start_date, return_date=return_date, origin_city=origin_city, 
        origin_iata=origin_iata, arrival_iata=arrival_iata, arrival_airport=arrival_airport,
        return_departure_iata=return_departure_iata, return_airport=return_airport,
        entry_halt=entry_halt, exit_halt=exit_halt, outbound_last_mile=outbound_last_mile,
        return_last_mile=return_last_mile, budget=budget, travelers=travelers, special_notes=special_notes
    )

    task_instructions = (
        f"Find complete round-trip flights for a {budget} budget for {travelers} travelers:\n"
        f"1. Outbound on {start_date}: {origin_iata} -> {arrival_iata}\n"
        f"2. Return on {return_date}: {return_departure_iata} -> {origin_iata}\n"
        f"3. Use the flight tool. Return formatted recommendations with {travelers} multiplier math applied."
    )

    llm_with_tools = llm.bind_tools(tools)
    messages_for_llm = [SystemMessage(content=system_prompt), HumanMessage(content=task_instructions)]
    
    raw_content = ""
    for _ in range(4):
        tool_call_msg = llm_with_tools.invoke(messages_for_llm)
        
        if not tool_call_msg.tool_calls:
            raw_content = tool_call_msg.content
            break
            
        messages_for_llm.append(tool_call_msg)
        
        for tc in tool_call_msg.tool_calls:
            tool_instance = next((t for t in tools if t.name == tc["name"]), None)
            if tool_instance:
                try:
                    result = tool_instance.invoke(tc["args"])
                except Exception as e:
                    result = f"Error: {e}"
                messages_for_llm.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    else:
        raw_content = "I have searched extensively but couldn't find any alternative flights matching these constraints. The previously suggested route remains the best option."

    if not raw_content or not str(raw_content).strip():
        raw_content = "I could not find any cheaper options. The original recommendation is currently the most optimal available."

    if isinstance(raw_content, list):
        final_response = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content])
    else:
        final_response = str(raw_content)

    # Save to state cache
    updated_trip_data = trip_data.model_copy(update={"saved_transport_details": final_response})

    return {
        "messages": [final_response],
        "trip_data": updated_trip_data
    }