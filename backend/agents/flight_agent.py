import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

from state.trip_state import GraphState
from prompts.agent_prompts import FLIGHT_AGENT_PROMPT
from core.tools.flight_tools import search_flights

def run_flight_agent(state: GraphState) -> dict:
    print("✈️ Flight Agent: Researching round-trip flights aligned to Halt 1 and Final Halt...")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    trip_data = state["trip_data"]

    origin_city = trip_data.origin_city or "Origin"
    origin_iata = trip_data.origin_iata or "HYD"
    arrival_iata = trip_data.arrival_iata or "COK"
    arrival_airport = trip_data.arrival_airport_name or "Arrival Airport"
    return_departure_iata = trip_data.return_departure_iata or arrival_iata
    return_airport = trip_data.return_departure_airport_name or arrival_airport

    start_date = trip_data.start_date or "Upcoming Date"
    return_date = trip_data.return_date or "Return Date"
    budget = trip_data.budget_tier or "standard"
    travelers = trip_data.number_of_travelers or 1  # Ensures math defaults to at least 1 person
    entry_halt = trip_data.entry_halt or (trip_data.halts[0] if trip_data.halts else trip_data.destination)
    exit_halt = trip_data.exit_halt or (trip_data.halts[-1] if trip_data.halts else trip_data.destination)
    outbound_last_mile = trip_data.outbound_last_mile_note or "Direct airport access"
    return_last_mile = trip_data.return_last_mile_note or "Direct airport access"
    special_notes = trip_data.special_transport_notes or "None"

    tools = [search_flights]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearch(max_results=3))

    system_prompt = FLIGHT_AGENT_PROMPT.format(
        start_date=start_date,
        return_date=return_date,
        origin_city=origin_city,
        origin_iata=origin_iata,
        arrival_iata=arrival_iata,
        arrival_airport=arrival_airport,
        return_departure_iata=return_departure_iata,
        return_airport=return_airport,
        entry_halt=entry_halt,
        exit_halt=exit_halt,
        outbound_last_mile=outbound_last_mile,
        return_last_mile=return_last_mile,
        budget=budget,
        travelers=travelers,
        special_notes=special_notes
    )

    react_agent = create_react_agent(llm, tools, prompt=system_prompt)

    task_instructions = (
        f"Find complete round-trip flights for a {budget} budget for {travelers} travelers:\n"
        f"1. Outbound on {start_date}: {origin_iata} -> {arrival_iata} (closest to Halt 1: {entry_halt}).\n"
        f"2. Return on {return_date}: {return_departure_iata} -> {origin_iata} (closest to Final Halt: {exit_halt}).\n"
        f"3. Check Tavily for delay history and hospitality reviews for candidate airlines.\n"
        f"4. Present the complete round-trip recommendations including road transit times for last-mile segments and the combined total price."
    )

    react_state = react_agent.invoke({
        "messages": [HumanMessage(content=task_instructions)]
    })

    raw_content = react_state["messages"][-1].content
    if isinstance(raw_content, list):
        final_response = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content])
    else:
        final_response = str(raw_content)

    return {"messages": [final_response]}