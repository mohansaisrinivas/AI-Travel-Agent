import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

from state.trip_state import GraphState
from prompts.agent_prompts import BUS_AGENT_PROMPT
from core.tools.bus_tools import search_buses

def run_bus_agent(state: GraphState) -> dict:
    print("🚌 Bus Agent: Researching routes, exact boarding points, and operator reviews...")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    trip_data = state["trip_data"]

    origin_city = trip_data.origin_city or "Origin"
    boarding_area = trip_data.origin_boarding_area or "Main Bus Stand"
    drop_area = trip_data.destination_drop_area or trip_data.halts[0] if trip_data.halts else trip_data.destination

    start_date = trip_data.start_date or "Upcoming Date"
    return_date = trip_data.return_date or "Return Date"
    budget = trip_data.budget_tier or "standard"
    travelers = trip_data.number_of_travelers or 1
    entry_halt = trip_data.entry_halt or (trip_data.halts[0] if trip_data.halts else trip_data.destination)
    exit_halt = trip_data.exit_halt or (trip_data.halts[-1] if trip_data.halts else trip_data.destination)
    outbound_last_mile = trip_data.outbound_last_mile_note or "Direct bus access"
    return_last_mile = trip_data.return_last_mile_note or "Direct bus access"
    special_notes = trip_data.special_transport_notes or "None"

    tools = [search_buses]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearch(max_results=3))

    system_prompt = BUS_AGENT_PROMPT.format(
        start_date=start_date,
        return_date=return_date,
        origin_city=origin_city,
        boarding_area=boarding_area,
        drop_area=drop_area,
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
        f"Find complete round-trip buses for a {budget} budget for {travelers} travelers:\n"
        f"1. Outbound on {start_date}: {origin_city} (boarding at {boarding_area}) -> {drop_area}.\n"
        f"2. Return on {return_date}: {drop_area} -> {origin_city}.\n"
        f"3. Check Tavily for operator safety, hygiene, and delay history.\n"
        f"4. Present the round-trip recommendations including the total price (multiply by {travelers}) and the generated Google Maps links."
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