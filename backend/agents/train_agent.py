import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch

from state.trip_state import GraphState
from prompts.agent_prompts import TRAIN_AGENT_PROMPT
from core.tools.train_tools import search_trains

def run_train_agent(state: GraphState) -> dict:
    print("🚆 Train Agent: Researching routes, exact railway stations, and IRCTC schedules...")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    trip_data = state["trip_data"]

    origin_code = trip_data.origin_iata or "SC" # Re-using field for train code mapping
    arrival_code = trip_data.arrival_iata or "ERS"
    return_code = trip_data.return_departure_iata or arrival_code

    start_date = trip_data.start_date or "Upcoming Date"
    return_date = trip_data.return_date or "Return Date"
    budget = trip_data.budget_tier or "standard"
    travelers = trip_data.number_of_travelers or 1

    entry_halt = trip_data.entry_halt or (trip_data.halts[0] if trip_data.halts else trip_data.destination)
    exit_halt = trip_data.exit_halt or (trip_data.halts[-1] if trip_data.halts else trip_data.destination)
    
    outbound_last_mile = trip_data.outbound_last_mile_note or "Direct station access"
    return_last_mile = trip_data.return_last_mile_note or "Direct station access"

    tools = [search_trains]
    system_prompt = TRAIN_AGENT_PROMPT.format(
        start_date=start_date, return_date=return_date,
        origin_code=origin_code, arrival_code=arrival_code, return_code=return_code,
        entry_halt=entry_halt, exit_halt=exit_halt,
        outbound_last_mile=outbound_last_mile, return_last_mile=return_last_mile,
        budget=budget, travelers=travelers
    )

    react_agent = create_react_agent(llm, tools, prompt=system_prompt)
    task_instructions = (
        f"Find complete round-trip IRCTC trains for {travelers} travelers:\n"
        f"1. Outbound on {start_date}: {origin_code} -> {arrival_code}.\n"
        f"2. Return on {return_date}: {return_code} -> {origin_code}.\n"
        f"3. Quote the Google Maps road transit times for last-mile segments exactly as provided."
    )

    react_state = react_agent.invoke({"messages": [HumanMessage(content=task_instructions)]})
    
    raw_content = react_state["messages"][-1].content
    final_response = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content]) if isinstance(raw_content, list) else str(raw_content)

    return {"messages": [final_response]}