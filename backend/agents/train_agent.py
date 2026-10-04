import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from state.trip_state import GraphState
from prompts.agent_prompts import TRAIN_AGENT_PROMPT
from core.tools.train_tools import search_trains

def run_train_agent(state: GraphState) -> dict:
    trip_data = state["trip_data"]

    # RECALL CHECK: If transport details are already saved and user isn't modifying the route, return cached version!
    last_msg = str(state["messages"][-1]).lower()
    if trip_data.saved_transport_details and not any(k in last_msg for k in ["switch", "flight", "bus", "change", "instead"]):
        print("🚆 Train Agent: Recalling saved train schedule from state memory (Zero LLM / Scraper Cost)...")
        return {"messages": [trip_data.saved_transport_details]}

    print("🚆 Train Agent: Researching exact railway stations (Direct Tool Binding)...")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    origin_code = trip_data.origin_iata or "SC" 
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
        start_date=start_date, return_date=return_date, origin_code=origin_code, 
        arrival_code=arrival_code, return_code=return_code, entry_halt=entry_halt, 
        exit_halt=exit_halt, outbound_last_mile=outbound_last_mile, 
        return_last_mile=return_last_mile, budget=budget, travelers=travelers
    )

    task_instructions = f"Find round-trip trains: Outbound ({origin_code}->{arrival_code}), Return ({return_code}->{origin_code}). Use the search tool."

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
        raw_content = "I have searched extensively but couldn't find any alternative trains matching these constraints. The previously suggested route remains the best option."

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