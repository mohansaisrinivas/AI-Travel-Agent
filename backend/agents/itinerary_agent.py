import os
import re
import json
import ast
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_tavily import TavilySearch 

from state.trip_state import GraphState, HaltAllocation
from prompts.agent_prompts import ITINERARY_AGENT_PROMPT

def run_itinerary_agent(state: GraphState) -> dict:
    trip_data = state["trip_data"]

    # RECALL CHECK: If itinerary is already saved and user isn't asking to change destinations, return cached version!
    last_msg = str(state["messages"][-1]).lower()
    if trip_data.saved_itinerary and not any(k in last_msg for k in ["change", "instead", "modify", "add", "kerala", "goa"]):
        print("🗺️ Itinerary Agent: Recalling saved itinerary from state memory (Zero LLM / Search Cost)...")
        return {"messages": [trip_data.saved_itinerary]}

    print("🗺️ Itinerary Agent is working (Direct Binding Mode)...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    
    duration = trip_data.duration_days or 3
    dest = trip_data.destination or "the requested destination"
    
    context = f"Chat Summary: {state.get('chat_summary', '')}\n\nRecent Interactions:\n"
    context += "\n".join([str(m) for m in state["messages"][-4:] if not str(m).startswith("SYSTEM_NOTE:")])

    tools = []
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearch(max_results=3))

    llm_with_tools = llm.bind_tools(tools)
    
    instructions = f"Draft a {duration}-day itinerary for {dest} based on this context:\n{context}"
    messages_for_llm = [SystemMessage(content=ITINERARY_AGENT_PROMPT), HumanMessage(content=instructions)]
    
    tool_call_msg = llm_with_tools.invoke(messages_for_llm)
    
    if tool_call_msg.tool_calls:
        messages_for_llm.append(tool_call_msg)
        for tc in tool_call_msg.tool_calls:
            tool_instance = next((t for t in tools if t.name == tc["name"]), None)
            if tool_instance:
                try:
                    result = tool_instance.invoke(tc["args"])
                except Exception as e:
                    result = f"Error: {e}"
                messages_for_llm.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        
        final_msg = llm.invoke(messages_for_llm)
        raw_content = final_msg.content
    else:
        raw_content = tool_call_msg.content

    if isinstance(raw_content, list):
        final_response = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content])
    else:
        final_response = str(raw_content)

    halts_match = re.search(r"EXTRACTED_HALTS:\s*(\[.*?\])", final_response, re.DOTALL)
    parsed_allocations = []
    if halts_match:
        raw_halts_str = halts_match.group(1).strip()
        try:
            data = json.loads(raw_halts_str)
            for item in data:
                if isinstance(item, dict):
                    parsed_allocations.append(HaltAllocation(
                        halt_name=item.get("halt_name", item.get("name", "Halt")),
                        nights=int(item.get("nights", 1))
                    ))
        except Exception:
            pass

    updated_trip_data = trip_data.model_copy(update={
        "halts": parsed_allocations if parsed_allocations else trip_data.halts,
        "saved_itinerary": final_response
    })

    final_response = re.sub(r"EXTRACTED_HALTS:\s*\[.*?\]", "", final_response, flags=re.DOTALL).strip()

    return {
        "messages": [final_response],
        "trip_data": updated_trip_data,
        "itinerary_drafted": True
    }