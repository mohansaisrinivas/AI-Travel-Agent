import os
import re
from typing import Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch 

from state.trip_state import GraphState
from prompts.agent_prompts import ITINERARY_AGENT_PROMPT

class DestDurationExtraction(BaseModel):
    destination: Optional[str] = Field(default="Not Specified", description="The destination mentioned by the user.")
    duration_days: Optional[int] = Field(default=3, description="The number of days for the trip. Default is 3 if not mentioned.")

def format_chat_history_for_itinerary(messages: list) -> str:
    script = ""
    for msg in messages:
        # Added a small safety check in case LangGraph passes message objects instead of raw strings
        if not isinstance(msg, str):
            msg = getattr(msg, 'content', str(msg))
            
        if msg.startswith("SYSTEM_NOTE:"):
            continue
        elif "I'd be delighted" in msg or "Here is your draft" in msg or "Perfect" in msg or "Agent:" in msg:
            script += f"Agent: {msg}\n"
        else:
            script += f"User: {msg}\n"
    return script

def run_itinerary_agent(state: GraphState) -> dict:
    print("🗺️ Itinerary Agent is working (ReAct Mode)...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    full_context = format_chat_history_for_itinerary(state["messages"])
    
    print("   [Itinerary Agent] Extracting destination and duration...")
    extractor = llm.with_structured_output(DestDurationExtraction)
    extraction = extractor.invoke([
        SystemMessage(content="Extract destination and duration from the user's request. Default to 3 days if not specified."),
        HumanMessage(content=full_context)
    ])
    
    updated_trip_data = state["trip_data"]
    if extraction.destination and extraction.destination != "Not Specified":
        updated_trip_data.destination = extraction.destination
    if extraction.duration_days:
        updated_trip_data.duration_days = extraction.duration_days
        
    print(f"   [Itinerary Agent] Saved -> Dest: {updated_trip_data.destination}, Days: {updated_trip_data.duration_days}")

    # --- STEP 2: Setup Tools (Tavily Only) ---
    tools = []
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearch(max_results=3))
    else:
        print("⚠️ WARNING: TAVILY_API_KEY not found in .env. Agent will run without web search.")

    print("   [Itinerary Agent] Researching and drafting halt-wise plan... (This may take a few seconds)")
    
    react_agent = create_react_agent(llm, tools, prompt=ITINERARY_AGENT_PROMPT)
    react_state = react_agent.invoke({
        "messages": [HumanMessage(content=f"Plan a trip based on this context:\n\n{full_context}")]
    })
    
    # --- THE FIX: Safely parse Gemini's content format ---
    raw_content = react_state["messages"][-1].content
    if isinstance(raw_content, list):
        # If it's a list, extract the text blocks and join them
        final_response = " ".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content])
    else:
        # Otherwise, treat it as a normal string
        final_response = str(raw_content)

    # --- STEP 4: Parse Halts and Clean Output ---
    halts_match = re.search(r"EXTRACTED_HALTS:\s*\[(.*?)\]", final_response)
    if halts_match:
        halts_str = halts_match.group(1)
        halts_list = [h.strip().strip("'\"") for h in halts_str.split(",") if h.strip()]
        updated_trip_data.halts = halts_list
        print(f"   [Itinerary Agent] Saved Halts to Memory -> {halts_list}")
        final_response = re.sub(r"EXTRACTED_HALTS:\s*\[.*?\]", "", final_response).strip()

    return {
        "messages": [final_response],
        "trip_data": updated_trip_data,
        "itinerary_drafted": True
    }