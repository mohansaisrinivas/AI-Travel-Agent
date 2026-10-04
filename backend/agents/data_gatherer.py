from pydantic import BaseModel, Field
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state.trip_state import GraphState, TripDetails
from prompts.agent_prompts import DATA_EXTRACTION_PROMPT, DATA_QUESTION_PROMPT

# Strict list of what the user actually needs to answer
USER_FACING_FIELDS = [
    "origin_city", "start_date", "duration_days", "number_of_travelers", 
    "budget_tier", "transport_mode"
]

class DataGathererResult(BaseModel):
    extracted_data: TripDetails = Field(description="The details extracted from the conversation.")
    next_question: Optional[str] = Field(description="A friendly, conversational question asking the user for ONE of the missing fields. Leave null if everything is collected.")

def run_data_gatherer(state: GraphState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    current_trip_data = state["trip_data"]
    
    # 1. PRE-CALCULATE what is currently missing
    missing_fields = [k for k in USER_FACING_FIELDS if getattr(current_trip_data, k) is None]
    
    if current_trip_data.transport_mode and current_trip_data.transport_mode.lower() in ["bus","train"]:
        if not current_trip_data.origin_boarding_area:
            missing_fields.append("origin_boarding_area (the specific neighborhood they want to board near)")

    # 2. FORMAT the conversation context
    context = f"Chat Summary: {state.get('chat_summary', '')}\n\nRecent Interactions:\n"
    context += "\n".join([str(m) for m in state["messages"][-3:] if not str(m).startswith("SYSTEM_NOTE:")])
    
    # 3. MERGE your exact prompts dynamically
    if missing_fields:
        formatted_question_prompt = DATA_QUESTION_PROMPT.format(missing_fields=missing_fields)
        combined_system_prompt = (
            f"{DATA_EXTRACTION_PROMPT}\n\n"
            f"---\n\n"
            f"If any of the required fields are STILL missing after extracting data from the latest message, "
            f"use the following instructions to draft the `next_question`:\n\n"
            f"{formatted_question_prompt}"
        )
    else:
        combined_system_prompt = DATA_EXTRACTION_PROMPT

    print(f"   [Data Gatherer] Extracting & formatting question (Single-Pass)...")
    extractor = llm.with_structured_output(DataGathererResult)
    
    extraction = extractor.invoke([
        SystemMessage(content=combined_system_prompt),
        HumanMessage(content=context)
    ])
    
    # 4. SAVE extracted data securely (Immutable Update)
    extracted_dict = extraction.extracted_data.dict(exclude_none=True)
    update_dict = {}
    
    for key, val in extracted_dict.items():
        if not val: # Ignore empty strings or lists
            continue
        # CRITICAL: Do not overwrite structured HaltAllocations with raw strings
        if key == "halts" and current_trip_data.halts:
            continue
            
        update_dict[key] = val
        print(f"   [Data Gatherer] Saved to memory -> {key}: {val}")
        
    updated_trip_data = current_trip_data.model_copy(update=update_dict)
        
    # 5. ROUTE output
    if not extraction.next_question:
        new_msg = "Perfect, I have all the details I need! Let's get this booked."
    else:
        new_msg = extraction.next_question
    
    return {
        "messages": [new_msg],
        "trip_data": updated_trip_data
    }