from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from state.trip_state import GraphState, TripDetails
from prompts.agent_prompts import DATA_EXTRACTION_PROMPT, DATA_QUESTION_PROMPT

def format_chat_history(messages: list) -> str:
    script = "" 
    for msg in messages:
        if not isinstance(msg, str):
            msg = getattr(msg, 'content', str(msg))
            
        if msg.startswith("SYSTEM_NOTE:"):
            continue 
        elif "I'd be delighted" in msg or "Here is your draft" in msg or "Perfect" in msg or "Agent:" in msg:
            script += f"Agent: {msg}\n"
        else:
            script += f"User: {msg}\n"
    return script

def run_data_gatherer(state: GraphState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    
    full_conversation_context = format_chat_history(state["messages"])
    print(f"   [Data Gatherer] Analyzing full context:\n{full_conversation_context}")
    
    extractor = llm.with_structured_output(TripDetails)
    extraction = extractor.invoke([
        SystemMessage(content=DATA_EXTRACTION_PROMPT + "\n\nReview the following conversation history and extract the current known details."),
        HumanMessage(content=full_conversation_context)
    ])
    
    extracted_dict = extraction.dict(exclude_none=True)
    updated_trip_data = state["trip_data"]
    
    for key, val in extracted_dict.items():
        # Prevent overwriting good data with empty strings or empty lists
        if val == [] or val == "":
            continue
        setattr(updated_trip_data, key, val)
        print(f"   [Data Gatherer] Saved to memory -> {key}: {val}")
        
    current_data = updated_trip_data.dict()
    missing_fields = [k for k, v in current_data.items() if v is None]
    
    if not missing_fields:
        new_msg = "Perfect, I have all the details I need! Let's get this booked."
    else:
        formatted_question_prompt = DATA_QUESTION_PROMPT.format(missing_fields=missing_fields)
        response = llm.invoke([
            SystemMessage(content=formatted_question_prompt),
            HumanMessage(content=f"Conversation so far:\n{full_conversation_context}\n\nGenerate the next question:")
        ])
        raw_content = response.content
        new_msg = raw_content[0].get("text", "") if isinstance(raw_content, list) else raw_content
    
    return {
        "messages": [new_msg],
        "trip_data": updated_trip_data
    }