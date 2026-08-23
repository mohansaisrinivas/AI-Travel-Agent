# agents/data_gatherer.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from state.trip_state import GraphState, TripDetails
from prompts.agent_prompts import DATA_EXTRACTION_PROMPT, DATA_QUESTION_PROMPT

def format_chat_history(messages: list) -> str:
    """
    Converts the raw message list into a clean script, dropping system routing notes.
    This gives the LLM the full conversational context it needs to reason.
    """
    script = ""
    for msg in messages:
        if msg.startswith("SYSTEM_NOTE:"):
            continue # Skip internal routing notes
        # Basic heuristic: if it mentions 'drafting' or 'gatherer', it's the agent
        elif "I'd be delighted" in msg or "Here is your draft" in msg or "Perfect" in msg:
            script += f"Agent: {msg}\n"
        else:
            script += f"User: {msg}\n"
    return script

def run_data_gatherer(state: GraphState) -> GraphState:
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    
    # --- THE FIX: Provide the Full Context ---
    full_conversation_context = format_chat_history(state["messages"])
    print(f"   [Data Gatherer] Analyzing full context:\n{full_conversation_context}")
    
    # --- STEP 1: EXTRACTION ---
    extractor = llm.with_structured_output(TripDetails)
    
    # We now feed the ENTIRE script to the extractor, not just the last message.
    extraction = extractor.invoke([
        SystemMessage(content=DATA_EXTRACTION_PROMPT + "\n\nReview the following conversation history and extract the current known details."),
        HumanMessage(content=full_conversation_context)
    ])
    
    # --- STEP 2: UPDATE STATE ---
    extracted_dict = extraction.dict(exclude_none=True)
    for key, val in extracted_dict.items():
        setattr(state["trip_data"], key, val)
        print(f"   [Data Gatherer] Saved to memory -> {key}: {val}")
        
    # --- STEP 3: CHECK WHAT IS STILL MISSING ---
    current_data = state["trip_data"].dict()
    missing_fields = [k for k, v in current_data.items() if v is None]
    
    # --- STEP 4: GENERATE THE NEXT QUESTION ---
    if not missing_fields:
        state["messages"].append("Perfect, I have all the details I need! Let's get this booked.")
    else:
        formatted_question_prompt = DATA_QUESTION_PROMPT.format(missing_fields=missing_fields)
        
        # We also provide the conversation history to the question generator so it doesn't repeat itself
        response = llm.invoke([
            SystemMessage(content=formatted_question_prompt),
            HumanMessage(content=f"Conversation so far:\n{full_conversation_context}\n\nGenerate the next question:")
        ])
        
        raw_content = response.content
        clean_text = raw_content[0].get("text", "") if isinstance(raw_content, list) else raw_content
        state["messages"].append(clean_text)
    
    return state