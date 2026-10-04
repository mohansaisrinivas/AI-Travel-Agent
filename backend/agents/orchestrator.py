from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state.trip_state import TripDetails, GraphState
from prompts.agent_prompts import ORCHESTRATOR_PROMPT

class OrchestratorDecision(BaseModel):
    next_nodes: List[str] = Field(description="A list containing one or more of: 'Itinerary_Agent', 'Data_Gatherer', 'Transport_Orchestrator', 'Hotel_Agent'")
    reasoning: str = Field(description="A brief explanation of why these agents were chosen.")
    user_approved_itinerary: bool = Field(description="Set to True ONLY if the user just approved the drafted itinerary plan.")

class ChatSummary(BaseModel):
    summary: str = Field(description="Concise summary of the user's travel preferences, chosen locations, and constraints.")

def is_core_data_missing(trip_data: dict) -> bool:
    """Hard Python guardrail to prevent premature booking routing."""
    required_fields = ["origin_city", "start_date", "duration_days", "number_of_travelers", "budget_tier", "transport_mode"]
    for req in required_fields:
        if not trip_data.get(req):
            return True
    return False

def run_orchestrator(state: GraphState) -> dict:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    
    chat_msgs = [m for m in state["messages"] if not str(m).startswith("SYSTEM_NOTE:")]
    user_message = getattr(state["messages"][-1], 'content', str(state["messages"][-1])).lower()
    current_trip_data = state["trip_data"]

    # --- INSTANT ARTIFACT RECALL INTERCEPTION ---
    # If the user asks for details and we have them saved, return them immediately!
    if any(k in user_message for k in ["itinerary", "plan", "schedule"]) and current_trip_data.saved_itinerary:
        print("📋 [Orchestrator] Instantly serving cached itinerary.")
        return {"messages": [current_trip_data.saved_itinerary]}
    
    if any(k in user_message for k in ["transport", "flight", "train", "bus"]) and current_trip_data.saved_transport_details:
        print("🚆 [Orchestrator] Instantly serving cached transport details.")
        return {"messages": [current_trip_data.saved_transport_details]}

    if any(k in user_message for k in ["hotel", "stay", "accommodation"]) and current_trip_data.saved_hotel_details:
        print("🏨 [Orchestrator] Instantly serving cached hotel details.")
        return {"messages": [current_trip_data.saved_hotel_details]}
    # ---------------------------------------------

    current_summary = state.get("chat_summary", "No user preferences captured yet.")
    
    if len(chat_msgs) > 4:
        print("   [Orchestrator] Compressing context into chat summary...")
        summarizer = llm.with_structured_output(ChatSummary)
        summary_prompt = f"Current summary: {current_summary}\n\nRecent unsummarized messages:\n" + "\n".join([str(m) for m in chat_msgs[-4:]])
        
        try:
            summary_res = summarizer.invoke([
                SystemMessage(content="You maintain a running concise summary of a user's travel preferences. Update it with the recent messages."), 
                HumanMessage(content=summary_prompt)
            ])
            current_summary = summary_res.summary
        except Exception as e:
            print(f"   [Orchestrator Warning] Summarization skipped: {e}")

    structured_llm = llm.with_structured_output(OrchestratorDecision)
    current_trip_dict = current_trip_data.dict()
    
    state_str = (
        f"Trip Data: {current_trip_dict}\n"
        f"Itinerary Drafted: {state['itinerary_drafted']}\n"
        f"Itinerary Approved: {state['itinerary_approved']}\n"
        f"Chat Summary: {current_summary}"
    )
    
    formatted_prompt = ORCHESTRATOR_PROMPT.format(trip_state=state_str)

    print("🧠 Main Orchestrator is thinking...")
    decision = structured_llm.invoke([
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=user_message)
    ])

    if ("Transport_Orchestrator" in decision.next_nodes or "Hotel_Agent" in decision.next_nodes) and is_core_data_missing(current_trip_dict):
        print("   [Orchestrator Guardrail] Intercepted premature booking route. Rerouting to Data_Gatherer.")
        decision.next_nodes = ["Data_Gatherer"]
    print(f"🔀 Decision: Route to {decision.next_nodes}. Reasoning: {decision.reasoning}")

    updates = {
        "messages": [f"SYSTEM_NOTE: Routing to {','.join(decision.next_nodes)}"],
        "chat_summary": current_summary
    }
    
    if decision.user_approved_itinerary:
        updates["itinerary_approved"] = True
        print("✅ SYSTEM: Itinerary marked as Approved.")
        
    return updates