# agents/orchestrator.py

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state.trip_state import TripDetails, GraphState
from prompts.agent_prompts import ORCHESTRATOR_PROMPT

# --- 1. Define the Routing Decision Model ---
class OrchestratorDecision(BaseModel):
    next_agent: str = Field(
        description="Must be one of: 'Itinerary_Agent', 'Data_Gatherer', 'Booking_Agent', or 'Reply_to_User'"
    )
    reasoning: str = Field(description="A brief explanation of why this agent was chosen.")
    user_approved_itinerary: bool = Field(
        description="Set to True ONLY if the user just said 'looks good', 'proceed', or approved the plan."
    )

def run_orchestrator(state: GraphState) -> GraphState:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(OrchestratorDecision)

    # Format the System Prompt
    current_trip_data = state["trip_data"].dict()
    # Add the flags to the prompt so the LLM sees the true state
    state_str = f"Trip Data: {current_trip_data}\nItinerary Drafted: {state['itinerary_drafted']}\nItinerary Approved: {state['itinerary_approved']}"
    
    formatted_prompt = ORCHESTRATOR_PROMPT.format(trip_state=state_str)
    user_message = state["messages"][-1]

    print("🧠 Orchestrator is thinking...")
    decision = structured_llm.invoke([
        SystemMessage(content=formatted_prompt),
        HumanMessage(content=user_message)
    ])

    print(f"🔀 Decision: Route to {decision.next_agent}. Reasoning: {decision.reasoning}")

    # --- THE FIX: Update the global state if the user approved! ---
    if decision.user_approved_itinerary:
        state["itinerary_approved"] = True
        print("✅ SYSTEM: Itinerary marked as Approved.")

    # Append routing note for the graph
    state["messages"].append(f"SYSTEM_NOTE: Routing to {decision.next_agent}")
    
    return state