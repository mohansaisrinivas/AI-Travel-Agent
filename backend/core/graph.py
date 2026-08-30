from langgraph.graph import StateGraph, END
from state.trip_state import GraphState

from agents.orchestrator import run_orchestrator
from agents.data_gatherer import run_data_gatherer
from agents.itinerary_agent import run_itinerary_agent

def booking_agent_node(state: GraphState) -> dict:
    print("✈️ Booking Agent is executing (Placeholder)...")
    return {"messages": ["All booked! Check your email for confirmations."]}

def route_next_step(state: GraphState) -> str:
    last_message = state["messages"][-1]
    
    if "Routing to Itinerary_Agent" in last_message:
        return "itinerary_agent"
    elif "Routing to Data_Gatherer" in last_message:
        return "data_gatherer"
    elif "Routing to Booking_Agent" in last_message:
        return "booking_agent"
    else:
        return END

def build_travel_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("itinerary_agent", run_itinerary_agent) 
    workflow.add_node("data_gatherer", run_data_gatherer) 
    workflow.add_node("booking_agent", booking_agent_node)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        {
            "itinerary_agent": "itinerary_agent",
            "data_gatherer": "data_gatherer",
            "booking_agent": "booking_agent",
            END: END
        }
    )

    workflow.add_edge("itinerary_agent", END)
    workflow.add_edge("data_gatherer", END)
    workflow.add_edge("booking_agent", END)

    return workflow.compile()