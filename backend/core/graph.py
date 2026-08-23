# core/graph.py

from langgraph.graph import StateGraph, END

# Import the state schema
from state.trip_state import GraphState

# Import our active agents
from agents.orchestrator import run_orchestrator
from agents.data_gatherer import run_data_gatherer


# --- Placeholder Nodes for Child Agents ---
def itinerary_agent_node(state: GraphState) -> GraphState:
    print("🗺️ Itinerary Agent is drafting a plan (Placeholder)...")
    
    # Simulate the agent doing work
    state["itinerary_drafted"] = True
    state["trip_data"].destination = "Goa" 
    
    state["messages"].append("Here is your draft itinerary for Goa. Does this look good to proceed?")
    
    return state


def booking_agent_node(state: GraphState) -> GraphState:
    print("✈️ Booking Agent is executing (Placeholder)...")
    state["messages"].append("All booked! Check your email for confirmations.")
    return state
# ----------------------------------------


# --- The Routing Function ---
def route_next_step(state: GraphState) -> str:
    """
    This reads the SYSTEM_NOTE added by the Orchestrator to decide where to go.
    """
    last_message = state["messages"][-1]
    
    if "Routing to Itinerary_Agent" in last_message:
        return "itinerary_agent"
    elif "Routing to Data_Gatherer" in last_message:
        return "data_gatherer"
    elif "Routing to Booking_Agent" in last_message:
        return "booking_agent"
    else:
        return END


# --- Build the Graph ---
def build_travel_graph():
    workflow = StateGraph(GraphState)

    # Add all the nodes
    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("itinerary_agent", itinerary_agent_node)
    workflow.add_node("data_gatherer", run_data_gatherer) 
    workflow.add_node("booking_agent", booking_agent_node)

    # Entry point
    workflow.set_entry_point("orchestrator")

    # Add Conditional Edges
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

    # Route child agents back to END
    workflow.add_edge("itinerary_agent", END)
    workflow.add_edge("data_gatherer", END)
    workflow.add_edge("booking_agent", END)

    return workflow.compile()