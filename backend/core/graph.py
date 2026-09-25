from langgraph.graph import StateGraph, END
from state.trip_state import GraphState

from agents.orchestrator import run_orchestrator
from agents.data_gatherer import run_data_gatherer
from agents.itinerary_agent import run_itinerary_agent
from agents.transport_orchestrator import run_transport_orchestrator
from agents.flight_agent import run_flight_agent
from agents.bus_agent import run_bus_agent
from agents.train_agent import run_train_agent
from agents.hotel_agent import run_hotel_agent  # <-- NEW: Imported the Hotel Agent

def route_next_step(state: GraphState) -> str:
    last_msg = state["messages"][-1]
    content = getattr(last_msg, 'content', str(last_msg))

    if "Routing to Itinerary_Agent" in content:
        return "itinerary_agent"
    elif "Routing to Data_Gatherer" in content:
        return "data_gatherer"
    elif "Routing to Booking_Agent" in content:
        return "transport_orchestrator"
    return END

def route_commute(state: GraphState) -> str:
    last_msg = state["messages"][-1]
    content = getattr(last_msg, 'content', str(last_msg))

    if "flight_agent" in content:
        return "flight_agent"
    elif "train_agent" in content:
        return "train_agent"
    elif "bus_agent" in content or "car_agent" in content:
        return "bus_agent"
    return END

def build_travel_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("itinerary_agent", run_itinerary_agent) 
    workflow.add_node("data_gatherer", run_data_gatherer) 
    workflow.add_node("transport_orchestrator", run_transport_orchestrator)
    workflow.add_node("flight_agent", run_flight_agent)
    workflow.add_node("train_agent", run_train_agent)
    workflow.add_node("bus_agent", run_bus_agent)
    workflow.add_node("hotel_agent", run_hotel_agent) # <-- NEW: Hotel Agent Node

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        {
            "itinerary_agent": "itinerary_agent",
            "data_gatherer": "data_gatherer",
            "transport_orchestrator": "transport_orchestrator",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "transport_orchestrator",
        route_commute,
        {
            "flight_agent": "flight_agent",
            "train_agent": "train_agent",
            "bus_agent": "bus_agent",
            END: END
        }
    )

    # Route Transport Output directly to the Hotel Agent to complete the booking package
    workflow.add_edge("flight_agent", "hotel_agent")
    workflow.add_edge("train_agent", "hotel_agent")
    workflow.add_edge("bus_agent", "hotel_agent")

    workflow.add_edge("itinerary_agent", END)
    workflow.add_edge("data_gatherer", END)
    
    # Hotel agent terminates the workflow
    workflow.add_edge("hotel_agent", END)

    return workflow.compile()