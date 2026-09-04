from langgraph.graph import StateGraph, END
from state.trip_state import GraphState

from agents.orchestrator import run_orchestrator
from agents.data_gatherer import run_data_gatherer
from agents.itinerary_agent import run_itinerary_agent
from agents.transport_orchestrator import run_transport_orchestrator
from agents.flight_agent import run_flight_agent

def train_agent_placeholder(state: GraphState) -> dict:
    return {"messages": ["🚆 Train Agent: Searching IRCTC express trains... (Module ready for integration)"]}

def bus_agent_placeholder(state: GraphState) -> dict:
    return {"messages": ["🚌 Bus Agent: Searching intercity premium bus operators... (Module ready for integration)"]}

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
    workflow.add_node("train_agent", train_agent_placeholder)
    workflow.add_node("bus_agent", bus_agent_placeholder)

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

    workflow.add_edge("itinerary_agent", END)
    workflow.add_edge("data_gatherer", END)
    workflow.add_edge("flight_agent", END)
    workflow.add_edge("train_agent", END)
    workflow.add_edge("bus_agent", END)

    return workflow.compile()