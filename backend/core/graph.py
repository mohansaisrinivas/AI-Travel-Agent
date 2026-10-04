import time
from typing import List, Union
from langgraph.graph import StateGraph, END
from state.trip_state import GraphState

from agents.orchestrator import run_orchestrator
from agents.data_gatherer import run_data_gatherer
from agents.itinerary_agent import run_itinerary_agent
from agents.transport_orchestrator import run_transport_orchestrator
from agents.flight_agent import run_flight_agent
from agents.bus_agent import run_bus_agent
from agents.train_agent import run_train_agent
from agents.hotel_agent import run_hotel_agent

def route_next_step(state: GraphState) -> Union[str, List[str]]:
    print("   [System] Pacing API to avoid Free Tier limits (2s breath)...")
    time.sleep(2.0) 
    
    last_msg = state["messages"][-1]
    content = getattr(last_msg, 'content', str(last_msg))

    if content.startswith("SYSTEM_NOTE: Routing to"):
        # FIX: Convert the entire routing string to lowercase to prevent case-sensitivity crashes
        requested_nodes = content.replace("SYSTEM_NOTE: Routing to ", "").lower().split(",")
        routes = []
        
        for n in requested_nodes:
            n = n.strip()
            if "itinerary_agent" in n: routes.append("itinerary_agent")
            elif "data_gatherer" in n: routes.append("data_gatherer")
            elif "transport_orchestrator" in n: routes.append("transport_orchestrator")
            elif "hotel_agent" in n: routes.append("hotel_agent")
            
        if routes:
            return routes if len(routes) > 1 else routes[0]
            
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

# MODIFIED: Now accepts a checkpointer for memory persistence
def build_travel_graph(checkpointer=None):
    workflow = StateGraph(GraphState)

    workflow.add_node("orchestrator", run_orchestrator)
    workflow.add_node("itinerary_agent", run_itinerary_agent) 
    workflow.add_node("data_gatherer", run_data_gatherer) 
    workflow.add_node("transport_orchestrator", run_transport_orchestrator)
    workflow.add_node("flight_agent", run_flight_agent)
    workflow.add_node("train_agent", run_train_agent)
    workflow.add_node("bus_agent", run_bus_agent)
    workflow.add_node("hotel_agent", run_hotel_agent)

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        ["itinerary_agent", "data_gatherer", "transport_orchestrator", "hotel_agent", END]
    )

    workflow.add_conditional_edges(
        "transport_orchestrator",
        route_commute,
        ["flight_agent", "train_agent", "bus_agent", END]
    )

    workflow.add_edge("flight_agent", END)
    workflow.add_edge("train_agent", END)
    workflow.add_edge("bus_agent", END)
    workflow.add_edge("hotel_agent", END)
    workflow.add_edge("itinerary_agent", END)
    workflow.add_edge("data_gatherer", END)
    
    # MODIFIED: Compile with the checkpointer
    return workflow.compile(checkpointer=checkpointer)