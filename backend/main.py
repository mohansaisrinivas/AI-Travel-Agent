import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

from core.graph import build_travel_graph
from state.trip_state import TripDetails

def run_cli_test():
    """Runs a terminal loop to test the Orchestrator routing."""
    app = build_travel_graph()
    
    current_state = {
        "messages": [],
        "trip_data": TripDetails(), 
        "itinerary_drafted": False,
        "itinerary_approved": False
    }

    print("\n✈️ Travel Agent CLI Active. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter a valid message.")
            continue

        if user_input.lower() in ['quit', 'exit']:
            break

        current_state["messages"].append(user_input)
        result_state = app.invoke(current_state)
        
        last_msg = result_state["messages"][-1]
        agent_response = getattr(last_msg, 'content', str(last_msg))
        
        print(f"\nAgent: {agent_response}")
        current_state = result_state

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in .env file.")
    else:
        run_cli_test()