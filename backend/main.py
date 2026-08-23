# main.py

import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Import the graph builder we just wrote
from core.graph import build_travel_graph
from state.trip_state import TripDetails

def run_cli_test():
    """Runs a terminal loop to test the Orchestrator routing."""
    
    # 1. Compile the graph
    app = build_travel_graph()
    
    # 2. Initialize the empty state
    # This is what the Orchestrator reads to see what is missing
    current_state = {
        "messages": [],
        "trip_data": TripDetails(), # All fields are None by default
        "itinerary_drafted": False,
        "itinerary_approved": False
    }

    print("\n✈️  Travel Agent CLI Active. Type 'quit' to exit.")
    print("-" * 50)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        #add safety check for empty input
        if not user_input:
            print("Please enter a valid message.")
            continue

        if user_input.lower() in ['quit', 'exit']:
            break

        # Add user message to state
        current_state["messages"].append(user_input)

        # Run the graph
        # This triggers the Orchestrator -> Router -> Child Agent
        result_state = app.invoke(current_state)
        
        # Print the child agent's response
        # The child agent always appends its message to the end of the list
        agent_response = result_state["messages"][-1]
        print(f"\nAgent: {agent_response}")
        
        # Update our running state for the next loop
        current_state = result_state

if __name__ == "__main__":
    # Safety check
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in .env file.")
    else:
        run_cli_test()