import os
from dotenv import load_dotenv

load_dotenv()

from core.graph import build_travel_graph
from state.trip_state import TripDetails

def run_cli_test():
    """Runs a terminal loop to test the Orchestrator and agent executions."""
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

        prev_message_count = len(current_state["messages"])
        current_state["messages"].append(user_input)

        # Invoke the multi-agent graph
        result_state = app.invoke(current_state)

        # Identify all new messages added during this invocation cycle
        new_messages = result_state["messages"][prev_message_count + 1:]

        for msg in new_messages:
            msg_content = getattr(msg, 'content', str(msg))
            # Suppress internal orchestrator routing tags
            if msg_content.startswith("SYSTEM_NOTE:"):
                continue
            print(f"\nAgent: {msg_content}\n" + ("=" * 50))
        
        current_state = result_state

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in .env file.")
    else:
        run_cli_test()