import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the variables from the .env file into the system environment
load_dotenv()

# 2. Explicitly grab the API key from the environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the key was successfully found in the .env file
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found. Please check your .env file.")
    sys.exit(1)

# 3. Explicitly declare the API key when initializing the client
try:
    client = genai.Client(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error: Could not initialize client. Details: {e}")
    sys.exit(1)

# 4. Define the System Prompt
COMFORT_AGENT_SYSTEM_PROMPT = """
You are an expert, empathetic, and detail-oriented luxury travel curator specializing in comfort-driven, slow-paced, and immersive travel. Your goal is to craft breathtaking, stress-free itineraries that balance maximum physical and mental ease with unforgettable local discoveries. You prioritize smooth pacing, aesthetic viewpoints, authentic culinary traditions, and rich cultural insights over rushed tourist checklists.

Core Guidelines:
1. Comfort-First Pacing: Design relaxed daily schedules (limit heavy activities to 2–3 per day). Factor in easy transit methods, minimal walking strain, weather considerations, and dedicated downtime or relaxation windows.
2. Stunning Viewpoints: Prioritize visually breathtaking vantage points (e.g., panoramic rooftop lounges, scenic lookouts accessible by cable car or short gentle walks, and premier sunset/sunrise spots).
3. Authentic Culinary Recommendations: For every meal recommendation, specify must-try local dishes and the exact restaurant, heritage café, or famous street food market where they are most authentic and popular.
4. Culture & Leisure Activities: Integrate meaningful cultural context for historical or sacred sites. Suggest tailored leisure activities like boutique street shopping districts, local artisan markets, or scenic neighborhood strolls that fit a relaxed tempo.

Output Formatting Style:
- Structure the response using clean Markdown, clear day-by-day headings, and expressive emojis (e.g., 🌅 for viewpoints, 🍜 for food, 🛍️ for shopping, 🛋️ for relaxation/downtime).
- Keep the tone warm, inspiring, upscale, and reassuring.
"""

def main():
    print("==================================================")
    print(" 🌴 Welcome to the Comfort-First AI Travel Agent 🌴 ")
    print("==================================================")
    print("Type 'exit' or 'quit' at any time to stop.\n")

    # 5. Create a continuous loop to take user input from the terminal
    while True:
        # Get input from the user
        user_input = input("\n📍 Where would you like to go and for how long? (e.g., '3 days in Kyoto'): \n> ")
        
        # Check if the user wants to quit
        if user_input.lower() in ['exit', 'quit']:
            print("\nSafe travels! Goodbye. ✈️")
            break
            
        # Ensure input isn't empty
        if not user_input.strip():
            print("Please enter a valid destination and duration.")
            continue

        print("\n✨ Curating your perfect, stress-free itinerary... Please wait.\n")

        try:
            # 6. Feed the user input and the system prompt to the LLM
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=COMFORT_AGENT_SYSTEM_PROMPT,
                    temperature=0.7,
                ),
            )
            
            # 7. Print the generated itinerary back to the terminal
            print("--------------------------------------------------")
            print(response.text)
            print("--------------------------------------------------")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")

if __name__ == "__main__":
    main()