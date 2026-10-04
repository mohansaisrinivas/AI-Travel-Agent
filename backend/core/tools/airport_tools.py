import os
import requests
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

class MultiHaltConnectivityResult(BaseModel):
    origin_city: str
    origin_iata: str = Field(description="3-letter IATA code for departure airport from origin, e.g., DEL")
    origin_airport_name: str

    entry_halt: str = Field(description="The first halt where travelers arrive.")
    arrival_iata: str = Field(description="3-letter IATA code of recommended arrival airport.")
    arrival_airport_name: str
    outbound_last_mile_note: str = Field(
        description="Distance and road commute time from arrival airport to Halt 1."
    )

    exit_halt: str = Field(description="The final halt where travelers conclude their trip.")
    return_departure_iata: str = Field(
        description="3-letter IATA code of recommended departure airport back to origin."
    )
    return_departure_airport_name: str
    return_last_mile_note: str = Field(
        description="Distance and road commute time from Final Halt to return airport."
    )

    is_direct_flight_available: bool = Field(
        description="True if scheduled direct non-stop flights typically operate on this route."
    )
    traveler_consensus_note: str = Field(
        description="A 1-2 sentence summary of what real travel blogs/forums recommend for this specific circuit."
    )

def get_driving_distance(origin: str, destination: str) -> dict:
    """Uses Google Maps Distance Matrix API to get exact drive time and distance."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return {"distance": "Unknown", "duration": "Unknown"}

    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={google_api_key}"
    try:
        response = requests.get(url).json()
        if response.get("status") == "OK":
            element = response["rows"][0]["elements"][0]
            if element.get("status") == "OK":
                return {
                    "distance": element["distance"]["text"],
                    "duration": element["duration"]["text"]
                }
    except Exception as e:
        print(f"   [Distance Matrix Error]: {e}")
        
    return {"distance": "Unknown", "duration": "Unknown"}

def resolve_multi_halt_airports(origin: str, entry_halt: str, exit_halt: str) -> MultiHaltConnectivityResult:
    print(f"   [Airport Tools] Mining travel blogs & forum consensus for {entry_halt} -> {exit_halt} circuit...")
    
    web_context = ""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(max_results=4)
        # Search specifically for traveler blogs, Reddit threads, and forum recommendations
        query = (
            f"best airport to fly into and out of for {entry_halt} to {exit_halt} itinerary "
            f"trip travel blog reddit forum advice"
        )
        try:
            results = tavily.invoke(query)
            web_context = f"\nREAL TRAVELER DISCUSSIONS & BLOG SNIPPETS:\n{results}\n"
        except Exception as e:
            print(f"   [Airport Tools] Tavily blog lookup failed: {e}")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(MultiHaltConnectivityResult)

    prompt = (
        "You are an expert travel logistics architect. Read the provided travel blog snippets, "
        "trip reports, and forum consensus to determine the smartest airport route for this circuit.\n\n"
        "GUIDELINES FOR EXTRACTING TRAVELER CONSENSUS:\n"
        "1. Identify the origin airport code for the starting city.\n"
        "2. READ THE BLOG ADVICE:\n"
        "   - What airport do seasoned travelers, blogs, and Reddit recommend landing at for this circuit?\n"
        "   - Do travelers recommend landing at a single major hub (e.g., BLR for a Coorg/Mysore loop) "
        "and hiring a cab/rental car, or do they endorse separate airports (Open-Jaw)?\n"
        "   - If blogs warn that regional airstrips have sparse/costly flights or unreliable schedules, "
        "follow their recommendation and route through the primary commercial hub.\n"
        "3. Output the exact 3-letter IATA codes and airport names.\n"
        "4. In `traveler_consensus_note`, summarize what the blogs suggest (e.g., 'Travelers recommend "
        "flying round-trip via Bangalore (BLR) and taking the expressway, as direct flights to Mysore are infrequent.')."
    )

    result = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Origin: {origin}\nEntry Halt: {entry_halt}\nExit Halt: {exit_halt}\n{web_context}")
    ])
    
    print(f"   [Airport Tools] Blog Consensus: {result.traveler_consensus_note}")
    print("   [Airport Tools] Calculating exact road commutes via Google Maps...")
    
    # Outbound Last Mile
    outbound_metrics = get_driving_distance(result.arrival_airport_name, entry_halt)
    if outbound_metrics["distance"] != "Unknown":
        result.outbound_last_mile_note = f"{outbound_metrics['distance']} / {outbound_metrics['duration']} drive from {result.arrival_airport_name} to {entry_halt}"
        
    # Return Last Mile
    return_metrics = get_driving_distance(exit_halt, result.return_departure_airport_name)
    if return_metrics["distance"] != "Unknown":
        result.return_last_mile_note = f"{return_metrics['distance']} / {return_metrics['duration']} drive from {exit_halt} to {result.return_departure_airport_name}"

    return result