import os
import requests
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

class MultiHaltConnectivityResult(BaseModel):
    origin_city: str
    origin_iata: str = Field(description="3-letter IATA code for departure airport from origin, e.g., HYD")
    origin_airport_name: str

    entry_halt: str = Field(description="The first halt where travelers arrive.")
    arrival_iata: str = Field(description="3-letter IATA code of airport closest to Halt 1.")
    arrival_airport_name: str
    outbound_last_mile_note: str = Field(
        description="LLM's guess for the road distance. (Will be overwritten by Google Maps)"
    )

    exit_halt: str = Field(description="The final halt where travelers conclude their trip.")
    return_departure_iata: str = Field(
        description="3-letter IATA code of airport closest to the Final Halt for departure back to origin."
    )
    return_departure_airport_name: str
    return_last_mile_note: str = Field(
        description="LLM's guess for the road distance. (Will be overwritten by Google Maps)"
    )

    is_direct_flight_available: bool = Field(
        description="True if scheduled direct non-stop flights typically operate on this route."
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
    print(f"   [Airport Tools] Resolving nearest commercial airports for {entry_halt} and {exit_halt}...")
    
    web_context = ""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(max_results=2)
        search_query = f"nearest commercial passenger airport to {entry_halt} and nearest commercial passenger airport to {exit_halt}"
        try:
            results = tavily.invoke(search_query)
            web_context = f"\nLIVE WEB RESEARCH RESULTS:\n{results}\n"
        except Exception as e:
            print(f"   [Airport Tools] Web search failed: {e}")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(MultiHaltConnectivityResult)

    prompt = (
        "You are an aviation geography expert. Given an origin city, an Entry Halt (Halt 1), and an Exit Halt (Final Halt):\n"
        "1. Identify the primary commercial airport and IATA code for the origin city.\n"
        "2. Use the web research to identify the closest commercial airport and IATA code to Halt 1 for arrival.\n"
        "3. Use the web research to identify the closest commercial airport and IATA code to the Final Halt for departure back to origin.\n"
        "4. Note whether direct flights typically exist between origin and arrival airport."
    )

    result = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Origin: {origin}\nEntry Halt (Halt 1): {entry_halt}\nExit Halt (Final Halt): {exit_halt}{web_context}")
    ])
    
    # --- NEW: Google Maps Integration ---
    print("   [Airport Tools] Calculating exact last-mile road commute using Google Maps...")
    
    # Outbound Last Mile (Arrival Airport -> Halt 1)
    outbound_metrics = get_driving_distance(result.arrival_airport_name, entry_halt)
    if outbound_metrics["distance"] != "Unknown":
        result.outbound_last_mile_note = f"{outbound_metrics['distance']} / {outbound_metrics['duration']} drive from {result.arrival_airport_name} to {entry_halt}"
        
    # Return Last Mile (Final Halt -> Return Airport)
    return_metrics = get_driving_distance(exit_halt, result.return_departure_airport_name)
    if return_metrics["distance"] != "Unknown":
        result.return_last_mile_note = f"{return_metrics['distance']} / {return_metrics['duration']} drive from {exit_halt} to {result.return_departure_airport_name}"

    return result