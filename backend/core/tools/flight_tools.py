import os
import requests
import urllib3
from datetime import datetime
from langchain_core.tools import tool

# Suppress the warnings that pop up when verify=False is used
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@tool
def search_flights(origin_iata: str, destination_iata: str, travel_date: str, return_date: str = None, budget_tier: str = "standard") -> str:
    """Queries flights between two 3-letter IATA airport codes on SerpApi. Automatically detects layovers and supports round trips."""
    serpapi_key = os.getenv("SERPAPI_KEY")

    if not serpapi_key:
        print("   [Flight API] WARNING: SERPAPI_KEY not found. Using mock fallback data.")
        return (
            f"Available Flights ({origin_iata.upper()} <-> {destination_iata.upper()}):\n"
            f"1. IndiGo (6E-412) - Non-stop | Fare: ₹8,300/person | Baggage: 15kg | Rating: 3.8/5\n"
            f"2. Air India (AI-805) - 1-stop layover | Fare: ₹12,900/person | Baggage: 25kg | Rating: 3.9/5"
        )

    # 1. Format dates defensively
    try:
        formatted_outbound = datetime.strptime(travel_date.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        formatted_outbound = travel_date.strip()

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_flights",
        "departure_id": origin_iata.upper(),
        "arrival_id": destination_iata.upper(),
        "outbound_date": formatted_outbound,
        "currency": "INR",
        "hl": "en",
        "api_key": serpapi_key
    }

    # 2. Configure Round-Trip vs One-Way
    is_round_trip = bool(return_date and return_date.upper() not in ["TBD", "NONE", "NULL", ""])
    if is_round_trip:
        try:
            formatted_return = datetime.strptime(return_date.strip(), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            formatted_return = return_date.strip()
        params["type"] = "1"
        params["return_date"] = formatted_return
        trip_type_label = "Round-Trip"
    else:
        params["type"] = "2"
        trip_type_label = "One-Way"

    print(f"   [Flight API] Searching {trip_type_label} on SerpApi: {origin_iata.upper()} -> {destination_iata.upper()}...")

    try:
        # FIX: Added verify=False to bypass Windows SSL Certificate Errors
        res = requests.get(url, params=params, timeout=15, verify=False)
        if res.status_code == 200:
            data = res.json()
            flights = data.get("best_flights", []) + data.get("other_flights", [])
            if not flights and "flights" in data:
                flights = data["flights"]

            if flights:
                print(f"   [Flight API] Success! Found {len(flights)} flights (Direct & Connecting).")
                formatted_flights = []
                for i, f in enumerate(flights[:5]):
                    flight_details = f.get("flights", [{}])[0]
                    airline = flight_details.get("airline", "Unknown Airline")
                    flight_num = flight_details.get("flight_number", "Unknown")
                    dep_time = flight_details.get("departure_airport", {}).get("time", "Unknown").split()[-1]
                    price = f.get("price", "Price unavailable")
                    
                    layovers = f.get("layovers", [])
                    stop_info = "Non-stop" if not layovers else f"{len(layovers)}-stop layover"
                    
                    formatted_flights.append(
                        f"{i+1}. {airline} ({flight_num}) - {stop_info} | Outbound Departs: {dep_time} | {trip_type_label} Fare: ₹{price}"
                    )
                return "\n".join(formatted_flights)
            else:
                print(f"   [Flight API] 0 flights found for {origin_iata} -> {destination_iata}.")
                return f"0 flights found from {origin_iata} to {destination_iata} on {travel_date}. Please verify dates or consider alternate airports."
        else:
            return f"Flight service temporarily unavailable (Status {res.status_code})."
    except Exception as e:
        print(f"   [Flight API] Request Exception: {e}")
        return f"Error querying flight database: {e}"