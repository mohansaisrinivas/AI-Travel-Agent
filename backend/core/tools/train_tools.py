import os
import requests
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

class MultiHaltTrainConnectivityResult(BaseModel):
    origin_station_code: str = Field(description="Exact IRCTC station code for the primary interstate express origin, e.g. SC, CSMT, HWH.")
    origin_station_name: str
    arrival_station_code: str = Field(description="Exact IRCTC code of station closest to Halt 1.")
    arrival_station_name: str
    outbound_last_mile_note: str = Field(description="LLM guess, overwritten by Google Maps.")
    
    return_departure_station_code: str = Field(description="Exact IRCTC code of station closest to the Final Halt.")
    return_departure_station_name: str
    return_last_mile_note: str = Field(description="LLM guess, overwritten by Google Maps.")

def get_driving_distance(origin: str, destination: str) -> dict:
    """Uses Google Maps Distance Matrix API to get exact drive time."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return {"distance": "Unknown", "duration": "Unknown"}

    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={google_api_key}"
    try:
        response = requests.get(url).json()
        if response.get("status") == "OK":
            element = response["rows"][0]["elements"][0]
            if element.get("status") == "OK":
                return {"distance": element["distance"]["text"], "duration": element["duration"]["text"]}
    except Exception as e:
        print(f"   [Distance Matrix Error]: {e}")
    return {"distance": "Unknown", "duration": "Unknown"}

def resolve_multi_halt_train_stations(origin: str, entry_halt: str, exit_halt: str) -> MultiHaltTrainConnectivityResult:
    print(f"   [Train Tools] Resolving primary IRCTC express hubs for {origin}, {entry_halt} and {exit_halt}...")
    
    web_context = ""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(max_results=3)
        try:
            query = (
                f"What is the primary major railway station code for long-distance express interstate trains departing from {origin}? "
                f"Also find the nearest major railway station codes for {entry_halt} and {exit_halt}."
            )
            web_context = f"\nLIVE WEB RESEARCH:\n{tavily.invoke(query)}\n"
        except Exception:
            pass

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(MultiHaltTrainConnectivityResult)

    prompt = (
        "You are an expert Indian Railways (IRCTC) logician. Given an origin location, Halt 1, and Final Halt:\n"
        "1. Use the live web research to find the primary major railway station code (e.g., SC for Hyderabad, CSMT or BCT for Mumbai, SBC for Bangalore, HWH for Kolkata) that handles major interstate express trains, NOT small local suburban terminals.\n"
        "2. Identify the closest major commercial railway station code to Halt 1 and the Final Halt.\n"
        "3. Always provide the exact 2 to 5 letter official IRCTC station codes."
    )

    result = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Origin: {origin}\nHalt 1: {entry_halt}\nFinal Halt: {exit_halt}{web_context}")
    ])
    
    print("   [Train Tools] Calculating exact last-mile road commute via Google Maps...")
    
    outbound_metrics = get_driving_distance(result.arrival_station_name, entry_halt)
    if outbound_metrics["distance"] != "Unknown":
        result.outbound_last_mile_note = f"{outbound_metrics['distance']} / {outbound_metrics['duration']} drive from {result.arrival_station_name} to {entry_halt}"
        
    return_metrics = get_driving_distance(exit_halt, result.return_departure_station_name)
    if return_metrics["distance"] != "Unknown":
        result.return_last_mile_note = f"{return_metrics['distance']} / {return_metrics['duration']} drive from {exit_halt} to {result.return_departure_station_name}"

    return result

def is_valid_layover(arr_time_str: str, dep_time_str: str) -> float:
    """Calculates if the layover buffer is between 2 and 5 hours, handling midnight crossovers."""
    try:
        arr = datetime.strptime(arr_time_str, "%H:%M")
        dep = datetime.strptime(dep_time_str, "%H:%M")
        if dep < arr:
            dep += timedelta(days=1)
        return (dep - arr).total_seconds() / 3600
    except:
        return -1.0

@tool
def search_trains(origin_code: str, dest_code: str, travel_date: str) -> str:
    """Queries Parse.bot IRCTC API using GET. If 0 direct trains, autonomously finds connecting trains with a 2-5 hr layover."""
    parse_key = os.getenv("PARSE_API_KEY")

    def fetch_irctc(src: str, dst: str, date: str) -> list:
        if not parse_key: 
            return []
        try:
            # Replace with your actual scraper endpoint URL if different
            url = "https://api.parse.bot/scraper/4cb03185-74ea-4b58-9a5b-c0662c659aa0/search_trains"
            headers = {"X-API-Key": parse_key}
            
            # Using GET with exact query parameters required by Parse.bot
            params = {
                "from_station": src,
                "to_station": dst,
                "date": date
            }
            
            res = requests.get(url, headers=headers, params=params, timeout=15)
            
            if res.status_code == 200:
                payload = res.json()
                
                # --- AGGRESSIVE JSON UNWRAPPING ---
                # 1. If it's a direct list
                if isinstance(payload, list):
                    return payload
                
                # 2. If it's wrapped in {"data": {...}} or {"result": {...}}
                data_block = payload.get("data", payload.get("result", payload))
                
                if isinstance(data_block, dict):
                    return data_block.get("trains", [])
                elif isinstance(data_block, list):
                    return data_block
                    
                return payload.get("trains", [])
                
            elif res.status_code == 500:
                # Suppress Parse.bot's scraper panic when IRCTC redirects to indirect trains
                print(f"   [Train API] Scraper returned 500 (IRCTC indirect redirect). Treating as 0 direct trains.")
                return []
            else:
                print(f"   [Train API] Error status {res.status_code}: {res.text}")
        except Exception as e:
            print(f"   [Train API] Request exception: {e}")
        return []

    print(f"   [Train API] Querying IRCTC for {origin_code} -> {dest_code} on {travel_date}...")
    direct_trains = fetch_irctc(origin_code, dest_code, travel_date)

    if direct_trains:
        print(f"   [Train API] Success! Found {len(direct_trains)} direct trains.")
        res_str = f"DIRECT TRAINS AVAILABLE ({origin_code} -> {dest_code}):\n"
        for i, t in enumerate(direct_trains[:5]):
            res_str += f"{i+1}. {t.get('train_name', 'Express')} ({t.get('train_number', 'N/A')}) | Classes: {', '.join(t.get('classes', ['SL', '3A']))} | Dep: {t.get('departure_time', 'N/A')} | Arr: {t.get('arrival_time', 'N/A')}\n"
        return res_str

    if parse_key:
        print(f"   [Train API] 0 direct trains found. Initiating SPLIT-JOURNEY engine...")
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
        junction_res = llm.invoke(f"What is the single best major Indian Railways junction station code (e.g. NGP, BZA) to connect trains from {origin_code} to {dest_code}? Return ONLY the station code.")
        
        raw_content = junction_res.content
        if isinstance(raw_content, list):
            junction_str = "".join([item.get("text", "") if isinstance(item, dict) else str(item) for item in raw_content])
        else:
            junction_str = str(raw_content)
            
        junction = junction_str.strip().upper()
        
        print(f"   [Train API] Junction discovered: {junction}. Fetching Leg 1 & Leg 2...")
        leg1_trains = fetch_irctc(origin_code, junction, travel_date)
        leg2_trains = fetch_irctc(junction, dest_code, travel_date)

        for t1 in leg1_trains:
            for t2 in leg2_trains:
                layover_hrs = is_valid_layover(t1.get("arrival_time", "00:00"), t2.get("departure_time", "00:00"))
                if 2.0 <= layover_hrs <= 5.0:
                    print("   [Train API] Perfect connecting train pair found!")
                    return (
                        f"SPLIT JOURNEY REQUIRED (No Direct Trains) via Junction {junction}:\n"
                        f"LEG 1: {t1.get('train_name', 'Express')} ({t1.get('train_number', 'N/A')}) | Departs {origin_code}: {t1.get('departure_time')} -> Arrives {junction}: {t1.get('arrival_time')}\n"
                        f"[LAYOVER: {round(layover_hrs, 1)} hours at {junction}]\n"
                        f"LEG 2: {t2.get('train_name', 'Express')} ({t2.get('train_number', 'N/A')}) | Departs {junction}: {t2.get('departure_time')} -> Arrives {dest_code}: {t2.get('arrival_time')}\n"
                        f"INSTRUCTION: Advise the user to book using IRCTC's 'Connecting Journey Booking' feature to link PNRs."
                    )
        return f"Direct check returned 0 trains for {origin_code} -> {dest_code} on {travel_date}. Please verify station codes."

    # Fallback if no key
    print("   [Train API] WARNING: Using mock fallback data.")
    return (
        f"Available Trains ({origin_code} -> {dest_code}) on {travel_date}:\n"
        f"1. TELANGANA EXP (12722) | Class: 1A, 2A, 3A, SL | Departs: 06:00 | Arrives: 06:25 (+1d) | Fare: ₹2,100/person\n"
        f"2. RAJDHANI EXP (12437) | Class: 1A, 2A, 3A | Departs: 08:10 | Arrives: 08:50 (+1d) | Fare: ₹3,400/person"
    )