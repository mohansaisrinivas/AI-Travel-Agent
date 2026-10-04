import os
import requests
import urllib3
import dateutil.parser
from datetime import datetime, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

# Suppress Windows SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class MultiHaltTrainConnectivityResult(BaseModel):
    origin_station_code: str = Field(description="Exact IRCTC station code for the primary interstate express origin.")
    origin_station_name: str
    arrival_station_code: str = Field(description="Exact IRCTC code of station closest to Halt 1.")
    arrival_station_name: str
    outbound_last_mile_note: str = Field(description="LLM guess, overwritten by Google Maps.")
    
    return_departure_station_code: str = Field(description="Exact IRCTC code of station closest to the Final Halt.")
    return_departure_station_name: str
    return_last_mile_note: str = Field(description="LLM guess, overwritten by Google Maps.")

class RouteConsensus(BaseModel):
    junction_code: str = Field(description="The 2-4 letter station code of the recommended transit hub.")
    junction_name: str = Field(description="The name of the transit city.")
    reasoning: str = Field(description="Brief explanation of why travelers use this route based on web consensus.")

def get_driving_distance(origin: str, destination: str) -> dict:
    """Uses Google Maps Distance Matrix API to get exact drive time."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return {"distance": "Unknown", "duration": "Unknown"}

    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={google_api_key}"
    try:
        response = requests.get(url, timeout=10, verify=False).json()
        if response.get("status") == "OK":
            element = response["rows"][0]["elements"][0]
            if element.get("status") == "OK":
                return {"distance": element["distance"]["text"], "duration": element["duration"]["text"]}
    except Exception as e:
        print(f"   [Distance Matrix Error]: {e}")
    return {"distance": "Unknown", "duration": "Unknown"}

def resolve_multi_halt_train_stations(origin: str, entry_halt: str, exit_halt: str) -> MultiHaltTrainConnectivityResult:
    print(f"   [Train Tools] Mining web consensus for DIRECT train routes from {origin} to {entry_halt} & {exit_halt}...")
    
    web_context = ""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(max_results=3)
        try:
            # FIX: Query explicitly for the ROUTE to discover the actual destination station used by direct trains
            query = f"direct trains from {origin} to {entry_halt} and {exit_halt} best destination railway station irctc"
            web_context = f"\nLIVE WEB RESEARCH:\n{tavily.invoke(query)}\n"
        except Exception:
            pass

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(MultiHaltTrainConnectivityResult)

    prompt = (
        "You are an expert Indian Railways Logistics Architect. Given an origin location, Halt 1, and Final Halt:\n"
        "1. Read the LIVE WEB RESEARCH to identify the best railway stations for this specific ROUTE.\n"
        "2. CRITICAL: CORRIDOR CONNECTIVITY > GEOGRAPHIC PROXIMITY. Always pick destination stations that maximize the chance of DIRECT trains from the origin. For example, if traveling from Hyderabad to North Goa, pick Madgaon (MAO) because direct trains run there, even if Thivim (THVM) is geographically closer to the hotel.\n"
        "3. In multi-station origins (like Hyderabad), pick the primary inter-state hub (e.g., Secunderabad (SC) or Kacheguda (KCG)).\n"
        "4. Output the exact official 2 to 5 letter IRCTC station codes (e.g., SC, MAO, NDLS, HWH)."
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

def get_web_consensus_junction(origin_code: str, dest_code: str) -> Optional[RouteConsensus]:
    """Uses Tavily to find the actual recommended connecting route from railway forums."""
    print(f"   [Train Tools] Mining forums for the best connecting route from {origin_code} to {dest_code}...")
    
    web_context = ""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(max_results=3)
        try:
            query = f"best connecting train route junction to change trains from {origin_code} to {dest_code} india rail info"
            web_context = f"\nLIVE FORUM CONSENSUS:\n{tavily.invoke(query)}\n"
        except Exception:
            pass

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(RouteConsensus)
    
    prompt = (
        f"There are no direct trains between {origin_code} and {dest_code}. "
        "Read the LIVE FORUM CONSENSUS to identify the best intermediate railway junction where travelers change trains.\n"
        "CRITICAL: The junction must be a major midway transit hub, NOT a station in the same city as the origin or destination.\n"
        "Output the official 2-4 letter station code."
    )
    
    try:
        return structured_llm.invoke([
            SystemMessage(content=prompt), 
            HumanMessage(content=f"Origin: {origin_code}\nDestination: {dest_code}{web_context}")
        ])
    except Exception:
        return None

def search_train_intelligence_fallback(origin_code: str, dest_code: str, travel_date: str) -> str:
    """Fallback: Fetches real-world schedules directly from web search if the live scraper crashes (e.g., >120 days)."""
    print(f"   [Train API] Live scraper failed (Likely >120 days ARP). Engaging Web Intelligence Fallback for {origin_code} -> {dest_code}...")
    
    web_data = ""
    if os.getenv("TAVILY_API_KEY"):
        tavily = TavilySearch(max_results=3)
        try:
            query = f"IRCTC express trains from {origin_code} to {dest_code} train name number departure days schedule"
            web_data = str(tavily.invoke(query))
        except Exception as e:
            print(f"   [Train API] Tavily query error: {e}")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    prompt = (
        f"You are an Indian Railways expert. A user requested trains from {origin_code} to {dest_code} on {travel_date}.\n"
        f"The live scraper failed (likely because {travel_date} is outside the 120-day IRCTC booking window).\n"
        f"Using this verified web intelligence:\n{web_data}\n\n"
        "Synthesize a helpful response:\n"
        "1. If direct express trains historically operate, list names, 5-digit numbers, departure times, and days of run.\n"
        "2. If no direct trains operate, clearly identify the primary connecting route and transfer hub based on the web data.\n"
        "3. Explicitly inform the user that booking only opens 120 days before travel."
    )
    
    return llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Provide verified train options.")]).content

def is_valid_layover(arr_time_str: str, dep_time_str: str) -> float:
    """Ensures layovers are between 2 and 7 hours."""
    try:
        arr = datetime.strptime(arr_time_str.strip(), "%H:%M")
        dep = datetime.strptime(dep_time_str.strip(), "%H:%M")
        if dep < arr:
            dep += timedelta(days=1)
        return (dep - arr).total_seconds() / 3600
    except Exception:
        return -1.0

@tool
def search_trains(origin_code: str, dest_code: str, travel_date: str) -> str:
    """Main routing engine: Direct trains -> Consensus Split Journey -> Web Intelligence Fallback."""
    parse_key = os.getenv("PARSE_API_KEY")

    def fetch_irctc(src: str, dst: str, raw_date: str) -> list:
        if not parse_key: return []
        
        # --- FIX: STRICT DATE FORMATTING ---
        try:
            formatted_date = dateutil.parser.parse(raw_date).strftime("%Y-%m-%d")
        except Exception:
            formatted_date = raw_date.strip()
            
        try:
            url = "https://api.parse.bot/scraper/4cb03185-74ea-4b58-9a5b-c0662c659aa0/search_trains"
            headers = {"X-API-Key": parse_key}
            params = {"from_station": src, "to_station": dst, "date": formatted_date}
            res = requests.get(url, headers=headers, params=params, timeout=15, verify=False)
            
            if res.status_code == 200:
                payload = res.json()
                if isinstance(payload, list): return payload
                data_block = payload.get("data", payload.get("result", payload))
                if isinstance(data_block, dict): return data_block.get("trains", [])
                elif isinstance(data_block, list): return data_block
                return payload.get("trains", [])
            elif res.status_code == 500:
                print(f"   [Train API] Scraper 500 error for {src} -> {dst} on {formatted_date}.")
                return [{"error": 500}] 
        except Exception as e:
            print(f"   [Train API] Request exception: {e}")
        return []

    print(f"   [Train API] Querying IRCTC for {origin_code} -> {dest_code} on {travel_date}...")
    direct_trains = fetch_irctc(origin_code, dest_code, travel_date)

    # 1. 500 Error Interception (Triggers if outside 120-day window)
    if direct_trains and direct_trains[0].get("error") == 500:
        return search_train_intelligence_fallback(origin_code, dest_code, travel_date)

    # 2. Direct Trains Found
    if direct_trains:
        print(f"   [Train API] Success! Found {len(direct_trains)} direct trains.")
        res_str = f"DIRECT TRAINS AVAILABLE ({origin_code} -> {dest_code}):\n"
        for i, t in enumerate(direct_trains[:5]):
            classes = ", ".join(t.get('classes', ['SL', '3A', '2A']))
            res_str += f"{i+1}. {t.get('train_name', 'Express')} ({t.get('train_number', 'N/A')}) | Classes: {classes} | Dep: {t.get('departure_time', 'N/A')} | Arr: {t.get('arrival_time', 'N/A')}\n"
        return res_str

    # 3. Web-Consensus Split Journey Engine
    print(f"   [Train API] 0 direct trains found. Initiating CONSENSUS SPLIT-JOURNEY engine...")
    consensus = get_web_consensus_junction(origin_code, dest_code)
    
    if consensus:
        junction = consensus.junction_code.upper()
        print(f"   [Train API] Web Consensus Junction: {junction}. Testing legs...")
        
        leg1 = fetch_irctc(origin_code, junction, travel_date)
        leg2 = fetch_irctc(junction, dest_code, travel_date)

        if (leg1 and leg1[0].get("error") == 500) or (leg2 and leg2[0].get("error") == 500):
            return search_train_intelligence_fallback(origin_code, dest_code, travel_date)

        for t1 in leg1:
            for t2 in leg2:
                layover = is_valid_layover(t1.get("arrival_time", "00:00"), t2.get("departure_time", "00:00"))
                if 2.0 <= layover <= 7.0:
                    print(f"   [Train API] Viable connecting train pair found via {junction}!")
                    return (
                        f"SPLIT JOURNEY REQUIRED (No Direct Trains) via {consensus.junction_name} ({junction}):\n"
                        f"Reasoning: {consensus.reasoning}\n\n"
                        f"LEG 1: {t1.get('train_name', 'Express')} ({t1.get('train_number', 'N/A')}) | Departs {origin_code}: {t1.get('departure_time')} -> Arrives {junction}: {t1.get('arrival_time')}\n"
                        f"[LAYOVER: {round(layover, 1)} hours at {junction}]\n"
                        f"LEG 2: {t2.get('train_name', 'Express')} ({t2.get('train_number', 'N/A')}) | Departs {junction}: {t2.get('departure_time')} -> Arrives {dest_code}: {t2.get('arrival_time')}\n"
                        f"INSTRUCTION: Advise the user to book using IRCTC's 'Connecting Journey Booking' feature."
                    )

    # 4. Ultimate Fallback (If all else fails)
    return search_train_intelligence_fallback(origin_code, dest_code, travel_date)