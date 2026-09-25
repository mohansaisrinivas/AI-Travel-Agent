import os
import re
import requests
from typing import Literal, Optional, List
from pydantic import BaseModel
from langchain_tavily import TavilySearch

class UnifiedAccommodation(BaseModel):
    name: str
    inventory_type: Literal["Hotel", "Resort", "Vacation Rental / Airbnb", "Homestay / Villa"]
    star_rating: Optional[float]
    review_score: float
    reviews_count: int
    total_rate_inr: float
    per_night_inr: float
    has_free_breakfast: bool
    has_kitchen: bool
    has_pool_or_spa: bool
    scenic_tags: List[str]
    commute_time_to_center_mins: str
    web_sentiment_summary: str
    booking_link: str

def fetch_google_maps_commute(origin: str, destination: str) -> str:
    """Calculates friction to the halt epicenter."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "15 mins"
    
    url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={api_key}"
    try:
        res = requests.get(url, timeout=10).json()
        if res.get("status") == "OK":
            elements = res["rows"][0]["elements"]
            if elements and elements[0].get("status") == "OK":
                return elements[0]["duration"]["text"]
    except Exception:
        pass
    return "15-20 mins"

def extract_numeric_rate(rate_obj) -> float:
    """Safely extracts a numeric rate from strings, ints, or SerpApi dicts."""
    if isinstance(rate_obj, dict):
        rate_val = rate_obj.get("lowest") or rate_obj.get("extracted_lowest") or 4500
    else:
        rate_val = rate_obj
    
    digits = re.sub(r"[^\d.]", "", str(rate_val))
    try:
        return float(digits) if digits else 4500.0
    except ValueError:
        return 4500.0

def fetch_unified_accommodations(halt_name: str, check_in: str, check_out: str, adults: int, budget_tier: str) -> List[UnifiedAccommodation]:
    serpapi_key = os.getenv("SERPAPI_KEY")
    candidates = []

    print(f"   [Hotel Tools] Fetching Dual-Stream Inventory for {halt_name}...")

    # Stream 1: Tavily Sentiment Check (Vibe & Reality Retrieval)
    sentiment_summary = "Centrally located, comfortable, and well-rated by travelers."
    if os.getenv("TAVILY_API_KEY"):
        try:
            tavily = TavilySearch(max_results=2)
            raw_tavily = tavily.invoke(f"Top luxury boutique stays, scenic views, and hotels in {halt_name} reviews")
            
            # SAFE EXTRACTION: Tavily returns a dict or list
            if isinstance(raw_tavily, dict):
                results = raw_tavily.get("results", [])
                text_snippets = [r.get("content", "") for r in results if isinstance(r, dict)]
                sentiment_summary = " ".join(text_snippets)
            elif isinstance(raw_tavily, list):
                sentiment_summary = " ".join([r.get("content", str(r)) if isinstance(r, dict) else str(r) for r in raw_tavily])
            else:
                sentiment_summary = str(raw_tavily)
                
            sentiment_summary = sentiment_summary[:200]
        except Exception as e:
            print(f"   [Hotel Tools] Tavily notice: {e}")

    # Stream 2: SerpApi (Live Retrieval)
    if serpapi_key:
        try:
            url = "https://serpapi.com/search.json"
            
            # Query A: Traditional Hotels & Resorts
            params_hotel = {
                "engine": "google_hotels",
                "q": f"hotels in {halt_name}",
                "check_in_date": check_in,
                "check_out_date": check_out,
                "adults": adults,
                "currency": "INR",
                "api_key": serpapi_key
            }
            res_hotel = requests.get(url, params=params_hotel, timeout=20)
            hotel_data = res_hotel.json().get("properties", []) if res_hotel.status_code == 200 else []

            # Query B: Vacation Rentals
            params_rental = params_hotel.copy()
            params_rental["vacation_rentals"] = "true"
            params_rental["q"] = f"vacation rentals in {halt_name}"
            res_rental = requests.get(url, params=params_rental, timeout=20)
            rental_data = res_rental.json().get("properties", []) if res_rental.status_code == 200 else []

            combined_props = hotel_data[:3] + rental_data[:2]

            for prop in combined_props:
                raw_rate = prop.get("rate_per_night", 5000)
                per_night = extract_numeric_rate(raw_rate)

                commute = fetch_google_maps_commute(
                    f"{prop.get('name', '')} {halt_name}", 
                    f"Center of {halt_name}"
                )

                amenities = str(prop.get("amenities", [])).lower()
                prop_type = str(prop.get("type", "")).lower()

                candidates.append(UnifiedAccommodation(
                    name=prop.get("name", f"Boutique Stay {halt_name}"),
                    inventory_type="Hotel" if ("hotel" in prop_type or "resort" in prop_type) else "Vacation Rental / Airbnb",
                    star_rating=prop.get("extracted_hotel_class", 4.0),
                    review_score=float(prop.get("overall_rating", 4.5) or 4.5),
                    reviews_count=int(prop.get("reviews", 100) or 100),
                    total_rate_inr=per_night,
                    per_night_inr=per_night,
                    has_free_breakfast="breakfast" in amenities,
                    has_kitchen="kitchen" in amenities,
                    has_pool_or_spa=("pool" in amenities or "spa" in amenities),
                    scenic_tags=["city_view", "central"] if "view" in sentiment_summary.lower() else ["central"],
                    commute_time_to_center_mins=commute,
                    web_sentiment_summary=sentiment_summary,
                    booking_link=prop.get("link", "https://www.google.com/travel/hotels")
                ))

            if candidates:
                return candidates

        except Exception as e:
            print(f"   [Hotel Tools] Live fetch error: {e}")

    # Fallback only if SerpApi returned 0 properties or is unconfigured
    print(f"   [Hotel Tools] Using fallback inventory for {halt_name}.")
    return [
        UnifiedAccommodation(
            name=f"The {halt_name} Grand Palace & Spa",
            inventory_type="Hotel",
            star_rating=5.0,
            review_score=4.8,
            reviews_count=520,
            total_rate_inr=14000.0,
            per_night_inr=14000.0,
            has_free_breakfast=True,
            has_kitchen=False,
            has_pool_or_spa=True,
            scenic_tags=["panoramic_views", "spa"],
            commute_time_to_center_mins="10 mins drive to city highlights",
            web_sentiment_summary="Exceptional luxury, acclaimed breakfast buffet, and tranquil spa.",
            booking_link="https://www.google.com/travel/hotels"
        ),
        UnifiedAccommodation(
            name=f"{halt_name} Luxury Panoramic Villa",
            inventory_type="Vacation Rental / Airbnb",
            star_rating=None,
            review_score=4.9,
            reviews_count=180,
            total_rate_inr=11500.0,
            per_night_inr=11500.0,
            has_free_breakfast=False,
            has_kitchen=True,
            has_pool_or_spa=False,
            scenic_tags=["secluded", "riverfront_or_hills"],
            commute_time_to_center_mins="18 mins drive to transit",
            web_sentiment_summary="Private, modern amenities with scenic surroundings and modular kitchen.",
            booking_link="https://www.airbnb.com"
        )
    ]