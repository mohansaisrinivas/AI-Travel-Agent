from typing import TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field
import operator

class HaltAllocation(BaseModel):
    halt_name: str = Field(description="Name of the city or town.")
    nights: int = Field(default=1, description="Number of nights allocated to this halt.")
    check_in: Optional[str] = Field(default=None, description="Check-in date (YYYY-MM-DD).")
    check_out: Optional[str] = Field(default=None, description="Check-out date (YYYY-MM-DD).")

class TripDetails(BaseModel):
    """The master checklist of information needed to finalize a trip."""
    
    destination: Optional[str] = Field(default=None, description="The city, state, or country the user wants to visit.")
    origin_city: Optional[str] = Field(default=None, description="Where the user is departing from.")
    start_date: Optional[str] = Field(default=None, description="Start date of the trip (e.g., 'YYYY-MM-DD' or 'Oct 15').")
    return_date: Optional[str] = Field(default=None, description="Calculated return date (start_date + duration_days).")
    duration_days: Optional[int] = Field(default=None, description="The number of days the trip will last.")
    number_of_travelers: Optional[int] = Field(default=None, description="Total number of people traveling.")
    budget_tier: Optional[str] = Field(default=None, description="Budget preference: 'affordable', 'standard', or 'premium'.")
    transport_mode: Optional[str] = Field(default=None, description="How the user travels FROM origin TO destination ('flight', 'train', 'bus', or 'car').")
    needs_airport_cab: Optional[bool] = Field(default=None, description="Does the user need a cab to the origin airport/station?")
    needs_local_rental: Optional[bool] = Field(default=None, description="Does the user need a vehicle rental locally at the destination?")
    
    halts: List[HaltAllocation] = Field(default_factory=list, description="List of base cities with night allocations.")

    # --- Halt-Aware Airport Routing ---
    entry_halt: Optional[str] = Field(default=None, description="The first halt where the trip begins.")
    exit_halt: Optional[str] = Field(default=None, description="The final halt where the trip concludes.")
    origin_iata: Optional[str] = Field(default=None, description="Departure airport IATA code from origin city.")
    arrival_iata: Optional[str] = Field(default=None, description="Arrival airport IATA code closest to Halt 1.")
    arrival_airport_name: Optional[str] = Field(default=None, description="Name of the airport closest to Halt 1.")
    return_departure_iata: Optional[str] = Field(default=None, description="Departure airport IATA code closest to the final halt.")
    return_departure_airport_name: Optional[str] = Field(default=None, description="Name of the airport closest to the final halt.")
    is_direct_flight_available: Optional[bool] = Field(default=None, description="Whether direct flights operate on the route.")
    outbound_last_mile_note: Optional[str] = Field(default=None, description="Road distance/time from arrival airport to Halt 1.")
    return_last_mile_note: Optional[str] = Field(default=None, description="Road distance/time from Final Halt to return airport.")
    special_transport_notes: Optional[str] = Field(default=None, description="Specific user constraints like non-stop, baggage, pet travel, etc.")

    # --- Bus Routing ---
    origin_boarding_area: Optional[str] = Field(default=None, description="Exact neighborhood/area in origin city for bus boarding.")
    destination_drop_area: Optional[str] = Field(default=None, description="Nearest bus terminus/junction to destination if no direct route exists.")

    # --- Cached Artifacts (Prevents Re-running Searches) ---
    saved_itinerary: Optional[str] = Field(default=None, description="Cached text of the generated itinerary.")
    saved_transport_details: Optional[str] = Field(default=None, description="Cached text of the transport booking options.")
    saved_hotel_details: Optional[str] = Field(default=None, description="Cached text of the hotel permutation matrix.")

def merge_trip_data(existing: TripDetails, update: TripDetails) -> TripDetails:
    if not existing:
        return update
    if not update:
        return existing
    
    existing_dict = existing.dict(exclude_none=True)
    update_dict = update.dict(exclude_none=True)
    merged_dict = {**existing_dict, **update_dict}
    return TripDetails(**merged_dict)

class GraphState(TypedDict):
    messages: Annotated[List[str], operator.add]
    chat_summary: str
    trip_data: Annotated[TripDetails, merge_trip_data]
    itinerary_drafted: bool
    itinerary_approved: bool
    booking_stage: Optional[str]