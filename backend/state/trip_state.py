from typing import TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field
import operator

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
    halts: List[str] = Field(default_factory=list, description="List of base cities/towns planned for the trip.")

    # --- Halt-Aware Airport Routing (Hidden from User) ---
    entry_halt: Optional[str] = Field(default=None, description="The first halt where the trip begins.")
    exit_halt: Optional[str] = Field(default=None, description="The final halt where the trip concludes.")
    origin_iata: Optional[str] = Field(default=None, description="Departure airport IATA code from origin city (e.g., HYD).")
    arrival_iata: Optional[str] = Field(default=None, description="Arrival airport IATA code closest to Halt 1 (e.g., COK).")
    arrival_airport_name: Optional[str] = Field(default=None, description="Name of the airport closest to Halt 1.")
    return_departure_iata: Optional[str] = Field(default=None, description="Departure airport IATA code closest to the final halt.")
    return_departure_airport_name: Optional[str] = Field(default=None, description="Name of the airport closest to the final halt.")
    is_direct_flight_available: Optional[bool] = Field(default=None, description="Whether direct flights operate on the route.")
    outbound_last_mile_note: Optional[str] = Field(default=None, description="Road distance/time from arrival airport to Halt 1.")
    return_last_mile_note: Optional[str] = Field(default=None, description="Road distance/time from Final Halt to return airport.")
    special_transport_notes: Optional[str] = Field(default=None, description="Specific user constraints like non-stop, baggage, pet travel, etc.")

class GraphState(TypedDict):
    messages: Annotated[List[str], operator.add]
    trip_data: TripDetails
    itinerary_drafted: bool
    itinerary_approved: bool