from typing import TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field
import operator

class TripDetails(BaseModel):
    """The master checklist of information needed to finalize a trip."""
    
    destination: Optional[str] = Field(default=None, description="The city, state, or country the user wants to visit.")
    origin_city: Optional[str] = Field(default=None, description="Where the user is departing from.")
    duration_days: Optional[int] = Field(default=None, description="The number of days the trip will last.")
    number_of_travelers: Optional[int] = Field(default=None, description="Total number of people traveling.")
    budget_tier: Optional[str] = Field(default=None, description="Budget preference: 'affordable', 'standard', or 'premium'.")
    
    # THE FIX: Stricter definitions for transport and cabs
    transport_mode: Optional[str] = Field(default=None, description="How the user is traveling FROM their origin TO the destination ('flight', 'train', 'bus', or 'car').")
    needs_airport_cab: Optional[bool] = Field(default=None, description="Does the user need a cab from their home to the airport/train station in their ORIGIN city?")
    needs_local_rental: Optional[bool] = Field(default=None, description="Does the user need to rent a vehicle for getting around LOCALLY at the DESTINATION?")
    
    halts: List[str] = Field(default_factory=list, description="List of base cities/towns planned for the trip.")

class GraphState(TypedDict):
    messages: Annotated[List[str], operator.add]
    trip_data: TripDetails
    itinerary_drafted: bool
    itinerary_approved: bool