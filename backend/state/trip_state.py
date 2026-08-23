# state/trip_state.py

from typing import TypedDict, Optional, List
from pydantic import BaseModel, Field

class TripDetails(BaseModel):
    """The master checklist of information needed to finalize a trip."""
    # Notice we now use default=None inside the Field definition
    destination: Optional[str] = Field(default=None, description="The city or country the user wants to visit.")
    origin_city: Optional[str] = Field(default=None, description="Where the user is departing from.")
    number_of_travelers: Optional[int] = Field(default=None, description="Total number of people traveling.")
    budget_tier: Optional[str] = Field(default=None, description="Budget preference: 'affordable', 'standard', or 'premium'.")
    transport_mode: Optional[str] = Field(default=None, description="Preferred transport: 'flight', 'train', 'bus', or 'car'.")
    needs_airport_cab: Optional[bool] = Field(default=None, description="Does the user need a cab from home to the station/airport?")
    needs_local_rental: Optional[bool] = Field(default=None, description="Does the user need a vehicle rental at the destination?")

class GraphState(TypedDict):
    """
    This is the global state that gets passed between all agents in LangGraph.
    """
    messages: List[str] 
    trip_data: TripDetails
    itinerary_drafted: bool
    itinerary_approved: bool