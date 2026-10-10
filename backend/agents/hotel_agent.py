from datetime import timedelta
import dateutil.parser
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state.trip_state import GraphState, HaltAllocation
from core.tools.hotel_tools import fetch_unified_accommodations
from prompts.agent_prompts import HOTEL_EVALUATION_PROMPT

# --- 1. Accommodation Schemas ---
class HotelOption(BaseModel):
    property_name: str
    inventory_type: Literal["Hotel", "Resort", "Vacation Rental / Airbnb", "Homestay / Villa"]
    price_per_night: str = Field(description="Formatted INR per night")
    total_stay_price: str = Field(description="Total INR across all nights for ALL travelers")
    rating_and_reviews: str = Field(description="e.g. 4.8/5 (340 reviews)")
    exact_location_commute: str = Field(description="Specific neighborhood and commute time to key sights")
    key_amenities: List[str] = Field(description="Top 4 distinct amenities e.g. Free Breakfast, Spa, Kitchen")
    permutation_reasoning: str = Field(description="Detailed economic and aesthetic justification based on the Permutation Matrix")
    booking_url: str

class HaltAccommodations(BaseModel):
    halt_name: str
    stay_dates: str
    budget_tier: str
    primary_recommendation: HotelOption
    alternative_contrasting_pick: HotelOption

class FinalTripAccommodations(BaseModel):
    halts_plan: List[HaltAccommodations]
    overall_accommodation_summary: str

# --- 2. Markdown Renderer ---
def render_hotel_markdown(data: FinalTripAccommodations, travelers: int) -> str:
    md = "Here are the optimal accommodation permutations based on your group size, budget, and exact itinerary schedule:\n\n"
    
    for halt in data.halts_plan:
        md += f"### Halt: {halt.halt_name} ({halt.stay_dates})\n"
        md += f"**Budget Tier:** {halt.budget_tier.title()} | **Travelers:** {travelers}\n\n"
        
        # Primary Pick
        prim = halt.primary_recommendation
        md += f"#### 🏆 Primary Recommendation: {prim.property_name}\n"
        md += f"* **Inventory Type:** {prim.inventory_type}\n"
        md += f"* **Pricing:** {prim.price_per_night} / night | **Total:** {prim.total_stay_price}\n"
        md += f"* **Guest Rating:** {prim.rating_and_reviews}\n"
        md += f"* **Location & Commute:** {prim.exact_location_commute}\n"
        md += f"* **Permutation Justification:** {prim.permutation_reasoning}\n"
        md += f"* **Key Amenities:** {', '.join(prim.key_amenities)}\n"
        md += f"* **Link:** [View Details]({prim.booking_url})\n\n"
        
        # Alternative Pick
        alt = halt.alternative_contrasting_pick
        md += f"#### 💡 Alternative Pick: {alt.property_name}\n"
        md += f"* **Inventory Type:** {alt.inventory_type}\n"
        md += f"* **Pricing:** {alt.price_per_night} / night | **Total:** {alt.total_stay_price}\n"
        md += f"* **Guest Rating:** {alt.rating_and_reviews}\n"
        md += f"* **Location & Commute:** {alt.exact_location_commute}\n"
        md += f"* **Permutation Justification:** {alt.permutation_reasoning}\n"
        md += f"* **Key Amenities:** {', '.join(alt.key_amenities)}\n"
        md += f"* **Link:** [View Details]({alt.booking_url})\n\n"
        md += "---\n\n"
        
    md += f"> **Agent Summary:** {data.overall_accommodation_summary}"
    return md

# --- 3. Main Agent Execution ---
# Inside run_hotel_agent in backend/agents/hotel_agent.py:

def run_hotel_agent(state: GraphState) -> dict:
    trip_data = state["trip_data"]
    
    # RECALL CHECK: Return cached hotels if already saved
    user_msgs = [m for m in state["messages"] if not str(m).startswith("SYSTEM_NOTE:")]
    last_msg = getattr(user_msgs[-1], 'content', str(user_msgs[-1])).lower() if user_msgs else ""
    
    if trip_data.saved_hotel_details and not any(k in last_msg for k in ["change", "different", "pool", "villa", "resort", "modify", "instead", "alternative"]):
        print("🏨 Hotel Agent: Recalling saved hotel matrix from state memory...")
        return {"messages": [trip_data.saved_hotel_details]}

    print("🏨 Hotel Agent: Running Deterministic-Retrieval & Probabilistic-Evaluation Engine...")
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    
    travelers = trip_data.number_of_travelers or 2
    budget = (trip_data.budget_tier or "standard").lower()
    start_date = trip_data.start_date or "TBD"

    halt_schedules = []
    curr_date = None
    if start_date and start_date.upper() not in ["TBD", "NONE", ""]:
        try:
            curr_date = dateutil.parser.parse(start_date)
        except Exception:
            curr_date = None

    halts_source = trip_data.halts if trip_data.halts else [HaltAllocation(halt_name=trip_data.destination or "Destination", nights=trip_data.duration_days or 3)]

    for h in halts_source:
        h_name = h.halt_name if isinstance(h, HaltAllocation) else str(h)
        h_nights = h.nights if isinstance(h, HaltAllocation) else 1
        
        if curr_date:
            c_in = curr_date.strftime("%Y-%m-%d")
            curr_date += timedelta(days=h_nights)
            c_out = curr_date.strftime("%Y-%m-%d")
        else:
            c_in = getattr(h, "check_in", "TBD") or "TBD"
            c_out = getattr(h, "check_out", "TBD") or "TBD"

        halt_schedules.append({
            "halt_name": h_name,
            "nights": h_nights,
            "check_in": c_in,
            "check_out": c_out
        })

    all_candidates_context = ""
    for schedule in halt_schedules:
        raw_inventory = fetch_unified_accommodations(
            halt_name=schedule["halt_name"], 
            check_in=schedule["check_in"], 
            check_out=schedule["check_out"], 
            adults=travelers, 
            budget_tier=budget
        )
        inventory_str = "\n".join([c.json() for c in raw_inventory])
        all_candidates_context += (
            f"=== HALT: {schedule['halt_name']} ===\n"
            f"Check-in: {schedule['check_in']} | Check-out: {schedule['check_out']} ({schedule['nights']} Nights)\n"
            f"AVAILABLE CANDIDATES:\n{inventory_str}\n\n"
        )

    structured_evaluator = llm.with_structured_output(FinalTripAccommodations)
    system_prompt = HOTEL_EVALUATION_PROMPT.format(
        current_draft=trip_data.saved_hotel_details or "No previous draft exists.",
        budget=budget, 
        travelers=travelers, 
        notes=trip_data.special_transport_notes or "None"
    )

    evaluation_result = structured_evaluator.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Evaluate these candidates and generate the final matrix:\n\n{all_candidates_context}")
    ])
    
    final_markdown = render_hotel_markdown(evaluation_result, travelers)
    updated_trip_data = trip_data.model_copy(update={"saved_hotel_details": final_markdown})

    return {
        "messages": [final_markdown],
        "trip_data": updated_trip_data
    }