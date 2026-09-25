from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from state.trip_state import GraphState
from core.tools.hotel_tools import fetch_unified_accommodations
from prompts.agent_prompts import HOTEL_EVALUATION_PROMPT

# --- 1. Temporal Extraction Schemas ---
class HaltDateAllocation(BaseModel):
    halt_name: str = Field(description="The name of the halt/city.")
    nights: int = Field(description="Exact number of nights spent at this halt based on the itinerary.")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format.")
    check_out: str = Field(description="Check-out date in YYYY-MM-DD format.")

class ItineraryDateExtraction(BaseModel):
    allocations: List[HaltDateAllocation] = Field(description="List of exact date allocations for each halt in chronological order.")

# --- 2. LLM Output Schemas (Evaluation) ---
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

# --- 3. Markdown Renderer ---
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

# --- 4. Main Agent Execution ---
def run_hotel_agent(state: GraphState) -> dict:
    print("🏨 Hotel Agent: Running Deterministic-Retrieval & Probabilistic-Evaluation Engine...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    trip_data = state["trip_data"]
    
    halts = trip_data.halts or [trip_data.destination]
    travelers = trip_data.number_of_travelers or 2
    budget = (trip_data.budget_tier or "standard").lower()
    start_date = trip_data.start_date or "TBD"
    return_date = trip_data.return_date or "TBD"

    # Fetch recent conversation context (to read the drafted itinerary)
    recent_messages = "\n".join([getattr(m, 'content', str(m)) for m in state["messages"][-6:] if not str(m).startswith("SYSTEM_NOTE:")])

    # ------------------------------------------------------------------
    # PHASE 1: TEMPORAL EXTRACTION
    # ------------------------------------------------------------------
    print("   [Hotel Agent] Reading drafted itinerary to extract exact check-in/check-out dates per halt...")
    date_extractor = llm.with_structured_output(ItineraryDateExtraction)
    
    extraction_prompt = f"""
    You are a Temporal Booking Assistant. 
    Read the provided conversation/itinerary and determine EXACTLY how many nights the user spends at each of these Halts: {halts}.
    
    CRITICAL RULES:
    1. Overall Start Date (Check-in Halt 1): {start_date}
    2. Overall Return Date (Check-out Final Halt): {return_date}
    3. The Check-out date of Halt 1 MUST be the exact Check-in date of Halt 2.
    4. Calculate the dates using YYYY-MM-DD format. Do not guess evenly—read the itinerary text to see how the days were distributed!
    """

    try:
        temporal_plan = date_extractor.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=f"Conversation Context:\n{recent_messages}")
        ])
        halt_schedules = temporal_plan.allocations
        
        for alloc in halt_schedules:
            print(f"      -> {alloc.halt_name}: {alloc.nights} Nights ({alloc.check_in} to {alloc.check_out})")
            
    except Exception as e:
        print(f"   [Hotel Agent ERROR] Date extraction failed: {e}. Cannot proceed with booking.")
        return {"messages": ["Sorry, I had trouble parsing the exact check-in dates from our itinerary plan. Let me try generating the plan again."]}

    # ------------------------------------------------------------------
    # PHASE 2: DETERMINISTIC RETRIEVAL
    # ------------------------------------------------------------------
    all_candidates_context = ""
    for schedule in halt_schedules:
        raw_inventory = fetch_unified_accommodations(
            halt_name=schedule.halt_name, 
            check_in=schedule.check_in, 
            check_out=schedule.check_out, 
            adults=travelers, 
            budget_tier=budget
        )
        
        inventory_str = "\n".join([c.json() for c in raw_inventory])
        
        all_candidates_context += (
            f"=== HALT: {schedule.halt_name} ===\n"
            f"Check-in: {schedule.check_in} | Check-out: {schedule.check_out} ({schedule.nights} Nights)\n"
            f"AVAILABLE CANDIDATES:\n{inventory_str}\n\n"
        )

    # ------------------------------------------------------------------
    # PHASE 3: PROBABILISTIC EVALUATION (LLM Matrix)
    # ------------------------------------------------------------------
    print("   [Hotel Agent] Normalization complete. Passing payload to LLM Permutation Matrix...")
    structured_evaluator = llm.with_structured_output(FinalTripAccommodations)
    
    system_prompt = HOTEL_EVALUATION_PROMPT.format(
        budget=budget, 
        travelers=travelers, 
        notes=trip_data.special_transport_notes or "None"
    )

    evaluation_result = structured_evaluator.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Evaluate these candidates and generate the final matrix:\n\n{all_candidates_context}")
    ])
    
    final_markdown = render_hotel_markdown(evaluation_result, travelers)

    return {"messages": [final_markdown]}