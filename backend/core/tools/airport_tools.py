from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

class MultiHaltConnectivityResult(BaseModel):
    origin_city: str
    origin_iata: str = Field(description="3-letter IATA code for departure airport from origin, e.g., HYD")
    origin_airport_name: str

    entry_halt: str = Field(description="The first halt where travelers arrive.")
    arrival_iata: str = Field(description="3-letter IATA code of airport closest to Halt 1.")
    arrival_airport_name: str
    outbound_last_mile_note: str = Field(
        description="Road distance and travel duration from arrival airport to Halt 1 (e.g., '35 km / 1 hr drive from COK to Fort Kochi')."
    )

    exit_halt: str = Field(description="The final halt where travelers conclude their trip.")
    return_departure_iata: str = Field(
        description="3-letter IATA code of airport closest to the Final Halt for departure back to origin."
    )
    return_departure_airport_name: str
    return_last_mile_note: str = Field(
        description="Road distance and travel duration from Final Halt to the return airport."
    )

    is_direct_flight_available: bool = Field(
        description="True if scheduled direct non-stop flights typically operate on this route."
    )

def resolve_multi_halt_airports(origin: str, entry_halt: str, exit_halt: str) -> MultiHaltConnectivityResult:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    structured_llm = llm.with_structured_output(MultiHaltConnectivityResult)

    prompt = (
        "You are an aviation geography expert. Given an origin city, an Entry Halt (Halt 1), and an Exit Halt (Final Halt):\n"
        "1. Identify the primary commercial airport and IATA code for the origin city.\n"
        "2. Identify the closest commercial airport and IATA code to Halt 1 for arrival, including road transit distance/time.\n"
        "3. Identify the closest commercial airport and IATA code to the Final Halt for departure back to origin, including road transit distance/time.\n"
        "4. Note whether direct flights typically exist between origin and arrival airport."
    )

    return structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Origin: {origin}\nEntry Halt (Halt 1): {entry_halt}\nExit Halt (Final Halt): {exit_halt}")
    ])