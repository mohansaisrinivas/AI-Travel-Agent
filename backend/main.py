import os
import sqlite3
import bcrypt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()
set_llm_cache(InMemoryCache())

from core.graph import build_travel_graph
from state.trip_state import TripDetails

app = FastAPI(title="AI Travel Agency API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/templates")))

# --- 1. LANGGRAPH MEMORY DB ---
conn = sqlite3.connect("travel_checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
travel_graph = build_travel_graph(checkpointer=memory)

# --- 2. USER AUTHENTICATION DB ---
auth_conn = sqlite3.connect("users.db", check_same_thread=False)
auth_cursor = auth_conn.cursor()
auth_cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL
    )
""")
auth_conn.commit()

# --- PYDANTIC MODELS ---
class AuthRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    status: str
    reply: str

# --- AUTH ENDPOINTS ---
@app.post("/register")
def register_user(request: AuthRequest):
    username = request.username.strip().lower()
    password = request.password.encode('utf-8')
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required.")

    auth_cursor.execute("SELECT username FROM users WHERE username = ?", (username,))
    if auth_cursor.fetchone():
        raise HTTPException(status_code=400, detail="Username already exists. Please choose a unique username.")
    
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password, salt)
    
    auth_cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed.decode('utf-8')))
    auth_conn.commit()
    
    return {"status": "success", "message": "Account created successfully!", "user_id": username}

@app.post("/login")
def login_user(request: AuthRequest):
    username = request.username.strip().lower()
    password = request.password.encode('utf-8')
    
    auth_cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    record = auth_cursor.fetchone()
    
    if not record:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
        
    stored_hash = record[0].encode('utf-8')
    
    if bcrypt.checkpw(password, stored_hash):
        return {"status": "success", "message": "Login successful!", "user_id": username}
    else:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

# --- SESSION & CHAT ENDPOINTS ---
@app.get("/", response_class=HTMLResponse)
def serve_frontend(request: Request):
    """Serves the Tailwind CSS web UI."""
    return templates.TemplateResponse(request, "index.html")

@app.get("/session/{username}")
def get_user_session(username: str):
    """Fetches the user's current trip details from the SQLite Checkpointer."""
    try:
        config = {"configurable": {"thread_id": username.lower()}}
        current_state = travel_graph.get_state(config)
        
        if not current_state.values:
            return {"status": "empty"}
            
        trip_data = current_state.values.get("trip_data")
        
        if trip_data and hasattr(trip_data, 'dict'):
            trip_data_dict = trip_data.dict(exclude_none=True)
        else:
            trip_data_dict = trip_data or {}
            
        return {
            "status": "active", 
            "trip_data": trip_data_dict,
            "itinerary_drafted": current_state.values.get("itinerary_drafted", False),
            "itinerary_approved": current_state.values.get("itinerary_approved", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.user_id}}
        
        current_state = travel_graph.get_state(config)
        if not current_state.values:
            travel_graph.update_state(config, {
                "messages": [],
                "chat_summary": "No user preferences captured yet.",
                "trip_data": TripDetails(), 
                "itinerary_drafted": False,
                "itinerary_approved": False,
                "booking_stage": "not_started"
            })
            current_state = travel_graph.get_state(config)

        previous_msg_count = len(current_state.values.get("messages", []))

        # The message goes directly to the Graph for the LLM to process
        result_state = travel_graph.invoke({"messages": [request.message]}, config=config)
        
        all_messages = result_state.get("messages", [])
        new_messages = all_messages[previous_msg_count:]
        
        valid_replies = []
        for msg in new_messages:
            if getattr(msg, 'type', '') == 'human':
                continue
            content = getattr(msg, 'content', str(msg))
            if content == request.message:
                continue
            if content and not content.startswith("SYSTEM_NOTE:"):
                valid_replies.append(content)

        if valid_replies:
            reply_content = "\n\n---\n\n".join(valid_replies)
        else:
            reply_content = "I'm here to help with your trip!"

        return ChatResponse(status="success", reply=reply_content)

    except Exception as e:
        print(f"❌ API Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting Full-Stack Python App on port {port} ...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)