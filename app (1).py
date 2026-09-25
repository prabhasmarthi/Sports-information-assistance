import os
import uvicorn
import requests
import json
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Sports Tools ---
@tool
def get_live_scores(sport: str) -> str:
    """Get live match scores and updates for a given sport (e.g., cricket, football, basketball)."""
    scores_db = {
        "cricket": "India vs Australia (T20): India 185/4 (18.2 overs) - India need 12 runs in 10 balls.",
        "football": "Real Madrid vs Barcelona (El Clasico): 2 - 1 (75th minute).",
        "basketball": "Lakers vs Warriors: 102 - 98 (4th Quarter, 3:15 remaining)."
    }
    return scores_db.get(sport.lower().strip(), f"No live matches currently tracked for '{sport}'.")

@tool
def get_team_schedule(team_name: str) -> str:
    """Look up upcoming match schedules, fixtures, and venues for a sports team."""
    schedules = {
        "india": "Next Match: India vs South Africa | Date: Sunday | Venue: Eden Gardens, Kolkata",
        "real madrid": "Next Match: Real Madrid vs Manchester City | Date: Wednesday | Venue: Santiago Bernabéu",
        "lakers": "Next Match: LA Lakers vs Boston Celtics | Date: Friday | Venue: Crypto.com Arena"
    }
    return schedules.get(team_name.lower().strip(), f"No upcoming schedule found for team '{team_name}'.")

tools = [get_live_scores, get_team_schedule]

# --- 2. Initialize Model & Guardrailed Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to sports information, live scores, and match schedules. "
        "For any other roles, topics, questions, or general knowledge outside of sports, "
        "you must say exactly: 'I am not authorized to answer questions outside of sports information.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the sports information agent")

def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(title="Real-Time Sports Information Agent")
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
