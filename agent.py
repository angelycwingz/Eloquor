"""
Eloquor — AI Career Communication Coach
Built with Google Agent Development Kit (ADK)
 
Architecture:
  root_agent (Orchestrator — LlmAgent)
    ├── job_intel_agent   (LlmAgent + google_search tool)
    ├── interview_agent   (LlmAgent)
    └── feedback_agent    (LlmAgent + save_session_to_firestore tool)
 
ADK discovers `root_agent` via the package __init__.py.
"""

import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.agents import SequentialAgent
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
 
from .prompts import (
    ORCHESTRATOR_PROMPT,
    JOB_INTEL_PROMPT,
    FEEDBACK_PROMPT,
)
from .tools import (
    check_interview_complete,
    save_session_to_memory, 
    get_past_sessions,
    search_similar_sessions,     
    format_scorecard_as_text,
    schedule_practice_session
)

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()

google_search_tool = GoogleSearchTool()

MODEL = os.getenv("MODEL")

logging.info("Chabot Invoked")

# ---------------------------------------------------------------------------
# Sub-Agent 1: Job Intel
# Uses ADK's built-in google_search to fetch real job data
# ---------------------------------------------------------------------------
job_intel_agent = LlmAgent(
    name="job_intel_agent",
    model=MODEL,
    description=(
        "Researches the target company and role and produces 5 tailored interview questions."
    ),
    instruction=JOB_INTEL_PROMPT,
    tools=[google_search_tool],  # ADK built-in — enables grounded web search
    output_key="job_intel",  # saves output to session.state["job_intel"]
)


# ---------------------------------------------------------------------------
# Sub-Agent 2: Feedback Coach
# Has access to Firestore save tool
# ---------------------------------------------------------------------------
feedback_agent = LlmAgent(
    name="feedback_agent",
    model=MODEL,
    description=(
        "Reads the interview transcript, scores every answer on content, "
        "structure, and language, produces a detailed scorecard and "
        "7-day improvement plan, then saves the session to AlloyDB."
    ),
    instruction=FEEDBACK_PROMPT,
    tools=[
        FunctionTool(save_session_to_memory),
        FunctionTool(get_past_sessions),
        FunctionTool(search_similar_sessions),
        FunctionTool(format_scorecard_as_text),
    ],
    output_key="scorecard",  # saves output to session.state["scorecard"]
)

memory_agent = LlmAgent(
    name="memory_agent",
    model=MODEL,
    description=(
        "Analyzes user's past interview sessions and provides performance insights, trends, and coaching feedback."
    ),
    instruction=(
        "You are Eloquor Memory Coach.\n\n"

        "Your job is to analyze past interview sessions and answer questions like:\n"
        "- How has the user been performing lately?\n"
        "- Are they improving or declining?\n"
        "- What are their strengths and weaknesses?\n\n"

        "TOOLS AVAILABLE:\n"
        "- get_past_sessions: fetch recent sessions\n"
        "- search_similar_sessions: find relevant past sessions\n\n"

        "INSTRUCTIONS:\n"
        "1. Always call get_past_sessions first.\n"
        "2. If needed, call search_similar_sessions for deeper insight.\n"
        "3. Analyze trends in scores, roles, and performance.\n"
        "4. search the user's session from their name in lower case.\n\n"
        "4. Respond like a coach, not like a database.\n\n"

        "OUTPUT STYLE:\n"
        "- Start with a summary (1–2 lines)\n"
        "- Then insights (bullets)\n"
        "- Then actionable advice\n\n"

        "Be encouraging but honest."
    ),
    tools=[
        FunctionTool(get_past_sessions),
        FunctionTool(search_similar_sessions),
    ],
)

# ---------------------------------------------------------------------------
# Root Agent: Orchestrator
# ADK discovers this as `root_agent` via __init__.py
# ---------------------------------------------------------------------------
root_agent = LlmAgent(
    name="eloquor_orchestrator",
    model=MODEL,
    description=(
        "Eloquor — an AI career communication coach with Google Calendar integration "
    ),
    instruction=ORCHESTRATOR_PROMPT,
    tools=[
        FunctionTool(check_interview_complete),
        FunctionTool(schedule_practice_session),
        AgentTool(agent=job_intel_agent),
        AgentTool(agent=feedback_agent),
        AgentTool(agent=memory_agent),
    ],
)