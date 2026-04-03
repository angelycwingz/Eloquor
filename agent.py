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
# from google.adk.tools import google_search  # ADK built-in grounded search tool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools import FunctionTool
from google.adk.tools.agent_tool import AgentTool
 
from .prompts import (
    ORCHESTRATOR_PROMPT,
    JOB_INTEL_PROMPT,
    FEEDBACK_PROMPT,
)
from .tools import (
    save_session_to_memory,      
    get_past_sessions,
    format_scorecard_as_text,
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
# Sub-Agent 2: Interview Conductor
# No external tools needed — pure LLM conversation
# ---------------------------------------------------------------------------
# interview_agent = LlmAgent(
#     name="interview_agent",
#     model=MODEL,
#     description=(
#         "Plays the role of a professional interviewer. "
#         "Conducts a realistic 5-question mock interview turn by turn, "
#         "Asks ONE question at a time and waits for the user to respond "
#         "before asking the next question. Never simulates user answers."
#         "then outputs the full Q&A transcript as JSON."
#     ),
#     instruction=INTERVIEW_PROMPT,
#     output_key="transcript",  # saves output to session.state["transcript"]
# )

# ---------------------------------------------------------------------------
# Sub-Agent 3: Feedback Coach
# Has access to Firestore save tool
# ---------------------------------------------------------------------------
feedback_agent = LlmAgent(
    name="feedback_agent",
    model=MODEL,
    description=(
        "Scores the completed interview transcript and produces a scorecard."
    ),
    instruction=FEEDBACK_PROMPT,
    tools=[
        FunctionTool(save_session_to_memory),   
        FunctionTool(get_past_sessions),
        FunctionTool(format_scorecard_as_text),
    ],
    output_key="scorecard",  # saves output to session.state["scorecard"]
)

# ---------------------------------------------------------------------------
# Fix 3: SequentialAgent runs the 3-step pipeline in order
# This avoids the orchestrator LLM mixing tool types during delegation
# ---------------------------------------------------------------------------
# interview_pipeline = SequentialAgent(
#     name="interview_pipeline",
#     description=(
#         "Runs the full interview workflow in order: "
#         "job research → mock interview → feedback and scorecard."
#     ),
#     sub_agents=[
#         job_intel_agent,
#         interview_agent,
#         feedback_agent,
#     ],
# )

# ---------------------------------------------------------------------------
# Root Agent: Orchestrator
# ADK discovers this as `root_agent` via __init__.py
# ---------------------------------------------------------------------------
root_agent = LlmAgent(
    name="eloquor_orchestrator",
    model=MODEL,
    description=(
        "Eloquor — an AI career communication coach. "
    ),
    instruction=ORCHESTRATOR_PROMPT,
    tools=[
        AgentTool(agent=job_intel_agent),
        AgentTool(agent=feedback_agent),
    ],
)