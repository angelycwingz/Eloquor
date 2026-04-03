"""
Tools for Eloquor — simplified for prototype.
Using in-memory storage instead of Firestore. No GCP setup needed.
"""

import json

# Simple in-memory store — lives as long as the container is running
_session_store = {}


def save_session_to_memory(
    user_id: str,
    target_role: str,
    overall_score: float,
    scorecard_text: str,
) -> dict:
    """
    Saves a completed interview session to memory for this container session.
    Call this after feedback_agent produces the final scorecard.
    """
    _session_store[user_id] = {
        "target_role": target_role,
        "overall_score": overall_score,
        "scorecard_text": scorecard_text,
    }
    return {"status": "saved", "user_id": user_id}


def get_past_sessions(user_id: str) -> dict:
    """
    Fetches past session data for the user from memory.
    """
    session = _session_store.get(user_id)
    if session:
        return {"sessions": [session]}
    return {"sessions": []}


def format_scorecard_as_text(scorecard_json: str) -> str:
    """
    Converts a raw JSON scorecard into clean readable markdown.
    Only call this if feedback_agent produced raw JSON instead of markdown.
    """
    try:
        data = json.loads(scorecard_json)
    except json.JSONDecodeError:
        return scorecard_json  # already text, return as-is

    lines = ["## Eloquor Scorecard\n"]
    lines.append(f"Overall Score: {data.get('overall_score', 'N/A')} / 5.0\n")
    for s in data.get("strengths", []):
        lines.append(f"- {s}")
    for i in data.get("improvements", []):
        lines.append(f"- {i}")
    return "\n".join(lines)