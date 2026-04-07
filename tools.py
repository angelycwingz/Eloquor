"""
Tools for Eloquor
Using in-memory storage instead of Firestore. No GCP setup needed.
"""

import os
import json
import logging
from datetime import date
 

# Simple in-memory store — lives as long as the container is running
_session_store = {}

def _get_connection():
    """
    Returns a psycopg2 connection to AlloyDB.
    Reads credentials from environment variables.
    """
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        dbname=os.getenv("DB_NAME", "postgres"),
        port=int(os.getenv("DB_PORT", "5432")),
        sslmode="require",
    )

def _generate_embedding(text: str) -> list:
    """
    Generates a vector embedding for the given text using Vertex AI
    text-embedding-004 model via the AlloyDB google_ml_integration extension.
    Falls back to None if embedding generation fails.
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT embedding('text-embedding-004', %s)::text",
            (text[:8000],)  # truncate to stay within token limits
        )
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        return result  # returns as string like "{0.123, -0.456, ...}"
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return None


def check_interview_complete(answers_so_far: int) -> dict:
    """
    Checks whether all 5 interview questions have been answered.
    Call this after every answer the user gives during the interview.
    Never rely on counting from conversation history — always use this tool.
 
    Args:
        answers_so_far: how many questions the user has answered so far.
                        Start at 0. Increment by 1 after each user answer.
 
    Returns:
        A dict with:
          complete      — True if all 5 questions are answered, False otherwise
          answers_so_far — the count passed in
          remaining     — how many questions are still left
          message       — a plain English summary for the orchestrator
    """
    total_questions = 5
    remaining = total_questions - answers_so_far
 
    if answers_so_far >= total_questions:
        return {
            "complete": True,
            "answers_so_far": answers_so_far,
            "remaining": 0,
            "message": "All 5 questions have been answered. Close the interview and compile the transcript."
        }
    else:
        return {
            "complete": False,
            "answers_so_far": answers_so_far,
            "remaining": remaining,
            "message": f"Interview in progress. {remaining} question(s) remaining. Ask the next question."
        }


def save_session_to_memory(
    user_id: str,
    target_role: str,
    overall_score: float,
    scorecard_text: str,
    transcript: str = "",
    experience_level: str = "",
) -> dict:
    """
    Saves a completed interview session to AlloyDB.
    Embedding generation is skipped for now — added back once basic save works.
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO sessions
                (user_id, session_date, target_role, experience_level,
                 overall_score, transcript, scorecard_text)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                date.today(),
                target_role,
                experience_level,
                overall_score,
                transcript,
                scorecard_text,
            )
        )
        session_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return {
            "status": "saved",
            "session_id": session_id,
            "user_id": user_id
        }

    except Exception as e:
        logging.error(f"Failed to save session: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def get_past_sessions(user_id: str, limit: int = 5) -> dict:
    """
    Fetches the user's most recent interview sessions from AlloyDB.
    Call this when the user asks about their previous practice history.
 
    Args:
        user_id: The unique identifier for the user.
        limit:   How many past sessions to return. Defaults to 5.
 
    Returns:
        A dict with a sessions list ordered by most recent first.
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()
 
        cur.execute(
            """
            SELECT id, session_date, target_role, experience_level, overall_score
            FROM sessions
            WHERE user_id = %s
            ORDER BY session_date DESC
            LIMIT %s
            """,
            (user_id, limit)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
 
        sessions = [
            {
                "session_id": row[0],
                "date": str(row[1]),
                "target_role": row[2],
                "experience_level": row[3],
                "overall_score": row[4],
            }
            for row in rows
        ]
 
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
 
    except Exception as e:
        logging.error(f"Failed to fetch sessions: {e}")
        return {"sessions": [], "error": str(e)}


def search_similar_sessions(user_id: str, query: str, limit: int = 3) -> dict:
    """
    Searches the user's past sessions semantically using vector similarity.
    Use this when the user asks things like:
      - "when did I struggle with technical questions?"
      - "show me my weakest interview"
      - "how have I done with behavioural questions?"
 
    Args:
        user_id: The unique identifier for the user.
        query:   A natural language description of what to search for.
        limit:   How many similar sessions to return. Defaults to 3.
 
    Returns:
        A dict with the most semantically similar past sessions.
    """
    try:
        query_embedding = _generate_embedding(query)
        if not query_embedding:
            return {"sessions": [], "error": "Could not generate query embedding."}
 
        conn = _get_connection()
        cur = conn.cursor()
 
        cur.execute(
            """
            SELECT id, session_date, target_role, overall_score, scorecard_text,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM sessions
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, user_id, query_embedding, limit)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
 
        sessions = [
            {
                "session_id": row[0],
                "date": str(row[1]),
                "target_role": row[2],
                "overall_score": row[3],
                "scorecard_summary": row[4][:500] if row[4] else "",
                "similarity_score": round(float(row[5]), 3),
            }
            for row in rows
        ]
 
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
 
    except Exception as e:
        logging.error(f"Semantic search failed: {e}")
        return {"sessions": [], "error": str(e)}
 


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