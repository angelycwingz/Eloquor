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
    Embedding is generated inside SQL.
    Falls back safely if embedding fails.
    """
    try:
        text_for_embedding = scorecard_text or transcript
        if not text_for_embedding:
            text_for_embedding = "empty session"

        with _get_connection() as conn:
            with conn.cursor() as cur:

                try:
                    # ✅ Insert with embedding
                    cur.execute(
                        """
                        INSERT INTO sessions
                            (user_id, session_date, target_role, experience_level,
                             overall_score, transcript, scorecard_text, embedding)
                        VALUES
                            (%s, %s, %s, %s, %s, %s, %s,
                             embedding('text-embedding-004', %s))
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
                            text_for_embedding,
                        )
                    )

                except Exception as embed_error:
                    logging.error(f"Embedding insert failed, fallback used: {embed_error}")

                    # ✅ fallback insert (no embedding)
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
    Semantic search using AlloyDB vector embeddings.
    Uses SQL-based embedding (no Python embedding).
    """
    try:
        with _get_connection() as conn:
            with conn.cursor() as cur:

                cur.execute(
                    """
                    SELECT id, session_date, target_role, overall_score, scorecard_text,
                           1 - (embedding <=> embedding('text-embedding-004', %s)) AS similarity
                    FROM sessions
                    WHERE user_id = %s
                    ORDER BY embedding <=> embedding('text-embedding-004', %s)
                    LIMIT %s
                    """,
                    (query, user_id, query, limit)
                )

                rows = cur.fetchall()

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

def schedule_practice_session(
    role: str,
    date: str,
    time: str,
    duration_minutes: int = 60,
) -> dict:
    """
    Schedules a mock interview practice session in Google Calendar.
    Call this when the user asks to schedule or book a practice session.

    Args:
        role:             The job role the user is preparing for.
        date:             The date for the session e.g. 'tomorrow', '2026-04-10', 'next Monday'.
        time:             The time for the session e.g. '6pm', '18:00', '10:30am'.
        duration_minutes: How long the session should be in minutes. Defaults to 60.

    Returns:
        A dict with status, event link, and scheduled time if successful.
    """
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from datetime import datetime, timedelta
        import json
        import re
        from dateutil import parser as dateparser

        # ── Load credentials ────────────────────────────────────────────
        key_json = os.getenv("CALENDAR_KEY_JSON")
        if not key_json:
            # Try loading from file for local dev
            if os.path.exists("calendar-key.json"):
                with open("calendar-key.json") as f:
                    key_data = json.load(f)
            else:
                return {"status": "error", "message": "Calendar credentials not found."}
        else:
            key_data = json.loads(key_json)

        credentials = service_account.Credentials.from_service_account_info(
            key_data,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )

        calendar_id = os.getenv("CALENDAR_ID")
        if not calendar_id:
            return {"status": "error", "message": "CALENDAR_ID not set in environment."}

        # ── Parse date and time ─────────────────────────────────────────
        now = datetime.now()

        # Handle relative dates
        date_lower = date.lower().strip()
        if date_lower == "tomorrow":
            base_date = now + timedelta(days=1)
        elif date_lower == "today":
            base_date = now
        elif "next" in date_lower:
            day_name = date_lower.replace("next", "").strip()
            days_ahead = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
            }.get(day_name, 1)
            current_day = now.weekday()
            days_until = (days_ahead - current_day + 7) % 7
            if days_until == 0:
                days_until = 7
            base_date = now + timedelta(days=days_until)
        else:
            base_date = dateparser.parse(date)
            if not base_date:
                base_date = now + timedelta(days=1)

        # Parse time
        time_str = time.strip().upper().replace(" ", "")
        try:
            parsed_time = dateparser.parse(time_str)
            hour = parsed_time.hour
            minute = parsed_time.minute
        except Exception:
            # Default to 6pm if parsing fails
            hour = 18
            minute = 0

        start_dt = base_date.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        # ── Build event ─────────────────────────────────────────────────
        event = {
            "summary": f"Eloquor Practice — {role}",
            "description": (
                f"Mock interview practice session for {role}.\n\n"
                f"Tips to prepare:\n"
                f"- Review the company's recent news and products\n"
                f"- Practice the STAR method for behavioural questions\n"
                f"- Revise core technical concepts for the role\n\n"
                f"Scheduled by Eloquor AI Career Coach."
            ),
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 30},
                    {"method": "email", "minutes": 60},
                ],
            },
        }

        # ── Create event ─────────────────────────────────────────────────
        service = build("calendar", "v3", credentials=credentials)
        created_event = service.events().insert(
            calendarId=calendar_id,
            body=event
        ).execute()

        return {
            "status": "scheduled",
            "event_title": event["summary"],
            "date": start_dt.strftime("%A, %d %B %Y"),
            "time": start_dt.strftime("%I:%M %p"),
            "duration": f"{duration_minutes} minutes",
            "event_link": created_event.get("htmlLink", ""),
            "message": f"Practice session scheduled for {start_dt.strftime('%A, %d %B at %I:%M %p')}."
        }

    except Exception as e:
        logging.error(f"Calendar scheduling failed: {e}")
        return {"status": "error", "message": str(e)}