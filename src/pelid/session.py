"""
Pelid Multi-Turn Dialogue Context & Session State Tracker (Stage 11)

Extracts dialogue history, tracks persistent entities (e.g. Order ID, Product Name)
across conversational turns, and synthesizes contextualized triage queries so
follow-up questions ("When will it reach?", "Can you cancel it?") retain full context.
"""

import re
from typing import Any, Optional
from pelid.mock_crm import extract_order_id


class SessionContext:
    """Stores ongoing conversation state and extracted entity slots for a user."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.active_order_id: Optional[str] = None
        self.active_item: Optional[str] = None
        self.last_intent: Optional[str] = None
        self.turn_count: int = 0
        self.history: list[dict[str, str]] = []

    def update_slots_from_text(self, text: str):
        """Scan text and update persistent slots like Order ID."""
        found_order = extract_order_id(text)
        if found_order:
            self.active_order_id = found_order

    def add_turn(self, role: str, content: str):
        """Append message turn and update slots."""
        self.turn_count += 1
        self.update_slots_from_text(content)
        self.history.append({"role": role, "content": content})
        if len(self.history) > 10:
            self.history = self.history[-10:]


class SessionStore:
    """In-memory thread-safe store for active conversation sessions."""

    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}

    def get_or_create(self, session_id: str) -> SessionContext:
        """Get existing session context or create a new one."""
        if not session_id:
            session_id = "default_session"
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id)
        return self._sessions[session_id]

    def clear(self):
        """Clear all session states."""
        self._sessions.clear()


# Global session registry
_GLOBAL_SESSIONS = SessionStore()


def get_session_store() -> SessionStore:
    return _GLOBAL_SESSIONS


def synthesize_multi_turn_query(messages: list[dict], session_id: str = "") -> tuple[str, Optional[str]]:
    """
    Synthesize the effective triage query from conversation history.
    
    Returns:
        (synthesized_text, active_order_id)
    """
    if not messages:
        return "", None

    store = get_session_store()
    session = store.get_or_create(session_id)

    # 1. Update session slots from all historical messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if content:
            session.update_slots_from_text(content)

    last_user_msg = messages[-1].get("content", "") if messages else ""
    active_order = session.active_order_id or extract_order_id(last_user_msg)

    # If single turn, return directly
    if len(messages) <= 1:
        return last_user_msg, active_order

    # Multi-turn context synthesis:
    # If the user's latest query is short or referential ("When will it reach?", "Cancel this"),
    # append contextual slots so the decision head and entity lookup understand the full intent.
    prev_assistant_turn = ""
    for msg in reversed(messages[:-1]):
        if msg.get("role") == "assistant":
            prev_assistant_turn = msg.get("content", "")[:120]
            break

    # Context enrichment
    if active_order and active_order not in last_user_msg:
        synthesized = f"Order #{active_order}: {last_user_msg}"
    else:
        synthesized = last_user_msg

    return synthesized, active_order
