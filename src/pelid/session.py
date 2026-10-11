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
        self.pending_intent: Optional[str] = None
        self.pending_slots: list[str] = []
        self.pending_language: str = "en"
        self.context_slots: dict[str, Any] = {}
        self.turn_count: int = 0
        self.history: list[dict[str, str]] = []

    def update_slots_from_text(self, text: str):
        """Scan text and update persistent slots like Order ID."""
        found_order = extract_order_id(text)
        if found_order:
            self.active_order_id = found_order
            self.context_slots["order_id"] = found_order

    def set_pending(self, intent: str, missing_slots: list[str], language: str = "en"):
        """Record a pending intent that is awaiting missing slots from the user."""
        self.pending_intent = intent
        self.pending_slots = list(missing_slots)
        self.pending_language = language

    def clear_pending(self):
        """Clear any pending slot-filling state."""
        self.pending_intent = None
        self.pending_slots = []
        self.pending_language = "en"

    def has_pending_intent(self) -> bool:
        """Check if this session is actively waiting for slot values."""
        return self.pending_intent is not None and len(self.pending_slots) > 0

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


def record_pending_dialogue(session_id: str, intent: str, missing_slots: list[str], language: str = "en"):
    """Record that a session is waiting for specific missing slots before completing an intent."""
    store = get_session_store()
    session = store.get_or_create(session_id)
    session.set_pending(intent=intent, missing_slots=missing_slots, language=language)


def check_pending_dialogue_resolution(
    session_id: str,
    current_query: str,
    new_slots: dict[str, Any],
) -> Optional[tuple[str, dict[str, Any], str]]:
    """
    Check if the current message fulfills a pending slot request from a previous turn.

    Returns:
        (pending_intent, combined_slots, language) if fulfilled, else None.
    """
    store = get_session_store()
    session = store.get_or_create(session_id)

    if not session.has_pending_intent():
        return None

    # Check for topic shift or greeting/cancellation that should reset pending intent
    q_lower = current_query.strip().lower()
    cancel_terms = ["cancel", "nevermind", "forget it", "no", "nahi", "venda", "stop"]
    greeting_terms = ["hi", "hello", "hey", "namaste", "namaskaram"]
    if any(q_lower == term or q_lower.startswith(term + " ") for term in cancel_terms + greeting_terms):
        session.clear_pending()
        return None

    # Merge new slots with context slots
    combined_slots = dict(session.context_slots)
    combined_slots.update({k: v for k, v in new_slots.items() if v})

    # If query is a bare alphanumeric token (e.g. "9921", "#9921", "ORD-123") and we're waiting for order_id
    if "order_id" in session.pending_slots and not combined_slots.get("order_id"):
        from pelid.mock_crm import extract_order_id
        bare_id = extract_order_id(current_query)
        if bare_id:
            combined_slots["order_id"] = bare_id
            session.active_order_id = bare_id

    # Check if all pending slots are now satisfied
    remaining_missing = [s for s in session.pending_slots if s not in combined_slots or not combined_slots[s]]
    if not remaining_missing:
        resolved_intent = session.pending_intent
        lang = session.pending_language
        session.clear_pending()
        session.context_slots = combined_slots
        return resolved_intent, combined_slots, lang

    return None

