"""
Tests for Pelid Multi-Turn Dialogue State Machine & Slot-Filling
"""

import pytest
from fastapi.testclient import TestClient
from pelid.proxy import app
from pelid.session import (
    get_session_store,
    record_pending_dialogue,
    check_pending_dialogue_resolution,
)


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear in-memory sessions before each test."""
    store = get_session_store()
    store.clear()
    yield
    store.clear()


def test_dialogue_state_record_and_fulfillment():
    """Verify pending slot request is registered and fulfilled on subsequent turn."""
    session_id = "test_sess_001"

    # Turn 1: Bot asks for missing order_id
    record_pending_dialogue(session_id, intent="order_status", missing_slots=["order_id"], language="en")
    store = get_session_store()
    session = store.get_or_create(session_id)
    assert session.has_pending_intent()
    assert session.pending_intent == "order_status"
    assert "order_id" in session.pending_slots

    # Turn 2: User provides '#9921'
    resolved = check_pending_dialogue_resolution(session_id, current_query="#9921", new_slots={"order_id": "9921"})
    assert resolved is not None
    intent, slots, lang = resolved
    assert intent == "order_status"
    assert slots.get("order_id") == "9921"
    assert lang == "en"

    # After fulfillment, pending state is cleared
    assert not session.has_pending_intent()


def test_dialogue_state_topic_shift_clears_pending():
    """If user changes topic or cancels, pending slot state must be dropped."""
    session_id = "test_sess_002"
    record_pending_dialogue(session_id, intent="order_status", missing_slots=["order_id"], language="en")

    # User says 'cancel' instead of providing order ID
    resolved = check_pending_dialogue_resolution(session_id, current_query="cancel", new_slots={})
    assert resolved is None

    store = get_session_store()
    session = store.get_or_create(session_id)
    assert not session.has_pending_intent()


def test_proxy_multi_turn_slot_fulfillment_e2e():
    """End-to-end test of multi-turn slot fulfillment through /v1/chat/completions."""
    client = TestClient(app)
    session_id = "e2e_sess_order_track"

    # Turn 1: User asks where their package is (missing order_id)
    res1 = client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Can you please track my delivery?"}],
        },
    )
    assert res1.status_code == 200
    data1 = res1.json()
    reply1 = data1["choices"][0]["message"]["content"]
    assert "order" in reply1.lower() or "tracking" in reply1.lower()

    # Turn 2: User replies with just the order ID: '#9921'
    res2 = client.post(
        "/v1/chat/completions",
        headers={"X-Session-ID": session_id},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": "Can you please track my delivery?"},
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": "#9921"},
            ],
        },
    )
    assert res2.status_code == 200
    data2 = res2.json()
    meta2 = data2.get("pelid_metadata", {})
    assert meta2.get("path") == "A"
    assert meta2.get("saved_cost") is True
    reply2 = data2["choices"][0]["message"]["content"]
    # Should resolve to mock CRM tracking for 9921 (Wireless Noise-Canceling Headphones)
    assert "9921" in reply2 or "headphones" in reply2.lower() or "transit" in reply2.lower()
