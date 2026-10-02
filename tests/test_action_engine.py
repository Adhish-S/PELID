"""
Tests for Pelid Declarative Action Engine & Dynamic Slot Extractor
"""

import pytest
from pelid.slot_extractor import extract_slots
from pelid.action_engine import get_action_engine
from pelid.responder import generate_crm_response, resolve_path_a_response


def test_slot_extractor_accuracy():
    # 1. Standard #ID
    slots = extract_slots("Where is my order #9921?")
    assert slots.get("order_id") == "9921"

    # 2. Hyphenated ORD-ID
    slots = extract_slots("Check delivery for ORD-89214 please")
    assert slots.get("order_id") == "89214"

    # 3. Alphanumeric tracking code and email
    slots = extract_slots("Send status of AWB992148 to customer@example.com")
    assert slots.get("email") == "customer@example.com"
    assert "AWB992148" in slots.get("tracking_number", "")

    # 4. Words like 'evide' or 'kahan' should NOT be extracted as IDs
    slots = extract_slots("ennte order evide?")
    assert "order_id" not in slots


def test_declarative_action_engine_sync():
    engine = get_action_engine()
    cfg = engine.get_intent_config("order_status")
    assert cfg is not None
    assert "slots" in cfg

    # 1. Existing mock order in English
    reply_en = generate_crm_response("Where is my order #9921?", "order_status", "en")
    assert "Sony" in reply_en
    assert "Out for Delivery" in reply_en

    # 2. Existing mock order in Manglish
    reply_ml = generate_crm_response("Ente order #9921 evide?", "order_status", "ml")
    assert "Sony" in reply_ml
    assert "Out for Delivery" in reply_ml

    # 3. Missing order ID prompt in Hinglish
    reply_missing = generate_crm_response("Mera order kahan hai?", "order_status", "hi")
    assert "Order ID" in reply_missing


@pytest.mark.asyncio
async def test_declarative_action_engine_async():
    # Test resolve_path_a_response uses action engine seamlessly
    res = await resolve_path_a_response("Where is my order #9921?", "order_status", "en")
    assert "Sony" in res
    assert "BlueDart" in res
