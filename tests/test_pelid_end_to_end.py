"""
Pelid — Automated Test Suite

Covers:
1. Language detection (English vs Romanized Indic/Manglish/Hinglish)
2. Mock CRM extraction and order status lookup
3. Context budget limits
4. Destructive action override rules
5. End-to-end decision engine & ergonomic response generation
"""

import pytest
import asyncio
from pelid.lang_router import detect_language
from pelid.mock_crm import extract_order_id, lookup_order
from pelid.context_budget import count_tokens, is_within_budget
from pelid.decision import run_decision, DecisionResult
from pelid.responder import generate_crm_response, resolve_path_a_response


def test_language_detection():
    assert detect_language("Where is my order #9921?") == "en"
    assert detect_language("hi there, how are you?") == "en"
    assert detect_language("ennte package pottiya vannath.") == "ml"
    assert detect_language("ennte order evide?") == "ml"
    assert detect_language("eda ennte package eedeya ulle?") == "ml"
    assert detect_language("aree mera oreder kahaa hai?") == "hi"
    assert detect_language("aree mera package tuuta hua hai") == "hi"
    assert detect_language("mera glass damaged aaya, enghana refund kittum?") == "ml"


def test_mock_crm():
    assert extract_order_id("Where is my order #9921?") == "9921"
    assert extract_order_id("Can you check #89214 tracking?") == "89214"
    assert extract_order_id("wherre is my order") is None

    order = lookup_order("9921")
    assert order is not None
    assert "Sony" in order["item"]
    assert order["status"] == "Out for Delivery"

    # Response generator with CRM (English)
    reply_en = generate_crm_response("Where is my order #9921?", "order_status", "en")
    assert "Sony WH-1000XM5" in reply_en
    assert "Out for Delivery" in reply_en

    # Response generator with CRM (Manglish)
    reply_ml = generate_crm_response("Ente order #9921 evide?", "order_status", "ml")
    assert "Ningalude Order #9921" in reply_ml

    # Response generator with CRM (Hinglish)
    reply_hi = generate_crm_response("Mera order #9921 kahan hai?", "order_status", "hi")
    assert "Aapka Order #9921" in reply_hi


def test_context_budget():
    short_text = "Where is my package?"
    assert is_within_budget(short_text) is True
    assert count_tokens(short_text) < 10

    long_text = "word " * 500
    assert is_within_budget(long_text) is False


def test_destructive_guardrails():
    # Destructive intent should NEVER use Path A, even with 100% confidence
    res = DecisionResult(label="refund_request", confidence=1.0)
    assert res.is_destructive is True
    assert res.should_use_path_a is False

    res_cancel = DecisionResult(label="cancel_order", confidence=0.99)
    assert res_cancel.is_destructive is True
    assert res_cancel.should_use_path_a is False

    # Safe intent with high confidence should use Path A
    res_safe = DecisionResult(label="order_status", confidence=0.95)
    assert res_safe.is_destructive is False
    assert res_safe.should_use_path_a is True


@pytest.mark.asyncio
async def test_end_to_end_decision_routing():
    # 1. English query with CRM lookup
    res_en = await run_decision("Where is my order #9921?", language="en")
    assert res_en.label == "order_status"
    assert res_en.should_use_path_a is True
    reply = await resolve_path_a_response("Where is my order #9921?", res_en.label, "en")
    assert "Sony" in reply

    # 2. Manglish tracking query
    res_ml = await run_decision("ennte order evide?", language="ml")
    assert res_ml.label == "order_status"
    assert res_ml.should_use_path_a is True
    reply_ml = await resolve_path_a_response("ennte order evide?", res_ml.label, "ml")
    assert "Ningalude" in reply_ml

    # 3. Hinglish broken package
    res_hi = await run_decision("aree mera package tuuta hua hai", language="hi")
    assert res_hi.label == "wrong_or_defective_item"
    assert res_hi.should_use_path_a is True
    reply_hi = await resolve_path_a_response("aree mera package tuuta hua hai", res_hi.label, "hi")
    assert "toota" in reply_hi

    # 4. Destructive refund query (must NOT use Path A)
    res_ref = await run_decision("mera glass damaged aaya, enghana refund kittum?", language="ml")
    assert res_ref.label == "refund_request"
    assert res_ref.should_use_path_a is False
