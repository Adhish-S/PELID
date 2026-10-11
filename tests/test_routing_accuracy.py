"""
Pelid Decision Engine & Hybrid Router Accuracy Verification

Tests high-difficulty real-world test cases including:
1. Compound delivery + defect sentences in Manglish, Hinglish, and English
2. Greetings and conversational chit-chat (<1ms, $0.00, Path A)
3. Destructive safety guardrails overriding delivery keywords
4. Dedicated /api/simulate diagnostic endpoint integration
"""

import pytest
from fastapi.testclient import TestClient

from pelid.decision import run_decision
from pelid.lang_router import detect_language
from pelid.proxy import app


@pytest.mark.asyncio
async def test_compound_defect_sentence_manglish():
    # User failure query: received package this morning, opened it, item was cracked
    query = "enikk inn morning aan package vannath. appo njan ath thorann nookiyappo sathanam cracked aan"
    lang = detect_language(query)
    decision = await run_decision(query, language=lang)
    assert decision.label == "wrong_or_defective_item", f"Expected wrong_or_defective_item, got {decision.label}"
    assert decision.confidence >= 0.95
    assert decision.should_use_path_a is True


@pytest.mark.asyncio
async def test_compound_defect_sentence_english():
    query = "package arrived this morning and when i opened it the item was cracked"
    lang = detect_language(query)
    decision = await run_decision(query, language=lang)
    assert decision.label == "wrong_or_defective_item"
    assert decision.should_use_path_a is True


@pytest.mark.asyncio
async def test_compound_defect_sentence_hinglish():
    query = "aaj subah packet mila tha lekin kholne par product poori tarah se toota nikla"
    lang = detect_language(query)
    decision = await run_decision(query, language=lang)
    assert decision.label == "wrong_or_defective_item"
    assert decision.should_use_path_a is True


@pytest.mark.asyncio
async def test_greetings_fast_path():
    greeting_queries = ["HI...", "Hi", "Hello", "good morning", "namaskaram", "hey team"]
    for q in greeting_queries:
        lang = detect_language(q)
        decision = await run_decision(q, language=lang)
        assert decision.label == "general_chitchat_or_faq", f"Failed on {q}: got {decision.label}"
        assert decision.confidence >= 0.99
        assert decision.should_use_path_a is True


@pytest.mark.asyncio
async def test_destructive_guardrails_override():
    cancel_query = "Cancel order #9921 immediately"
    lang = detect_language(cancel_query)
    decision = await run_decision(cancel_query, language=lang)
    assert decision.label == "cancel_order"
    assert decision.is_destructive is True
    assert decision.should_use_path_a is False  # Must escalate to Path B


def test_simulate_endpoint_accuracy():
    client = TestClient(app)

    # 1. Defect Manglish
    res1 = client.post("/api/simulate", json={"query": "enikk inn morning aan package vannath. appo njan ath thorann nookiyappo sathanam cracked aan"})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["intent"] == "wrong_or_defective_item"
    assert data1["path"] == "A"
    assert data1["cost_saved"] == "$0.0000 (Saved)"

    # 2. Greeting
    res2 = client.post("/api/simulate", json={"query": "HI..."})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["intent"] == "general_chitchat_or_faq"
    assert data2["path"] == "A"
    assert data2["cost_saved"] == "$0.0000 (Saved)"

    # 3. Order status
    res3 = client.post("/api/simulate", json={"query": "Where is my order #9921?"})
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["intent"] == "order_status"
    assert data3["path"] == "A"
    assert data3["slots"].get("order_id") == "9921"
