"""
Tests for Pelid Universal Zero-Code Semantic Knowledge Base (FAQ Engine)
"""

import pytest
from fastapi.testclient import TestClient
from pelid.faq_engine import FAQEngine, get_faq_engine
from pelid.proxy import app


@pytest.fixture(scope="module")
def faq_engine():
    engine = get_faq_engine()
    engine.load_faqs()
    return engine


def test_faq_engine_matching_bangalore(faq_engine: FAQEngine):
    """Verify semantic matching of logistics coverage FAQ with variations."""
    res1 = faq_engine.match("Do you deliver to Bangalore?")
    assert res1 is not None
    assert res1.matched is True
    assert res1.score >= 0.85
    assert "bangalore" in res1.answer.lower()

    # Variation test
    res2 = faq_engine.match("Can you ship to Bengaluru?")
    assert res2 is not None
    assert res2.matched is True
    assert res2.score >= 0.82


def test_faq_engine_matching_payment_methods(faq_engine: FAQEngine):
    """Verify semantic matching of payment options FAQ."""
    res = faq_engine.match("Can I pay using UPI or cash on delivery?")
    assert res is not None
    assert res.matched is True
    assert "upi" in res.answer.lower()


def test_faq_engine_no_false_positive(faq_engine: FAQEngine):
    """Unrelated queries must not falsely trigger FAQ matches."""
    res = faq_engine.match("Write a python script to sort a list")
    assert res is None


def test_dynamic_add_faq(faq_engine: FAQEngine):
    """Test dynamic ingestion of startup-specific FAQ without server reboot."""
    faq_engine.add_faq(
        question="How do I get my certificate of completion?",
        answer="You can download your verified course certificate from the 'My Learning' tab upon passing the final exam.",
        category="edtech",
        variations=["Where is my course certificate?", "Download completion certificate"],
    )

    res = faq_engine.match("Where can I find my course completion certificate?")
    assert res is not None
    assert res.matched is True
    assert "certificate" in res.answer.lower()


def test_proxy_faq_path_a_resolution():
    """Verify that an FAQ query routed to /v1/chat/completions resolves on Path A for $0.00."""
    client = TestClient(app)
    res = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "What payment methods do you accept?"}],
        },
    )
    assert res.status_code == 200
    data = res.json()
    meta = data.get("pelid_metadata", {})
    assert meta.get("path") == "A"
    assert meta.get("saved_cost") is True
    assert meta.get("faq_matched") is True
    reply = data["choices"][0]["message"]["content"]
    assert "upi" in reply.lower() or "cards" in reply.lower()
