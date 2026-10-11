"""
Tests for Pelid Enterprise Admin Console & Management Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from pelid.proxy import app


@pytest.fixture
def client():
    return TestClient(app)


def test_admin_html_served(client: TestClient):
    """Verify that GET /admin serves the admin console HTML page."""
    res = client.get("/admin")
    assert res.status_code == 200
    assert "Pelid Enterprise Admin Console" in res.text
    assert "text/html" in res.headers["content-type"]


def test_admin_overview_api(client: TestClient):
    """Verify GET /api/admin/overview returns structured JSON with KPIs, logs, tenants, and FAQs."""
    res = client.get("/api/admin/overview")
    assert res.status_code == 200
    data = res.json()
    assert "kpis" in data
    assert "total_requests" in data["kpis"]
    assert "recent_logs" in data
    assert "tenants" in data
    assert "faqs" in data
    assert len(data["faqs"]) > 0


def test_admin_key_lifecycle(client: TestClient):
    """Test creating and revoking tenant keys via admin API."""
    tenant_id = "test_fintech_bot"
    name = "Fintech Support Bot"

    # 1. Create key
    create_res = client.post(
        "/api/admin/keys",
        json={"id": tenant_id, "name": name, "rate_limit_rpm": 100},
    )
    assert create_res.status_code == 200
    data = create_res.json()
    assert data["tenant_id"] == tenant_id
    assert data["raw_key"].startswith("pelid_live_")

    # 2. Revoke key
    revoke_res = client.delete(f"/api/admin/keys/{tenant_id}")
    assert revoke_res.status_code == 200
    assert revoke_res.json()["success"] is True


def test_admin_faq_creation(client: TestClient):
    """Test adding custom FAQ through admin API."""
    res = client.post(
        "/api/admin/faqs",
        json={
            "question": "What is the maximum loan amount?",
            "answer": "You can borrow up to $50,000 subject to credit score evaluation.",
            "category": "fintech",
            "variations": ["How much loan can I get?"],
        },
    )
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_admin_query_inspector(client: TestClient):
    """Test query inspection sandbox endpoint."""
    res = client.post(
        "/api/admin/test",
        json={"query": "Can you check tracking for parcel #9921?"},
    )
    assert res.status_code == 200
    data = res.json()
    assert "path" in data
    assert "intent" in data
    assert "latency_ms" in data


def test_admin_auth_security(monkeypatch, client: TestClient):
    """Verify that setting PELID_ADMIN_KEY enforces strict auth on admin endpoints."""
    monkeypatch.setenv("PELID_ADMIN_KEY", "super_secret_admin_master_key_99")

    # 1. Without header -> 403 Forbidden
    res_no_key = client.get("/api/admin/overview")
    assert res_no_key.status_code == 403

    # 2. With wrong header -> 403 Forbidden
    res_bad_key = client.get("/api/admin/overview", headers={"X-Pelid-Admin-Key": "wrong_key"})
    assert res_bad_key.status_code == 403

    # 3. With valid header -> 200 OK
    res_valid_key = client.get("/api/admin/overview", headers={"X-Pelid-Admin-Key": "super_secret_admin_master_key_99"})
    assert res_valid_key.status_code == 200
    assert "kpis" in res_valid_key.json()

