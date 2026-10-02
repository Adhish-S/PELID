"""
Tests for Pelid Multi-Tenant API Key & Access Control Engine
"""

import pytest
import time
from fastapi.testclient import TestClient

from pelid.auth import (
    create_tenant,
    revoke_tenant,
    list_tenants,
    authenticate_request,
    get_tenant_limiter,
    reload_key_cache,
    has_active_tenants,
)
from pelid.proxy import app
from pelid.db import get_analytics_summary


@pytest.fixture(autouse=True)
def setup_teardown_auth():
    """Ensure key cache is reloaded and clean state for tests."""
    from pelid.db import get_connection
    reload_key_cache()
    yield
    with get_connection() as conn:
        conn.execute("DELETE FROM tenants WHERE id LIKE 'test_%' OR id LIKE 'revoke_%' OR id LIKE 'tenant_%' OR id LIKE 'proxy_%' OR id LIKE 'demo_%'")
        conn.commit()
    reload_key_cache()


def test_key_generation_and_verification():
    # 1. Register a new tenant
    tenant_id = f"test_corp_{int(time.time() * 1000)}"
    raw_key, tenant = create_tenant(tenant_id, "Test Corporation", rate_limit_rpm=120)

    assert raw_key.startswith("pelid_live_")
    assert len(raw_key) > 30
    assert tenant.id == tenant_id
    assert tenant.name == "Test Corporation"
    assert tenant.is_active is True
    assert tenant.rate_limit_rpm == 120

    # 2. Authenticate using X-Pelid-Key header
    is_auth, auth_tenant, err = authenticate_request(pelid_key_header=raw_key)
    assert is_auth is True
    assert auth_tenant is not None
    assert auth_tenant.id == tenant_id
    assert err == ""

    # 3. Authenticate using Bearer Authorization header
    is_auth_bearer, bearer_tenant, _ = authenticate_request(auth_header=f"Bearer {raw_key}")
    assert is_auth_bearer is True
    assert bearer_tenant.id == tenant_id

    # 4. Invalid key should be rejected
    is_invalid, _, err_inv = authenticate_request(pelid_key_header="pelid_live_invalidkey1234567890")
    assert is_invalid is False
    assert "Invalid or revoked" in err_inv

    # 5. Missing key should be rejected
    is_missing, _, err_mis = authenticate_request()
    assert is_missing is False
    assert "Missing authentication" in err_mis


def test_tenant_revocation():
    tenant_id = f"revoke_corp_{int(time.time() * 1000)}"
    raw_key, _ = create_tenant(tenant_id, "Revoke Corp", rate_limit_rpm=60)

    # Verify key works initially
    is_auth, _, _ = authenticate_request(pelid_key_header=raw_key)
    assert is_auth is True

    # Revoke key
    revoked = revoke_tenant(tenant_id)
    assert revoked is True

    # Key must be immediately rejected
    is_auth_after, _, err = authenticate_request(pelid_key_header=raw_key)
    assert is_auth_after is False
    assert "Invalid or revoked" in err


def test_per_tenant_rate_limiting():
    tenant_id_a = f"tenant_a_{int(time.time() * 1000)}"
    tenant_id_b = f"tenant_b_{int(time.time() * 1000)}"

    _, tenant_a = create_tenant(tenant_id_a, "Tenant A", rate_limit_rpm=2)
    _, tenant_b = create_tenant(tenant_id_b, "Tenant B", rate_limit_rpm=50)

    limiter_a = get_tenant_limiter(tenant_a)
    limiter_b = get_tenant_limiter(tenant_b)

    # Tenant A allows 2 requests
    allowed1, _ = limiter_a.is_allowed(tenant_a.id)
    allowed2, _ = limiter_a.is_allowed(tenant_a.id)
    assert allowed1 is True
    assert allowed2 is True

    # 3rd request for Tenant A should be throttled
    allowed3, retry_after = limiter_a.is_allowed(tenant_a.id)
    assert allowed3 is False
    assert retry_after > 0

    # Tenant B is completely unaffected
    allowed_b, _ = limiter_b.is_allowed(tenant_b.id)
    assert allowed_b is True


def test_proxy_auth_integration():
    tenant_id = f"proxy_test_{int(time.time() * 1000)}"
    raw_key, _ = create_tenant(tenant_id, "Proxy Integration Corp", rate_limit_rpm=100)

    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Where is my order #9921?"}],
    }

    # 1. Request with NO auth should return 401 Unauthorized
    resp_no_auth = client.post("/v1/chat/completions", json=payload)
    assert resp_no_auth.status_code == 401
    assert "auth_error" in resp_no_auth.json()["error"]["type"]

    # 2. Request with INVALID key should return 401 Unauthorized
    resp_bad_key = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Pelid-Key": "pelid_live_fake_key_0000000000000000"},
    )
    assert resp_bad_key.status_code == 401

    # 3. Request with VALID key should return 200 OK (and resolve locally on Path A)
    resp_valid = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Pelid-Key": raw_key},
    )
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert "choices" in data
    assert data["pelid_metadata"]["path"] == "A"
    assert "Sony" in data["choices"][0]["message"]["content"]
