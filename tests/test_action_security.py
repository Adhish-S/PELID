"""
Tests for Pelid Action Engine Enterprise Security & Nested JSONPath Execution
"""

import pytest
from pelid.action_engine import (
    is_safe_webhook_url,
    flatten_nested_dict,
    get_action_engine,
)


def test_ssrf_blocks_cloud_metadata():
    """Verify SSRF defense blocks AWS/GCP/Azure link-local metadata endpoints."""
    bad_urls = [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://169.254.169.254/metadata/v1/maintenance",
    ]
    for url in bad_urls:
        is_safe, reason = is_safe_webhook_url(url)
        assert is_safe is False
        assert "ssrf" in reason.lower() or "metadata" in reason.lower() or "blocked" in reason.lower()


def test_ssrf_blocks_private_and_loopback():
    """Verify SSRF blocks localhost and non-routable IPs in standard mode."""
    bad_urls = [
        "http://127.0.0.1:8000/internal-secrets",
        "http://localhost:6379/keys",
        "http://10.0.0.1/admin",
        "http://192.168.1.1/router-config",
    ]
    for url in bad_urls:
        is_safe, reason = is_safe_webhook_url(url)
        assert is_safe is False
        assert "blocked" in reason.lower() or "loopback" in reason.lower() or "private" in reason.lower()


def test_ssrf_blocks_dns_rebinding():
    """Verify SSRF blocks domain names that resolve to local loopback addresses."""
    rebinding_urls = [
        "http://127.0.0.1.nip.io/admin",
    ]
    for url in rebinding_urls:
        is_safe, reason = is_safe_webhook_url(url)
        assert is_safe is False
        assert "blocked" in reason.lower() or "rebinding" in reason.lower() or "private" in reason.lower()



def test_ssrf_allows_valid_public_apis():
    """Verify that public HTTPS APIs are properly permitted."""
    valid_urls = [
        "https://api.stripe.com/v1/charges",
        "https://api.shopify.com/admin/api/2024-01/orders.json",
        "https://api.zendesk.com/v2/tickets",
    ]
    for url in valid_urls:
        is_safe, reason = is_safe_webhook_url(url)
        assert is_safe is True


def test_nested_jsonpath_flattening():
    """Verify recursive flattening of complex API responses into dot notation."""
    nested_response = {
        "status": "success",
        "order": {
            "id": "ORD-7721",
            "state": "Out for delivery",
            "logistics": {
                "carrier": "BlueDart",
                "tracking_number": "BD-9912",
                "eta": "Today by 6 PM",
            },
        },
    }
    flattened = flatten_nested_dict(nested_response)

    assert flattened["order.id"] == "ORD-7721"
    assert flattened["order.state"] == "Out for delivery"
    assert flattened["order.logistics.carrier"] == "BlueDart"
    assert flattened["order.logistics.eta"] == "Today by 6 PM"
    assert flattened["status"] == "success"


@pytest.mark.asyncio
async def test_action_engine_render_with_nested_data():
    """Test template rendering with nested context data."""
    engine = get_action_engine()
    templates = {
        "en": "Your order {order.id} is currently '{order.state}' via {order.logistics.carrier} (ETA: {order.logistics.eta}).",
    }
    context = flatten_nested_dict({
        "order": {
            "id": "9921",
            "state": "In Transit",
            "logistics": {
                "carrier": "FedEx",
                "eta": "Tomorrow morning",
            },
        }
    })
    rendered = engine._render_template(templates, context, language="en")
    assert "9921" in rendered
    assert "In Transit" in rendered
    assert "FedEx" in rendered
    assert "Tomorrow morning" in rendered
