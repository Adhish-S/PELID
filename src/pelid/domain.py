"""
Pelid Multi-Domain Manager (Stage 12)

Loads declarative domain profiles from `pelid.yaml` (e.g. ecommerce, food_delivery, b2b_manufacturing)
so developers can adapt the gateway to completely different industries without touching code.
"""

from pathlib import Path
from typing import Any, Optional
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = PROJECT_ROOT / "pelid.yaml"

# Fallback default configuration if yaml parser is absent
DEFAULT_CONFIG = {
    "active_domain": "ecommerce",
    "domains": {
        "ecommerce": {
            "name": "E-Commerce & Retail Logistics",
            "intents": {
                "safe": ["order_status", "order_delayed_or_missing", "wrong_or_defective_item", "invoice_or_receipt", "general_chitchat_or_faq"],
                "destructive": ["cancel_order", "refund_request", "delete_account", "shipping_address_change"],
            },
        }
    },
}


def load_domain_config() -> dict[str, Any]:
    """Parse pelid.yaml configuration manifest."""
    if not YAML_PATH.exists():
        return DEFAULT_CONFIG

    try:
        import yaml
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or DEFAULT_CONFIG
    except Exception:
        # Fallback simple parser if pyyaml is not loaded
        return DEFAULT_CONFIG


def get_active_domain() -> str:
    """Get the currently active domain profile from env or config."""
    return os.getenv("PELID_DOMAIN", "ecommerce").lower()


def get_domain_destructive_intents(domain_name: Optional[str] = None) -> list[str]:
    """Retrieve all intents marked as destructive for the active domain."""
    cfg = load_domain_config()
    d_name = domain_name or get_active_domain()
    domain_data = cfg.get("domains", {}).get(d_name, {})
    return domain_data.get("intents", {}).get("destructive", [])
