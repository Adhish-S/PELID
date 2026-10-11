"""
Pelid Declarative Action Engine

Executes live business actions (REST APIs, Webhooks, CRM Lookups) declared in `pelid.yaml`
for Path A resolutions, completely eliminating hardcoded Python templates.

Features:
1. Dynamic URL & Payload interpolation using extracted slots (e.g. {order_id}).
2. Strict latency SLAs (e.g. 150-300ms HTTP timeout) to preserve sub-100ms total turn speed.
3. Resilient fallback to mock CRM or graceful failure if downstream services are offline.
4. Tri-lingual response template rendering with live API payload fields.
"""

import os
import re
import time
import urllib.parse
import ipaddress
import socket
from typing import Any, Dict, Optional
import httpx

from pelid.domain import load_domain_config, get_active_domain
from pelid.mock_crm import lookup_order as mock_lookup_order


def is_safe_webhook_url(url: str) -> tuple[bool, str]:
    """
    Enterprise SSRF Guardrail.
    Blocks requests to loopback, link-local, cloud metadata, private IP spaces,
    and DNS rebinding targets unless explicitly enabled via PELID_ALLOW_LOCAL_WEBHOOKS=true.
    """
    allow_local = os.getenv("PELID_ALLOW_LOCAL_WEBHOOKS", "false").lower() in ("true", "1", "yes")
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme '{parsed.scheme}'. Only http/https allowed."

        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: missing hostname"

        # Cloud metadata protection
        if hostname.lower() in ("169.254.169.254", "metadata.google.internal", "instance-data"):
            return False, "Blocked: Cloud metadata IP detected (SSRF prevention)"

        if allow_local:
            return True, "Allowed (local dev mode enabled)"

        # Check IP address resolution if hostname is an IP
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False, f"Blocked: Target IP '{hostname}' is non-routable/private (SSRF prevention)"
            return True, "Safe URL"
        except ValueError:
            pass

        # Hostname is a domain name — check local blocklist
        if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return False, "Blocked: Localhost loopback detected (SSRF prevention)"

        # DNS resolution check to prevent DNS rebinding attacks (e.g. 127.0.0.1.nip.io)
        try:
            resolved_addrs = socket.getaddrinfo(hostname, None)
            for _, _, _, _, sockaddr in resolved_addrs:
                ip_str = sockaddr[0]
                ip = ipaddress.ip_address(ip_str)
                if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                    return False, f"Blocked: Hostname '{hostname}' resolves to private/internal IP {ip_str} (DNS rebinding prevention)"
        except socket.gaierror:
            return False, f"Blocked: Cannot resolve hostname '{hostname}'"

        return True, "Safe URL"
    except Exception as e:
        return False, f"URL parse failure: {e}"


def flatten_nested_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested dictionary to dot notation (e.g. data.order.status)."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            # Also maintain leaf keys for convenience if not already present
            if parent_key and k not in [item[0] for item in items]:
                items.append((k, v))
    return dict(items)


class DeclarativeActionEngine:
    """
    Executes declarative actions defined in pelid.yaml for resolved intents.
    """

    def __init__(self):
        self._config = None

    def get_intent_config(self, intent: str, domain_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve declarative configuration for a specific intent from pelid.yaml."""
        cfg = load_domain_config()
        domain = domain_name or get_active_domain()
        domains = cfg.get("domains", {})
        domain_data = domains.get(domain, {})
        intents_data = domain_data.get("intents", {})

        # Check in safe or custom intent declarations
        safe_intents = intents_data.get("safe", {})
        if isinstance(safe_intents, dict) and intent in safe_intents:
            return safe_intents[intent]
        elif isinstance(safe_intents, list):
            # Legacy list format: check if defined as dict in list
            for item in safe_intents:
                if isinstance(item, dict) and intent in item:
                    return item[intent]

        # Check root actions block if present
        actions_block = domain_data.get("actions", {})
        if intent in actions_block:
            return actions_block[intent]

        return None

    async def execute_action(
        self,
        intent: str,
        slots: Dict[str, Any],
        language: str = "en",
        domain_name: Optional[str] = None,
        session_id: str = "",
    ) -> Optional[str]:
        """
        Execute the action for the given intent and return the formatted response message.

        Args:
            intent: Classified intent (e.g. 'order_status')
            slots: Extracted parameter dictionary (e.g. {'order_id': '9921'})
            language: 'en', 'hi', or 'ml'
            domain_name: Optional active domain override
            session_id: Optional conversation session ID for multi-turn state tracking

        Returns:
            Rendered string response, or None if no action declared for intent.
        """
        intent_cfg = self.get_intent_config(intent, domain_name)
        if not intent_cfg or not isinstance(intent_cfg, dict):
            return None

        required_slots = intent_cfg.get("slots", intent_cfg.get("required_slots", []))
        templates = intent_cfg.get("response", intent_cfg.get("templates", {}))

        # 1. Check if required slots are present
        missing_slots = [s for s in required_slots if s not in slots or not slots[s]]
        if missing_slots:
            # Register in session state machine for next turn resolution
            if session_id:
                try:
                    from pelid.session import record_pending_dialogue
                    record_pending_dialogue(session_id, intent, missing_slots, language)
                except Exception:
                    pass

            # Prompt the user for the missing slot in their exact language
            prompt_templates = intent_cfg.get("prompt_for_slots", {})
            if language in prompt_templates:
                return prompt_templates[language]
            return prompt_templates.get("en", f"Could you please provide your {missing_slots[0].replace('_', ' ')}?")

        # 2. Execute Action (HTTP REST API or Mock Fallback)
        action_spec = intent_cfg.get("action", {})
        action_type = action_spec.get("type", "mock").lower()
        context_data: Dict[str, Any] = dict(slots)

        if action_type == "http":
            live_data = await self._execute_http_action(action_spec, slots)
            if live_data:
                context_data.update(flatten_nested_dict(live_data))
            elif action_spec.get("fallback_to_mock", True):
                # Fallback to local mock if remote API fails or times out
                mock_data = self._execute_mock_action(intent, slots)
                if mock_data:
                    context_data.update(flatten_nested_dict(mock_data))
                else:
                    return self._render_not_found(intent_cfg, slots, language)
            else:
                return self._render_not_found(intent_cfg, slots, language)

        elif action_type == "mock":
            mock_data = self._execute_mock_action(intent, slots)
            if mock_data:
                context_data.update(flatten_nested_dict(mock_data))
            else:
                return self._render_not_found(intent_cfg, slots, language)

        # 3. Render Final Response Template
        return self._render_template(templates, context_data, language)

    async def _execute_http_action(self, action_spec: Dict[str, Any], slots: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute external REST API request with SSRF guardrails, slot interpolation, and strict timeout."""
        raw_url = action_spec.get("url", "")
        method = action_spec.get("method", "GET").upper()
        timeout_ms = action_spec.get("timeout_ms", 250)
        timeout_sec = max(0.05, timeout_ms / 1000.0)

        # Interpolate slots into URL: e.g. https://api.store.com/orders/{order_id}
        url = raw_url
        for k, v in slots.items():
            url = url.replace(f"{{{k}}}", urllib.parse.quote(str(v)))

        # SSRF Security Guardrail Check
        is_safe, ssrf_reason = is_safe_webhook_url(url)
        if not is_safe:
            print(f"  [SECURITY SHIELD] Blocked unsafe action URL: {url} ({ssrf_reason})")
            return None

        # Interpolate headers
        headers = {}
        for hk, hv in action_spec.get("headers", {}).items():
            h_str = str(hv)
            # Environment variable substitution: ${ENV_VAR}
            env_matches = re.findall(r"\${([A-Z0-9_]+)}", h_str)
            for env_var in env_matches:
                h_str = h_str.replace(f"${{{env_var}}}", os.getenv(env_var, ""))
            for k, v in slots.items():
                h_str = h_str.replace(f"{{{k}}}", str(v))
            headers[hk] = h_str

        try:
            async with httpx.AsyncClient(timeout=timeout_sec, follow_redirects=False) as client:
                if method == "GET":
                    resp = await client.get(url, headers=headers)
                elif method == "POST":
                    json_body = action_spec.get("body", {})
                    resp = await client.post(url, headers=headers, json=json_body)
                else:
                    resp = await client.request(method, url, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    return data if isinstance(data, dict) else {"result": data}
        except Exception:
            # Network failure or timeout -> returns None for graceful fallback
            pass

        return None

    def _execute_mock_action(self, intent: str, slots: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Mock fallback resolver for standard e-commerce and logistics testing."""
        if "order_id" in slots:
            order = mock_lookup_order(str(slots["order_id"]))
            return order
        return None

    def _render_template(self, templates: Any, context: Dict[str, Any], language: str) -> str:
        """Render response template with context fields."""
        template_str = ""
        if isinstance(templates, dict):
            template_str = templates.get(language, templates.get("en", ""))
        elif isinstance(templates, str):
            template_str = templates

        if not template_str:
            return f"Processed request for intent with data: {context}"

        # Interpolate context fields into template
        rendered = template_str
        for k, v in context.items():
            rendered = rendered.replace(f"{{{k}}}", str(v))

        return rendered

    def _render_not_found(self, intent_cfg: Dict[str, Any], slots: Dict[str, Any], language: str) -> str:
        """Render not-found error message in user's dialect."""
        not_found_templates = intent_cfg.get("not_found", {
            "en": "I couldn't find any record matching the details provided. Please verify and try again.",
            "hi": "Humein aapke diye gaye details se koi record nahi mila. Kripya check karke dobara batayein.",
            "ml": "Ningal nalkiya vivarangal vechu records onnum kandethan kazhinjilla. Dayavayi onnu koodi verify cheyyamo?",
        })
        msg = not_found_templates.get(language, not_found_templates.get("en", "Record not found."))
        for k, v in slots.items():
            msg = msg.replace(f"{{{k}}}", str(v))
        return msg


# Global singleton instance
_action_engine = DeclarativeActionEngine()


def get_action_engine() -> DeclarativeActionEngine:
    """Return singleton action engine instance."""
    return _action_engine
