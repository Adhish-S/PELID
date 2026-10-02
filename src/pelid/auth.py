"""
Pelid Multi-Tenant API Key & Access Control Engine

Provides:
- Cryptographically secure API key generation (`pelid_live_<token>`)
- SHA-256 one-way hashed storage (raw keys are never stored)
- Sub-microsecond in-memory key cache for zero proxy latency overhead
- Per-tenant sliding-window rate limiting
- Fast constant-time key verification & instant revocation
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pelid.config import PELID_PROXY_API_KEY
from pelid.db import get_connection, init_db
from pelid.security import SlidingWindowRateLimiter


@dataclass
class Tenant:
    id: str
    name: str
    api_key_hash: str
    key_prefix: str
    rate_limit_rpm: int
    is_active: bool
    created_at: str
    revoked_at: Optional[str] = None


# In-memory hot caches to preserve sub-millisecond proxy response times
_KEY_CACHE: dict[str, Tenant] = {}  # sha256_hash -> Tenant
_TENANT_LIMITERS: dict[str, SlidingWindowRateLimiter] = {}  # tenant_id -> SlidingWindowRateLimiter
_TOTAL_TENANTS: int = 0
_CACHE_INITIALIZED: bool = False


def _hash_key(raw_key: str) -> str:
    """Generate SHA-256 digest of raw API key for secure lookups."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


def reload_key_cache():
    """Reload all active tenants from SQLite database into memory cache."""
    global _KEY_CACHE, _TENANT_LIMITERS, _TOTAL_TENANTS, _CACHE_INITIALIZED
    init_db()

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, api_key_hash, key_prefix, rate_limit_rpm, is_active, created_at, revoked_at
            FROM tenants
            """
        )
        rows = cur.fetchall()

    _TOTAL_TENANTS = len(rows)
    new_cache: dict[str, Tenant] = {}
    for r in rows:
        tenant = Tenant(
            id=r["id"],
            name=r["name"],
            api_key_hash=r["api_key_hash"],
            key_prefix=r["key_prefix"],
            rate_limit_rpm=int(r["rate_limit_rpm"]),
            is_active=bool(r["is_active"]),
            created_at=r["created_at"],
            revoked_at=r["revoked_at"],
        )
        if tenant.is_active:
            new_cache[tenant.api_key_hash] = tenant
            if tenant.id not in _TENANT_LIMITERS:
                _TENANT_LIMITERS[tenant.id] = SlidingWindowRateLimiter(max_requests_per_minute=tenant.rate_limit_rpm)

    _KEY_CACHE = new_cache
    _CACHE_INITIALIZED = True


def get_tenant_limiter(tenant: Tenant) -> SlidingWindowRateLimiter:
    """Get or create the sliding-window rate limiter for a specific tenant."""
    if tenant.id not in _TENANT_LIMITERS:
        _TENANT_LIMITERS[tenant.id] = SlidingWindowRateLimiter(max_requests_per_minute=tenant.rate_limit_rpm)
    return _TENANT_LIMITERS[tenant.id]


def create_tenant(tenant_id: str, name: str, rate_limit_rpm: int = 60) -> tuple[str, Tenant]:
    """
    Register a new tenant and generate a unique, cryptographically secure API key.

    Returns:
        (raw_api_key, Tenant)
        NOTE: raw_api_key is only returned once and cannot be recovered if lost.
    """
    init_db()

    # Format: pelid_live_<32 hex chars>
    raw_key = f"pelid_live_{secrets.token_hex(16)}"
    key_hash = _hash_key(raw_key)
    key_prefix = raw_key[:18] + "..."  # e.g. pelid_live_a1b2c3d...
    now_iso = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tenants (id, name, api_key_hash, key_prefix, rate_limit_rpm, is_active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                api_key_hash=excluded.api_key_hash,
                key_prefix=excluded.key_prefix,
                rate_limit_rpm=excluded.rate_limit_rpm,
                is_active=1,
                created_at=excluded.created_at,
                revoked_at=NULL
            """,
            (tenant_id, name, key_hash, key_prefix, rate_limit_rpm, now_iso),
        )
        conn.commit()

    tenant = Tenant(
        id=tenant_id,
        name=name,
        api_key_hash=key_hash,
        key_prefix=key_prefix,
        rate_limit_rpm=rate_limit_rpm,
        is_active=True,
        created_at=now_iso,
    )

    global _TOTAL_TENANTS
    _TOTAL_TENANTS += 1

    # Immediately update memory cache
    _KEY_CACHE[key_hash] = tenant
    _TENANT_LIMITERS[tenant_id] = SlidingWindowRateLimiter(max_requests_per_minute=rate_limit_rpm)

    return raw_key, tenant


def revoke_tenant(tenant_id: str) -> bool:
    """
    Revoke access for a tenant. Their API key will immediately stop working.
    """
    init_db()
    now_iso = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cur = conn.execute(
            "UPDATE tenants SET is_active = 0, revoked_at = ? WHERE id = ?",
            (now_iso, tenant_id),
        )
        conn.commit()
        affected = cur.rowcount

    # Invalidate cache
    reload_key_cache()
    return affected > 0


def list_tenants() -> list[dict]:
    """Retrieve all registered tenants with their metadata and status."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, key_prefix, rate_limit_rpm, is_active, created_at, revoked_at
            FROM tenants
            ORDER BY created_at DESC
            """
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def has_active_tenants() -> bool:
    """Check if any tenants are registered in the system or system proxy key is configured."""
    global _CACHE_INITIALIZED
    if not _CACHE_INITIALIZED:
        reload_key_cache()
    return bool(_KEY_CACHE) or (_TOTAL_TENANTS > 0) or bool(PELID_PROXY_API_KEY)


def authenticate_request(
    auth_header: Optional[str] = None,
    pelid_key_header: Optional[str] = None,
) -> tuple[bool, Optional[Tenant], str]:
    """
    Authenticate an incoming request via either:
    1. `X-Pelid-Key: pelid_live_...`
    2. `Authorization: Bearer pelid_live_...` or configured legacy `PELID_PROXY_API_KEY`

    Returns:
        (is_authenticated: bool, tenant: Optional[Tenant], error_message: str)
    """
    global _CACHE_INITIALIZED
    if not _CACHE_INITIALIZED:
        reload_key_cache()

    # Extract raw token
    raw_token: Optional[str] = None

    if pelid_key_header:
        raw_token = pelid_key_header.strip()
    elif auth_header:
        parts = auth_header.strip().split(" ")
        if len(parts) == 2 and parts[0].lower() == "bearer":
            raw_token = parts[1].strip()

    # If a token was provided, ALWAYS validate it strictly
    if raw_token:
        # Check against master proxy key
        if PELID_PROXY_API_KEY and hmac.compare_digest(raw_token, PELID_PROXY_API_KEY):
            admin_tenant = Tenant(
                id="admin_proxy",
                name="System Master Admin",
                api_key_hash=_hash_key(PELID_PROXY_API_KEY),
                key_prefix="master_key",
                rate_limit_rpm=1000,
                is_active=True,
                created_at=datetime.now().isoformat(),
            )
            return True, admin_tenant, ""

        # Check against Multi-Tenant Registry (SHA-256 in memory cache)
        token_hash = _hash_key(raw_token)
        tenant = _KEY_CACHE.get(token_hash)
        if tenant and tenant.is_active:
            return True, tenant, ""

        # Check SQLite once in case another process/thread created it
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, api_key_hash, key_prefix, rate_limit_rpm, is_active, created_at, revoked_at
                FROM tenants
                WHERE api_key_hash = ? AND is_active = 1
                """,
                (token_hash,),
            )
            row = cur.fetchone()
            if row:
                refreshed_tenant = Tenant(
                    id=row["id"],
                    name=row["name"],
                    api_key_hash=row["api_key_hash"],
                    key_prefix=row["key_prefix"],
                    rate_limit_rpm=int(row["rate_limit_rpm"]),
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    revoked_at=row["revoked_at"],
                )
                _KEY_CACHE[token_hash] = refreshed_tenant
                return True, refreshed_tenant, ""

        return False, None, "Invalid or revoked Pelid API key."

    # If NO token was provided:
    # If tenants are registered or master key is set, authentication is mandatory
    if has_active_tenants():
        return False, None, "Missing authentication. Provide 'X-Pelid-Key' header or 'Authorization: Bearer <key>'."

    # Otherwise, unconfigured open development mode allows traffic
    return True, None, ""
