"""
Pelid CLI — Command Line Interface for Enterprise AI Decision Gateway
"""

import argparse
import sys
import uvicorn
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def cmd_start(args):
    """Start the Pelid Gateway server."""
    print("=" * 60)
    print("  Starting Pelid AI Decision Gateway")
    print(f"  Serving on http://{args.host}:{args.port}")
    print("=" * 60)
    uvicorn.run(
        "pelid.proxy:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_eval(args):
    """Run empirical benchmark evaluation."""
    import asyncio
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / "scripts"))
    from run_eval import run_evaluation
    asyncio.run(run_evaluation())


def cmd_stats(args):
    """Print current telemetry and cache savings statistics."""
    from pelid.db import get_analytics_summary
    from pelid.cache import get_cache
    tenant = getattr(args, "tenant", None)
    summary = get_analytics_summary(tenant_id=tenant)
    c_stats = get_cache().stats()

    print("=" * 60)
    tenant_label = f" (Tenant: {tenant})" if tenant else " (Global All Tenants)"
    print(f"  Pelid Gateway Live Telemetry Summary{tenant_label}")
    print("=" * 60)
    print(f"  Total Requests:         {summary.get('total_requests', 0)}")
    print(f"  Resolved Locally (A):   {summary.get('path_a_count', 0)} ({summary.get('local_resolution_rate', 0)}%)")
    print(f"  Escalated Upstream (B): {summary.get('path_b_count', 0)}")
    print(f"  Total Tokens Saved:     {summary.get('total_tokens_saved', 0)}")
    print(f"  Estimated Cost Saved:   ${summary.get('total_cost_saved_usd', 0):.4f}")
    print(f"  Avg Path A Latency:     {summary.get('avg_latency_path_a', 0):.1f} ms")
    print(f"  Avg Path B Latency:     {summary.get('avg_latency_path_b', 0):.1f} ms")
    print("-" * 60)
    print(f"  Cache Hits (Total):     {c_stats.get('total_hits', 0)} ({c_stats.get('hit_rate_pct', 0)}%)")
    print(f"  Exact Hash Hits (<0.05ms): {c_stats.get('exact_hits', 0)}")
    print(f"  Semantic Cosine Hits:   {c_stats.get('semantic_hits', 0)}")
    print("=" * 60)


def cmd_key_create(args):
    """Generate and register a new scoped tenant API key."""
    from pelid.auth import create_tenant
    raw_key, tenant = create_tenant(
        tenant_id=args.tenant,
        name=args.name or args.tenant,
        rate_limit_rpm=args.rate_limit,
    )
    print("=" * 60)
    print("  [SUCCESS] Created New Scoped Tenant API Key")
    print("=" * 60)
    print(f"  Tenant ID:   {tenant.id}")
    print(f"  Tenant Name: {tenant.name}")
    print(f"  Rate Limit:  {tenant.rate_limit_rpm} req/min")
    print(f"  Prefix:      {tenant.key_prefix}")
    print("-" * 60)
    print(f"  API KEY:     {raw_key}")
    print("-" * 60)
    print("  [IMPORTANT] Store this key securely. Raw key is never stored in DB.")
    print("  Pass via header: 'X-Pelid-Key: <key>' or 'Authorization: Bearer <key>'")
    print("=" * 60)


def cmd_key_list(args):
    """List all registered tenants and their API key status."""
    from pelid.auth import list_tenants
    tenants = list_tenants()
    print("=" * 75)
    print(f"  {'TENANT ID':<18} {'NAME':<20} {'PREFIX':<18} {'RPM':<6} {'STATUS'}")
    print("=" * 75)
    if not tenants:
        print("  (No tenants registered yet. Gateway is running in open development mode.)")
    for t in tenants:
        status = "ACTIVE" if t["is_active"] else "REVOKED"
        print(f"  {t['id']:<18} {t['name'][:18]:<20} {t['key_prefix']:<18} {t['rate_limit_rpm']:<6} {status}")
    print("=" * 75)


def cmd_key_revoke(args):
    """Revoke a tenant's access key."""
    from pelid.auth import revoke_tenant
    success = revoke_tenant(args.tenant)
    if success:
        print(f"[OK] Successfully revoked API access for tenant '{args.tenant}'.")
    else:
        print(f"[ERROR] Tenant '{args.tenant}' not found in registry.")


def cmd_cache_clear(args):
    """Clear memory cache and telemetry logs."""
    from pelid.cache import get_cache
    from pelid.db import clear_logs
    get_cache().clear()
    deleted = clear_logs()
    print(f"[OK] Cleared semantic cache and pruned {deleted} rows from telemetry database.")


def main():
    parser = argparse.ArgumentParser(
        prog="pelid",
        description="Pelid — Enterprise AI Decision Gateway & Cost-Reduction Proxy",
    )
    parser.add_argument("--version", action="version", version="pelid 1.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # pelid start
    start_parser = subparsers.add_parser("start", help="Launch the gateway server")
    start_parser.add_argument("--host", default="127.0.0.1", help="Binding host (default: 127.0.0.1)")
    start_parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    start_parser.set_defaults(func=cmd_start)

    # pelid eval
    eval_parser = subparsers.add_parser("eval", help="Run held-out benchmark evaluation suite")
    eval_parser.set_defaults(func=cmd_eval)

    # pelid stats
    stats_parser = subparsers.add_parser("stats", help="Display real-time analytics and savings")
    stats_parser.add_argument("--tenant", default=None, help="Filter analytics by tenant ID")
    stats_parser.set_defaults(func=cmd_stats)

    # pelid key (subcommands: create, list, revoke)
    key_parser = subparsers.add_parser("key", help="Manage multi-tenant API keys")
    key_subparsers = key_parser.add_subparsers(dest="key_command", help="Key operations")

    # pelid key create
    create_key_parser = key_subparsers.add_parser("create", help="Issue a new scoped tenant API key")
    create_key_parser.add_argument("--tenant", required=True, help="Unique tenant identifier (e.g. acme_corp)")
    create_key_parser.add_argument("--name", default=None, help="Human-readable company or project name")
    create_key_parser.add_argument("--rate-limit", type=int, default=60, help="Max requests per minute (default: 60)")
    create_key_parser.set_defaults(func=cmd_key_create)

    # pelid key list
    list_key_parser = key_subparsers.add_parser("list", help="List all registered tenant keys")
    list_key_parser.set_defaults(func=cmd_key_list)

    # pelid key revoke
    revoke_key_parser = key_subparsers.add_parser("revoke", help="Revoke a tenant API key")
    revoke_key_parser.add_argument("tenant", help="Tenant ID to revoke")
    revoke_key_parser.set_defaults(func=cmd_key_revoke)

    # pelid clear
    clear_parser = subparsers.add_parser("clear", help="Reset local cache and purge audit database")
    clear_parser.set_defaults(func=cmd_cache_clear)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
