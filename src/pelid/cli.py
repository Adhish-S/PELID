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
    summary = get_analytics_summary()
    c_stats = get_cache().stats()

    print("=" * 60)
    print("  Pelid Gateway Live Telemetry Summary")
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
    stats_parser.set_defaults(func=cmd_stats)

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
