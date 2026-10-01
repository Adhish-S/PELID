"""
Pelid Persistent Analytics Database (SQLite)

Logs every incoming API request and maintains metrics:
- Query text, detected language, and predicted intent
- Calibrated confidence and decision path (Path A vs Path B)
- Exact tokens consumed or saved (prompt tokens + completion tokens)
- Latency (ms) and estimated cost saved ($ USD) across multiple LLM baselines
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "pelid.db"

# Official Frontier LLM Pricing Baselines (per 1,000,000 tokens)
PRICING_CATALOG = {
    "gpt-4o": {
        "name": "GPT-4o (OpenAI)",
        "prompt_rate": 2.50 / 1_000_000,       # $2.50 per 1M tokens
        "completion_rate": 10.00 / 1_000_000,  # $10.00 per 1M tokens
    },
    "claude-3-5-sonnet": {
        "name": "Claude 3.5 Sonnet (Anthropic)",
        "prompt_rate": 3.00 / 1_000_000,       # $3.00 per 1M tokens
        "completion_rate": 15.00 / 1_000_000,  # $15.00 per 1M tokens
    },
    "gpt-4o-mini": {
        "name": "GPT-4o-mini (OpenAI)",
        "prompt_rate": 0.15 / 1_000_000,       # $0.15 per 1M tokens
        "completion_rate": 0.60 / 1_000_000,   # $0.60 per 1M tokens
    },
}

DEFAULT_MODEL = "gpt-4o"


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory and WAL mode enabled for high concurrency."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db():
    """Initialize database tables and indexes."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                intent TEXT NOT NULL,
                confidence REAL NOT NULL,
                language TEXT NOT NULL,
                path TEXT NOT NULL,
                tokens_prompt INTEGER NOT NULL DEFAULT 0,
                tokens_completion INTEGER NOT NULL DEFAULT 0,
                tokens_saved INTEGER NOT NULL DEFAULT 0,
                cost_saved_usd REAL NOT NULL DEFAULT 0.0,
                latency_ms REAL NOT NULL,
                reason TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON request_logs(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON request_logs(path)")
        conn.commit()


def log_request(
    query: str,
    intent: str,
    confidence: float,
    language: str,
    path: str,
    latency_ms: float,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    tokens_saved: int = 0,
    reason: str = "",
) -> int:
    """
    Log a processed request into SQLite with exact prompt and completion token cost accounting.
    Returns the inserted row ID.
    """
    init_db()

    # Exact cost accounting based on GPT-4o input & output token pricing
    gpt4o = PRICING_CATALOG["gpt-4o"]
    if path == "A":
        # When resolved locally, we saved both the prompt tokens and the completion tokens
        cost_saved = (tokens_prompt * gpt4o["prompt_rate"]) + (tokens_completion * gpt4o["completion_rate"])
        # If tokens_saved was passed as a combined sum, ensure positive non-zero calculation
        if cost_saved == 0.0 and tokens_saved > 0:
            cost_saved = tokens_saved * ((gpt4o["prompt_rate"] + gpt4o["completion_rate"]) / 2)
    else:
        cost_saved = 0.0

    now_iso = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO request_logs (
                timestamp, query, intent, confidence, language, path,
                tokens_prompt, tokens_completion, tokens_saved, cost_saved_usd,
                latency_ms, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso,
                query[:500],
                intent,
                float(confidence),
                language,
                path,
                tokens_prompt,
                tokens_completion,
                tokens_saved,
                cost_saved,
                float(latency_ms),
                reason,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def clear_logs() -> int:
    """Wipe all logs from the database for a clean demo session."""
    init_db()
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM request_logs")
        deleted = cursor.rowcount
        conn.commit()
        conn.execute("VACUUM")
    return deleted


def get_analytics_summary() -> dict[str, Any]:
    """Calculate aggregated stats and multi-model financial comparisons across all requests."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()

        # Total counts
        cur.execute("SELECT COUNT(*) FROM request_logs")
        total_requests = cur.fetchone()[0] or 0

        if total_requests == 0:
            return {
                "total_requests": 0,
                "path_a_count": 0,
                "path_b_count": 0,
                "local_resolution_rate": 0.0,
                "total_tokens_saved": 0,
                "total_cost_saved_usd": 0.0,
                "claude_sonnet_saved_usd": 0.0,
                "gpt_4o_mini_saved_usd": 0.0,
                "projected_monthly_savings_100k": 0.0,
                "avg_latency_path_a": 0.0,
                "avg_latency_path_b": 0.0,
            }

        cur.execute("SELECT COUNT(*) FROM request_logs WHERE path = 'A'")
        path_a_count = cur.fetchone()[0] or 0

        path_b_count = total_requests - path_a_count
        resolution_rate = (path_a_count / total_requests) * 100.0

        # Tokens and dollars saved
        cur.execute("SELECT SUM(tokens_saved), SUM(cost_saved_usd) FROM request_logs")
        row = cur.fetchone()
        tokens_saved = int(row[0] or 0)
        cost_saved_gpt4o = float(row[1] or 0.0)

        # Multi-model cost comparisons
        # Claude 3.5 Sonnet is ~1.5x - 1.8x GPT-4o price
        cost_saved_claude = cost_saved_gpt4o * 1.55
        # GPT-4o-mini is ~0.06x GPT-4o price
        cost_saved_mini = cost_saved_gpt4o * 0.06

        # Average latencies
        cur.execute("SELECT AVG(latency_ms) FROM request_logs WHERE path = 'A'")
        avg_a = cur.fetchone()[0] or 0.0

        cur.execute("SELECT AVG(latency_ms) FROM request_logs WHERE path = 'B'")
        avg_b = cur.fetchone()[0] or 0.0

        # Projected monthly savings for a standard 100,000 requests/month volume
        # Avg cost saved per local resolution * (100,000 * resolution_rate / 100)
        avg_saving_per_path_a = (cost_saved_gpt4o / path_a_count) if path_a_count > 0 else 0.0003
        projected_monthly_100k = (100_000 * (resolution_rate / 100.0)) * avg_saving_per_path_a

        return {
            "total_requests": total_requests,
            "path_a_count": path_a_count,
            "path_b_count": path_b_count,
            "local_resolution_rate": round(resolution_rate, 1),
            "total_tokens_saved": tokens_saved,
            "total_cost_saved_usd": round(cost_saved_gpt4o, 4),
            "claude_sonnet_saved_usd": round(cost_saved_claude, 4),
            "gpt_4o_mini_saved_usd": round(cost_saved_mini, 4),
            "projected_monthly_savings_100k": round(projected_monthly_100k, 2),
            "avg_latency_path_a": round(avg_a, 1),
            "avg_latency_path_b": round(avg_b, 1),
        }


def get_recent_logs(limit: int = 50) -> list[dict[str, Any]]:
    """Retrieve the latest request logs formatted for the dashboard."""
    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, query, intent, confidence, language, path,
                   tokens_saved, latency_ms, reason
            FROM request_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]


def export_logs_csv() -> str:
    """Export all telemetry logs as a CSV string for enterprise audit & evaluation."""
    import csv
    import io

    init_db()
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, query, intent, confidence, language, path,
                   tokens_prompt, tokens_completion, tokens_saved, cost_saved_usd, latency_ms, reason
            FROM request_logs
            ORDER BY id ASC
            """
        )
        rows = cur.fetchall()

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "ID", "Timestamp", "Query", "Intent", "Confidence", "Language", "Path",
            "PromptTokens", "CompletionTokens", "TokensSaved", "CostSavedUSD", "LatencyMS", "Reason"
        ])
        for r in rows:
            writer.writerow(list(r))
        return output.getvalue()

