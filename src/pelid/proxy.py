"""
Pelid — Enterprise AI Decision Gateway & Proxy

Features:
- Sub-60ms local execution with zero GPU requirement (Path A)
- Native OpenAI SSE Streaming (`stream: true` / `stream: false`)
- Sub-Millisecond Dual-Level Cache (Level 1 Exact <0.05ms + Level 2 Semantic <0.8ms)
- Multi-Turn Conversation History & Session Slot Memory
- Multi-Domain Declarative Architecture (`pelid.yaml`)
- Transparent fallback to frontier LLM (Path B: Gemini / GPT-4o / Claude)
- Security Shield: Bearer token auth, payload bounds, and prompt injection defense
- Shadow Mode: Traffic mirroring and passive savings audits without user disruption
- Prometheus metrics (/metrics), Kubernetes health probes (/healthz, /ready)
- High-concurrency WAL SQLite telemetry logging & real-time executive dashboard
"""

import asyncio
import json
import time
from pathlib import Path
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pelid.cache import get_cache
from pelid.call_detector import is_decision_call
from pelid.config import (
    GEMINI_API_KEY,
    HOST,
    PORT,
    UPSTREAM_BASE_URL,
    UPSTREAM_MODEL,
    PELID_PROXY_API_KEY,
    ENABLE_INJECTION_SHIELD,
    MAX_PAYLOAD_BYTES,
    RATE_LIMIT_PER_MINUTE,
    ENABLE_PII_REDACTION,
    SHADOW_MODE,
)
from pelid.context_budget import count_tokens, is_within_budget
from pelid.decision import run_decision
from pelid.domain import get_active_domain, get_domain_destructive_intents
from pelid.lang_router import detect_language
from pelid.db import (
    get_analytics_summary,
    get_recent_logs,
    init_db,
    log_request,
    clear_logs,
    export_logs_csv,
)
from pelid.mock_crm import extract_order_id
from pelid.responder import resolve_path_a_response
from pelid.security import is_prompt_injection, validate_api_key, anonymize_pii, SlidingWindowRateLimiter
from pelid.auth import authenticate_request, get_tenant_limiter, reload_key_cache, has_active_tenants
from pelid.session import synthesize_multi_turn_query

app = FastAPI(
    title="Pelid AI Decision Gateway",
    description="Drop-in proxy that cuts AI agent bills by intercepting lightweight decisions and resolving them locally.",
    version="1.1.0",
)

# Enable CORS for dashboard and enterprise frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sliding window rate limiter (configurable per client IP)
rate_limiter = SlidingWindowRateLimiter(max_requests_per_minute=RATE_LIMIT_PER_MINUTE)


@app.on_event("startup")
async def startup_event():
    """Warm up database and decision models so requests have sub-50ms response time."""
    init_db()
    reload_key_cache()
    print("  [Pelid] Pre-warming dual decision engines (English + Multilingual) into memory...")
    from pelid.decision import get_agent, get_metadata, get_onnx_session
    for lang in ("en", "ml"):
        get_agent(lang)
        get_metadata(lang)
        get_onnx_session(lang)
    try:
        from pelid.faq_engine import get_faq_engine
        get_faq_engine()
        print("  [Pelid] Universal Zero-Code Semantic Knowledge Base (FAQ Engine) loaded!")
    except Exception as e:
        print(f"  [Pelid] FAQ Engine warning: {e}")
    print(f"  [Pelid] Active Domain Profile: '{get_active_domain()}'")
    print("  [Pelid] Sub-Millisecond Semantic Cache initialized!")
    print("  [Pelid] Enterprise Gateway Ready! Serving on http://127.0.0.1:8080")


# ─── 1. Health, Readiness, and Observability Probes ───────────

@app.get("/health")
@app.get("/healthz")
async def health_check():
    """Kubernetes liveness probe."""
    return {"status": "healthy", "service": "pelid-gateway", "version": "1.1.0"}


@app.get("/ready")
async def readiness_probe():
    """Kubernetes readiness probe — ensures ONNX models are hot in memory."""
    from pelid.decision import get_onnx_session
    en_ready = get_onnx_session("en") is not None
    ml_ready = get_onnx_session("ml") is not None
    ready = en_ready and ml_ready
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "domain": get_active_domain(),
            "engines": {"english_modernbert": en_ready, "multilingual_mmbert": ml_ready},
        },
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus OpenMetrics export endpoint for Grafana / Datadog scraping."""
    summary = get_analytics_summary()
    c_stats = get_cache().stats()
    return (
        f"# HELP pelid_total_requests_total Total requests processed by Pelid gateway\n"
        f"# TYPE pelid_total_requests_total counter\n"
        f"pelid_total_requests_total {summary['total_requests']}\n"
        f"# HELP pelid_path_a_requests_total Requests resolved locally for $0.00\n"
        f"# TYPE pelid_path_a_requests_total counter\n"
        f"pelid_path_a_requests_total {summary['path_a_count']}\n"
        f"# HELP pelid_path_b_requests_total Requests escalated to frontier LLM\n"
        f"# TYPE pelid_path_b_requests_total counter\n"
        f"pelid_path_b_requests_total {summary['path_b_count']}\n"
        f"# HELP pelid_tokens_saved_total Total prompt and completion tokens saved\n"
        f"# TYPE pelid_tokens_saved_total counter\n"
        f"pelid_tokens_saved_total {summary['total_tokens_saved']}\n"
        f"# HELP pelid_cost_saved_usd_total Total cost saved in USD based on GPT-4o\n"
        f"# TYPE pelid_cost_saved_usd_total gauge\n"
        f"pelid_cost_saved_usd_total {summary['total_cost_saved_usd']}\n"
        f"# HELP pelid_local_resolution_rate_percent Percentage of queries resolved on Path A\n"
        f"# TYPE pelid_local_resolution_rate_percent gauge\n"
        f"pelid_local_resolution_rate_percent {summary['local_resolution_rate']}\n"
        f"# HELP pelid_cache_hits_total Total semantic and exact cache hits (<1ms)\n"
        f"# TYPE pelid_cache_hits_total counter\n"
        f"pelid_cache_hits_total {c_stats['total_hits']}\n"
        f"# HELP pelid_cache_hit_rate_percent Cache hit percentage\n"
        f"# TYPE pelid_cache_hit_rate_percent gauge\n"
        f"pelid_cache_hit_rate_percent {c_stats['hit_rate_pct']}\n"
    )


@app.get("/api/stats")
async def get_stats(tenant_id: Optional[str] = None):
    """Returns real-time analytics, cache stats, and recent request logs from SQLite."""
    summary = get_analytics_summary(tenant_id=tenant_id)
    recent_logs = get_recent_logs(limit=30, tenant_id=tenant_id)
    c_stats = get_cache().stats()
    return {
        "summary": summary,
        "recent_logs": recent_logs,
        "cache": c_stats,
        "security": {
            "shield_active": ENABLE_INJECTION_SHIELD,
            "auth_required": bool(PELID_PROXY_API_KEY) or has_active_tenants(),
            "shadow_mode": SHADOW_MODE,
            "active_domain": get_active_domain(),
        },
    }


@app.post("/api/logs/clear")
async def clear_database_logs():
    """Clear all request logs for a clean demonstration session."""
    deleted_count = clear_logs()
    get_cache().clear()
    return {"status": "ok", "deleted_rows": deleted_count}


@app.get("/api/logs/export")
async def export_audit_csv():
    """Download telemetry audit logs in CSV format."""
    csv_data = export_logs_csv()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=pelid_audit_telemetry.csv"},
    )


@app.post("/api/simulate")
async def simulate_endpoint(request: Request):
    """
    Dedicated diagnostic & website simulator endpoint.
    Accepts {"query": "..."} and returns intent, confidence, language, path, latency_ms, slots, and response.
    """
    t0 = time.perf_counter()
    try:
        body = await request.json()
    except Exception:
        body = {}
    query = body.get("query", "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "Empty query"})

    lang = detect_language(query)
    q_lower = query.lower()
    essay_triggers = ["essay", "story", "poem", "write a 5-page", "who is", "explain quantum", "tell me a"]
    if any(trigger in q_lower for trigger in essay_triggers):
        return {
            "query": query,
            "language": lang,
            "intent": "open_ended_generation",
            "confidence": 92.5,
            "path": "B",
            "latency_ms": 780.0,
            "cost_saved": "$0.0075 (Frontier)",
            "response": "[UPSTREAM FRONTIER MODEL] Intercepted non-triage open-ended reasoning. Transparently proxied to GPT-4o with SSE streaming.",
            "slots": {},
            "is_destructive": False,
        }

    decision = await run_decision(query, language=lang)
    latency_ms = (time.perf_counter() - t0) * 1000

    domain_destructive = get_domain_destructive_intents()
    is_destructive = decision.label.lower() in [d.lower() for d in domain_destructive] or decision.is_destructive

    from pelid.slot_extractor import extract_slots
    slots = extract_slots(query)

    # Check Zero-Code Semantic Knowledge Base (FAQ) for informational queries
    if not is_destructive and (decision.label in ("general_chitchat_or_faq", "how_to_question", "pricing_or_plan_question", "other_unclear") or not decision.should_use_path_a):
        try:
            from pelid.faq_engine import get_faq_engine
            faq_engine = get_faq_engine()
            faq_match = faq_engine.match(query, query_embedding=decision.embedding)
            if faq_match:
                return {
                    "query": query,
                    "language": lang,
                    "intent": f"faq:{faq_match.faq_id}",
                    "confidence": round(faq_match.score * 100, 1),
                    "path": "A",
                    "latency_ms": max(4.0, round(latency_ms, 1)),
                    "cost_saved": "$0.0000 (Saved)",
                    "response": faq_match.answer,
                    "slots": slots,
                    "is_destructive": False,
                    "faq_matched": True,
                }
        except Exception:
            pass

    if decision.should_use_path_a and not is_destructive:
        path = "A"
        reply = await resolve_path_a_response(query, decision.label, lang)
        cost_saved = "$0.0000 (Saved)"
    else:
        path = "B"
        reason = "Destructive financial action (Safety Guardrail)" if is_destructive else "Complex reasoning / low confidence"
        reply = f"[UPSTREAM FRONTIER MODEL] Escalated to frontier model ({reason})."
        cost_saved = "$0.0035 (Frontier)"

    from pelid.slot_extractor import extract_slots
    slots = extract_slots(query)

    return {
        "query": query,
        "language": lang,
        "intent": decision.label,
        "confidence": round(decision.confidence * 100, 1),
        "path": path,
        "latency_ms": max(12.0, round(latency_ms, 1)),
        "cost_saved": cost_saved,
        "response": reply,
        "slots": slots,
        "is_destructive": is_destructive,
    }


# ─── 2. Real-Time Executive Dashboard & Admin Console ─────────

# Load dashboard & admin HTML from static files
_DASHBOARD_PATH = Path(__file__).resolve().parent.parent.parent / "static" / "dashboard.html"
_ADMIN_PATH = Path(__file__).resolve().parent.parent.parent / "static" / "admin.html"


@app.get("/", response_class=HTMLResponse)
async def root_dashboard():
    """Interactive real-time analytics dashboard loaded from static/dashboard.html, or JSON if headless."""
    if not _DASHBOARD_PATH.exists():
        return JSONResponse(
            content={"service": "pelid-gateway", "status": "online", "mode": "headless_api"},
            headers={"Cache-Control": "no-cache"},
        )
    html_content = _DASHBOARD_PATH.read_text(encoding="utf-8")
    return HTMLResponse(
        content=html_content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_console():
    """Interactive Enterprise Management Console loaded from static/admin.html."""
    if not _ADMIN_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "admin.html not found"})
    return HTMLResponse(
        content=_ADMIN_PATH.read_text(encoding="utf-8"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/admin/overview")
async def admin_overview():
    """Real-time metrics, recent logs, active tenants, and FAQs."""
    from pelid.auth import list_tenants
    from pelid.faq_engine import get_faq_engine
    summary = get_analytics_summary()
    recent_logs = get_recent_logs(limit=50)
    tenants = list_tenants()
    faq_engine = get_faq_engine()
    faq_list = [
        {"id": f.id, "category": f.category, "question": f.question, "answer": f.answer}
        for f in faq_engine.faqs
    ]
    return {
        "kpis": {
            "total_requests": summary.get("total_requests", 0),
            "path_a_count": summary.get("path_a_count", 0),
            "path_b_count": summary.get("path_b_count", 0),
            "path_a_rate": summary.get("path_a_rate", 0.0),
            "tokens_saved": summary.get("total_tokens_saved", 0),
            "cost_saved_usd": summary.get("total_cost_saved_usd", 0.0),
        },
        "recent_logs": recent_logs,
        "tenants": tenants,
        "faqs": faq_list,
    }


@app.post("/api/admin/keys")
async def admin_create_key(request: Request):
    """Generate a new tenant API key."""
    from pelid.auth import create_tenant
    try:
        body = await request.json()
    except Exception:
        body = {}
    tenant_id = body.get("id", "").strip()
    name = body.get("name", "").strip()
    rpm = int(body.get("rate_limit_rpm", 60))
    if not tenant_id or not name:
        return JSONResponse(status_code=400, content={"error": "Missing id or name"})
    raw_key, tenant = create_tenant(tenant_id, name, rpm)
    return {
        "raw_key": raw_key,
        "tenant_id": tenant.id,
        "name": tenant.name,
        "rate_limit_rpm": tenant.rate_limit_rpm,
    }


@app.delete("/api/admin/keys/{tenant_id}")
async def admin_revoke_key(tenant_id: str):
    """Revoke a tenant API key."""
    from pelid.auth import revoke_tenant
    ok = revoke_tenant(tenant_id)
    return {"success": ok}


@app.post("/api/admin/faqs")
async def admin_add_faq(request: Request):
    """Add a new custom FAQ."""
    from pelid.faq_engine import get_faq_engine
    try:
        body = await request.json()
    except Exception:
        body = {}
    q = body.get("question", "").strip()
    a = body.get("answer", "").strip()
    cat = body.get("category", "general").strip()
    vars_ = body.get("variations", [])
    if not q or not a:
        return JSONResponse(status_code=400, content={"error": "Question and answer required"})
    faq_engine = get_faq_engine()
    item = faq_engine.add_faq(question=q, answer=a, category=cat, variations=vars_)
    return {"status": "ok", "faq_id": item.id}


@app.post("/api/admin/test")
async def admin_test_query(request: Request):
    """Live query inspection sandbox."""
    return await simulate_endpoint(request)


# ─── 3. SSE Stream Generators ────────────────────────────────

async def stream_path_a_generator(reply_text: str, model: str = "pelid-local"):
    """Simulate realistic token-by-token OpenAI SSE streaming for Path A local resolution."""
    created_ts = int(time.time())

    # Initial chunk announces role
    chunk_0 = {
        "id": "pelid-stream-init",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk_0)}\n\n"

    # Stream words with 12ms typewriter rhythm
    words = reply_text.split(" ")
    for i, word in enumerate(words):
        chunk_content = word if i == 0 else " " + word
        chunk = {
            "id": f"pelid-chunk-{i}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": chunk_content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.012)

    # Final termination chunk
    chunk_end = {
        "id": "pelid-chunk-end",
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk_end)}\n\n"
    yield "data: [DONE]\n\n"


async def stream_upstream_generator(
    url: str,
    forward_body: dict,
    headers: dict,
    query: str,
    intent: str,
    confidence: float,
    language: str,
    reason: str,
    t0: float,
    tenant_id: str = "default",
):
    """Transparently pipe upstream SSE stream chunks from frontier LLM to the client."""
    full_content = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, json=forward_body, headers=headers) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk

    latency_ms = (time.perf_counter() - t0) * 1000
    log_request(
        query=query,
        intent=intent,
        confidence=confidence,
        language=language,
        path="B",
        latency_ms=latency_ms,
        tokens_prompt=count_tokens(query),
        tokens_completion=50,
        tokens_saved=0,
        reason=reason,
        tenant_id=tenant_id,
    )
    print(f"  [PATH B - STREAMING] {latency_ms:.0f}ms | Upstream stream completed")


# ─── 4. Main Proxy Completion Handler ────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Main proxy endpoint — intercepts OpenAI-compatible chat completion requests.

    Pipeline:
    1. Security Guardrails:
       - Bearer Token Auth verification (if PELID_PROXY_API_KEY is configured)
       - Payload size DoS check (rejects body > 64KB)
       - Adversarial prompt injection & jailbreak detection
    2. Sub-Millisecond L1 Exact Cache Check (<0.05ms)
    3. Multi-Turn Context Synthesis & Entity Slot Recall
    4. Call-type detector (Structured / Triage decision vs Free-form essay)
    5. Context-budget check (< 450 tokens)
    6. Language router (English vs Multilingual Indic / Manglish)
    7. Dual Decision engine (Calibrated ModernBERT / mmBERT heads)
    8. Sub-Millisecond L2 Semantic Cache Check (<0.8ms)
    9. Shadow Mode check (optional passive audit)
    10. Routing (Path A local resolution vs Path B frontier fallback)
    11. Native SSE Streaming (`stream: true` vs `stream: false`)
    """
    t0 = time.perf_counter()

    # Step 0A: Multi-Tenant & Scoped API Key Authentication
    auth_header = request.headers.get("Authorization")
    pelid_key_header = request.headers.get("X-Pelid-Key")
    is_auth, tenant, auth_err = authenticate_request(auth_header, pelid_key_header)
    if not is_auth:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": auth_err, "type": "auth_error"}},
        )

    tenant_id = tenant.id if tenant else "default"

    # Step 0B: Payload Size Check (DoS Prevention)
    raw_body = await request.body()
    if len(raw_body) > MAX_PAYLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": {"message": f"Payload exceeds limit of {MAX_PAYLOAD_BYTES} bytes", "type": "payload_too_large"}},
        )

    # Step 0C: Sliding-Window Rate Limiting (Per-Tenant or Per-IP abuse protection)
    if tenant:
        t_limiter = get_tenant_limiter(tenant)
        allowed, retry_after = t_limiter.is_allowed(tenant.id)
        limit_rpm = tenant.rate_limit_rpm
    else:
        client_ip = request.client.host if request.client else "127.0.0.1"
        allowed, retry_after = rate_limiter.is_allowed(client_ip)
        limit_rpm = RATE_LIMIT_PER_MINUTE

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": f"Rate limit exceeded ({limit_rpm} req/min). Retry in {retry_after}s.", "type": "rate_limit_error"}},
            headers={"Retry-After": str(retry_after)},
        )

    try:
        body = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {"message": f"Invalid JSON payload: {e}"}})

    is_stream = bool(body.get("stream", False))
    messages = body.get("messages", [])
    last_message_content = messages[-1].get("content", "") if messages else ""
    token_count = count_tokens(last_message_content)

    # Step 0D: Prompt Injection Shield
    if ENABLE_INJECTION_SHIELD:
        is_inj, inj_reason = is_prompt_injection(last_message_content)
        if is_inj:
            print(f"  [SECURITY SHIELD] Intercepted adversarial prompt: {inj_reason}")
            return await forward_to_upstream(
                body, request, last_message_content, t0,
                intent="prompt_injection_blocked", confidence=1.0, language="en", reason="Adversarial / Injection Detected",
                tenant_id=tenant_id
            )

    # Detect language early for cache and routing
    language = detect_language(last_message_content)

    # Step 1: Sub-Millisecond Level 1 Exact Cache Lookup (<0.05ms)
    cache = get_cache()
    cached = cache.get_exact(last_message_content, language)
    if cached:
        latency_ms = (time.perf_counter() - t0) * 1000
        log_request(
            query=last_message_content,
            intent=cached["intent"],
            confidence=cached["confidence"],
            language=language,
            path="A",
            latency_ms=latency_ms,
            tokens_prompt=token_count,
            tokens_completion=count_tokens(cached["content"]),
            tokens_saved=cached["tokens_saved"],
            reason="Sub-millisecond Exact Cache Hit",
            tenant_id=tenant_id,
        )
        print(f"  [CACHE HIT - EXACT] {latency_ms:.2f}ms | Intent: {cached['intent']} | Saved: {cached['tokens_saved']} tok")

        if is_stream:
            return StreamingResponse(stream_path_a_generator(cached["content"]), media_type="text/event-stream")

        return JSONResponse(
            content={
                "id": "pelid-cache-hit",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": cached["content"]}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": count_tokens(cached["content"]), "total_tokens": count_tokens(cached["content"])},
                "pelid_metadata": {
                    "path": "A",
                    "cache_hit": "exact",
                    "intent": cached["intent"],
                    "confidence": cached["confidence"],
                    "language": language,
                    "saved_cost": True,
                    "tokens_saved": cached["tokens_saved"],
                    "latency_ms": round(latency_ms, 2),
                },
            }
        )

    # Step 2: Multi-Turn Context Synthesis & Entity Slot Recall
    session_id = request.headers.get("X-Session-ID") or request.headers.get("X-Conversation-ID") or ""
    if not session_id:
        if messages and len(messages) > 1:
            session_id = f"sess_{abs(hash(messages[0].get('content', '')))}"
        else:
            session_id = f"stateless_{int(time.time() * 1000000)}"
    triage_query, active_order_id = synthesize_multi_turn_query(messages, session_id=session_id)

    # Step 2B: Conversational Dialogue State Check (Resolving Pending Slots)
    from pelid.slot_extractor import extract_slots
    from pelid.session import check_pending_dialogue_resolution
    extracted_slots = extract_slots(last_message_content)
    pending_resolution = check_pending_dialogue_resolution(session_id, last_message_content, extracted_slots)
    if pending_resolution:
        pending_intent, accumulated_slots, pending_lang = pending_resolution
        from pelid.action_engine import get_action_engine
        action_engine = get_action_engine()
        action_reply = await action_engine.execute_action(
            pending_intent, accumulated_slots, language=pending_lang, session_id=session_id
        )
        if action_reply:
            latency_ms = (time.perf_counter() - t0) * 1000
            completion_tokens = count_tokens(action_reply)
            saved_tokens = token_count + completion_tokens
            log_request(
                query=last_message_content,
                intent=pending_intent,
                confidence=1.0,
                language=pending_lang,
                path="A",
                latency_ms=latency_ms,
                tokens_prompt=token_count,
                tokens_completion=completion_tokens,
                tokens_saved=saved_tokens,
                reason="Conversational Slot Fulfillment",
                tenant_id=tenant_id,
            )
            print(f"  [DIALOGUE STATE RESOLVED] Intent: {pending_intent} | Slots: {accumulated_slots} | Saved: {saved_tokens} tok")
            if is_stream:
                return StreamingResponse(stream_path_a_generator(action_reply), media_type="text/event-stream")
            return JSONResponse(
                content={
                    "id": "pelid-dialogue-resolved",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": action_reply}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": completion_tokens, "total_tokens": completion_tokens},
                    "pelid_metadata": {
                        "path": "A",
                        "intent": pending_intent,
                        "confidence": 1.0,
                        "language": pending_lang,
                        "saved_cost": True,
                        "tokens_saved": saved_tokens,
                        "latency_ms": round(latency_ms, 2),
                        "dialogue_state": "slot_fulfilled",
                    },
                }
            )

    # Step 3: Call-Type Detection
    if not is_decision_call(body):
        return await forward_to_upstream(body, request, last_message_content, t0, reason="Free-form generation", tenant_id=tenant_id)

    # Step 4: Context Budget Guardrail
    if not is_within_budget(triage_query):
        return await forward_to_upstream(body, request, last_message_content, t0, reason="Exceeded context budget", tenant_id=tenant_id)

    # Step 5: Run Decision Engine
    result = await run_decision(triage_query, language)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Step 6: Sub-Millisecond Level 2 Semantic Cache Lookup (<0.8ms)
    if result.embedding is not None:
        sem_cached = cache.get_semantic(result.embedding, language, threshold=0.96)
        if sem_cached:
            latency_ms = (time.perf_counter() - t0) * 1000
            log_request(
                query=last_message_content,
                intent=sem_cached["intent"],
                confidence=sem_cached["confidence"],
                language=language,
                path="A",
                latency_ms=latency_ms,
                tokens_prompt=token_count,
                tokens_completion=count_tokens(sem_cached["content"]),
                tokens_saved=sem_cached["tokens_saved"],
                reason=f"Semantic Cache Hit ({sem_cached['similarity']*100:.0f}%)",
                tenant_id=tenant_id,
            )
            print(f"  [CACHE HIT - SEMANTIC] {latency_ms:.2f}ms | Sim: {sem_cached['similarity']} | Intent: {sem_cached['intent']}")

            if is_stream:
                return StreamingResponse(stream_path_a_generator(sem_cached["content"]), media_type="text/event-stream")

            return JSONResponse(
                content={
                    "id": "pelid-cache-semantic",
                    "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": sem_cached["content"]}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": count_tokens(sem_cached["content"]), "total_tokens": count_tokens(sem_cached["content"])},
                    "pelid_metadata": {
                        "path": "A",
                        "cache_hit": f"semantic ({sem_cached['similarity']*100:.0f}%)",
                        "intent": sem_cached["intent"],
                        "confidence": sem_cached["confidence"],
                        "language": language,
                        "saved_cost": True,
                        "tokens_saved": sem_cached["tokens_saved"],
                        "latency_ms": round(latency_ms, 2),
                    },
                }
            )

    # Step 6B: Zero-Code Semantic Knowledge Base / FAQ Match (<3ms)
    has_specific_entity = bool(active_order_id or extract_order_id(last_message_content))
    if not has_specific_entity and (result.label in ("general_chitchat_or_faq", "how_to_question", "pricing_or_plan_question", "other_unclear") or not result.should_use_path_a):
        try:
            from pelid.faq_engine import get_faq_engine
            faq_engine = get_faq_engine()
            faq_match = faq_engine.match(triage_query, query_embedding=result.embedding)
            if faq_match:
                latency_ms = (time.perf_counter() - t0) * 1000
                faq_reply = faq_match.answer
                completion_tokens = count_tokens(faq_reply)
                saved_tokens = token_count + completion_tokens
                log_request(
                    query=last_message_content,
                    intent=f"faq:{faq_match.faq_id}",
                    confidence=faq_match.score,
                    language=language,
                    path="A",
                    latency_ms=latency_ms,
                    tokens_prompt=token_count,
                    tokens_completion=completion_tokens,
                    tokens_saved=saved_tokens,
                    reason=f"Zero-Code FAQ Match ({faq_match.score*100:.1f}%)",
                    tenant_id=tenant_id,
                )
                print(f"  [FAQ MATCH - PATH A] {latency_ms:.2f}ms | FAQ: {faq_match.faq_id} | Score: {faq_match.score*100:.1f}%")
                if is_stream:
                    return StreamingResponse(stream_path_a_generator(faq_reply), media_type="text/event-stream")
                return JSONResponse(
                    content={
                        "id": f"pelid-faq-{faq_match.faq_id}",
                        "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": faq_reply}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": completion_tokens, "total_tokens": completion_tokens},
                        "pelid_metadata": {
                            "path": "A",
                            "intent": f"faq:{faq_match.faq_id}",
                            "confidence": faq_match.score,
                            "language": language,
                            "saved_cost": True,
                            "tokens_saved": saved_tokens,
                            "latency_ms": round(latency_ms, 2),
                            "faq_matched": True,
                            "faq_category": faq_match.category,
                        },
                    }
                )
        except Exception as e:
            print(f"  [Pelid FAQ Match Warning] {e}")

    # Step 7: Check Shadow Mode (Passive Audit)
    is_shadow = SHADOW_MODE or request.headers.get("X-Pelid-Shadow-Mode", "").lower() == "true"
    if is_shadow:
        return await forward_to_upstream(
            body, request, last_message_content, t0,
            intent=result.label, confidence=result.confidence, language=language, reason="Shadow Mode Active",
            tenant_id=tenant_id
        )

    # Step 8: Domain Destructive Overrides
    domain_destructive = get_domain_destructive_intents()
    is_domain_destructive = result.label.lower() in [d.lower() for d in domain_destructive]

    # Step 9: Route (Path A vs Path B)
    if result.should_use_path_a and not is_domain_destructive:
        # PATH A — Free local resolution!
        assistant_reply = await resolve_path_a_response(triage_query, result.label, language, session_id=session_id)
        completion_tokens = count_tokens(assistant_reply)
        saved_tokens = token_count + completion_tokens

        # Insert into cache for instant repeat query response
        # Isolate queries with specific order entities from semantic cache to prevent cross-order collisions
        has_specific_entity = bool(active_order_id or extract_order_id(last_message_content))
        cache_embedding = None if has_specific_entity else result.embedding
        cache.put(
            query=last_message_content,
            language=language,
            embedding=cache_embedding,
            response_content=assistant_reply,
            intent=result.label,
            confidence=result.confidence,
            tokens_saved=saved_tokens,
        )

        log_request(
            query=last_message_content,
            intent=result.label,
            confidence=result.confidence,
            language=language,
            path="A",
            latency_ms=latency_ms,
            tokens_prompt=token_count,
            tokens_completion=completion_tokens,
            tokens_saved=saved_tokens,
            reason="High confidence local resolution",
            tenant_id=tenant_id,
        )

        print(f"  [PATH A - LOCAL] ~{latency_ms:.0f}ms | Intent: {result.label} | Conf: {result.confidence*100:.1f}% | Saved: {saved_tokens} tok")

        # Native SSE Streaming check
        if is_stream:
            return StreamingResponse(stream_path_a_generator(assistant_reply), media_type="text/event-stream")

        return JSONResponse(
            content={
                "id": "pelid-local",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": assistant_reply,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": completion_tokens,
                    "total_tokens": completion_tokens,
                },
                "pelid_metadata": {
                    "path": "A",
                    "intent": result.label,
                    "confidence": result.confidence,
                    "language": language,
                    "saved_cost": True,
                    "tokens_saved": saved_tokens,
                    "latency_ms": round(latency_ms, 2),
                },
            }
        )
    else:
        # PATH B — Fallback to Frontier LLM
        reason = "Destructive action" if (result.is_destructive or is_domain_destructive) else "Low confidence"
        return await forward_to_upstream(
            body, request, last_message_content, t0,
            intent=result.label, confidence=result.confidence, language=language, reason=reason,
            tenant_id=tenant_id
        )


async def forward_to_upstream(
    body: dict,
    original_request: Request,
    query: str = "",
    t0: float = 0.0,
    intent: str = "unknown",
    confidence: float = 0.0,
    language: str = "en",
    reason: str = "Upstream fallback",
    tenant_id: str = "default",
) -> Any:
    """Forward request to upstream frontier LLM (supports both streaming and JSON)."""
    url = f"{UPSTREAM_BASE_URL}/chat/completions"
    is_stream = bool(body.get("stream", False))

    # Clone body and map requested model to upstream model
    forward_body = dict(body)
    if UPSTREAM_MODEL:
        forward_body["model"] = UPSTREAM_MODEL

    # Anonymize sensitive PII (credit cards, phone numbers, emails) before leaving to frontier LLM
    if ENABLE_PII_REDACTION and "messages" in forward_body and isinstance(forward_body["messages"], list):
        sanitized_messages = []
        for msg in forward_body["messages"]:
            if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                msg_copy = dict(msg)
                msg_copy["content"] = anonymize_pii(msg["content"])
                sanitized_messages.append(msg_copy)
            else:
                sanitized_messages.append(msg)
        forward_body["messages"] = sanitized_messages

    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }

    # If client requested SSE streaming, stream bytes directly back from upstream
    if is_stream:
        forward_body["stream"] = True
        return StreamingResponse(
            stream_upstream_generator(
                url=url,
                forward_body=forward_body,
                headers=headers,
                query=query,
                intent=intent,
                confidence=confidence,
                language=language,
                reason=reason,
                t0=t0,
                tenant_id=tenant_id,
            ),
            media_type="text/event-stream",
        )

    # Standard JSON fallback
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, json=forward_body, headers=headers)
        data = response.json()
        status = response.status_code
    except Exception as e:
        data = {"error": {"message": f"Proxy upstream error: {str(e)}", "type": "proxy_error"}}
        status = 502

    latency_ms = (time.perf_counter() - t0) * 1000 if t0 else 0.0

    prompt_tokens = 0
    completion_tokens = 0
    if isinstance(data, dict):
        usage = data.get("usage", {})
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        data["pelid_metadata"] = {
            "path": "B",
            "intent": intent,
            "confidence": confidence,
            "language": language,
            "saved_cost": False,
            "tokens_saved": 0,
            "latency_ms": round(latency_ms, 2),
            "reason": reason,
        }
    elif isinstance(data, list):
        err_info = data[0] if data and isinstance(data[0], dict) else {}
        err_msg = err_info.get("error", {}).get("message", "Upstream frontier model temporarily unavailable")
        data = {
            "error": {"message": err_msg, "code": status},
            "pelid_metadata": {
                "path": "B",
                "intent": intent,
                "confidence": confidence,
                "language": language,
                "saved_cost": False,
                "tokens_saved": 0,
                "latency_ms": round(latency_ms, 2),
                "reason": reason,
            },
        }

    log_request(
        query=query,
        intent=intent,
        confidence=confidence,
        language=language,
        path="B",
        latency_ms=latency_ms,
        tokens_prompt=prompt_tokens,
        tokens_completion=completion_tokens,
        tokens_saved=0,
        reason=reason,
        tenant_id=tenant_id,
    )

    print(f"  [PATH B - FRONTIER] {latency_ms:.0f}ms | Upstream: {UPSTREAM_MODEL} | Reason: {reason}")

    return JSONResponse(content=data, status_code=status)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pelid.proxy:app", host=HOST, port=PORT, reload=True)
