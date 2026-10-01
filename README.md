# ⚡ Pelid — Enterprise AI Decision Gateway & Proxy

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Inference Latency](https://img.shields.io/badge/Decision_Latency-sub--60ms-emerald)](https://github.com/)
[![Local Cost](https://img.shields.io/badge/Path_A_Cost-$0.00-success)](https://github.com/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://github.com/)

**Pelid** is an open-source, drop-in **AI Decision Gateway (Reverse Proxy)** designed to cut enterprise AI agent bills by **60–80%** and slash decision latency from 1,200ms to **sub-60ms**.

It intercepts standard OpenAI-compatible API calls, evaluates customer support and agent triage decisions locally using **calibrated non-autoregressive decision models (Laya ModernBERT / mmBERT)** for **$0.00**, and only routes to frontier LLMs (Gemini / GPT-4o / Claude 3.5 Sonnet) when deep reasoning or destructive operations are required.

---

## 🎯 The Core Problem

When an enterprise customer service chatbot or AI agent assists a user, it executes **several lightweight reasoning steps** before generating a final response:
- *"What is the user's intent?"* (triage / routing)
- *"Is this a standard FAQ or tracking lookup?"*
- *"Is this a destructive financial request (like refund or account deletion)?"*
- *"Does this need a human supervisor or frontier model?"*

Sending 100% of these routine queries to expensive frontier models costs **$2,500 – $10,000+ per million requests** and incurs 800ms–2,000ms latency. Over 65% of customer support requests can be resolved deterministically in **30–50ms for $0.00**.

---

## 🏗️ Architecture

```
Developer AI Agent / Chatbot UI / Python SDK
                    │
                    ▼  POST /v1/chat/completions (stream: true / false)
┌──────────────────────────────────────────────────────────────────┐
│  PELID ENTERPRISE PROXY GATEWAY (localhost:8080)                 │
│                                                                  │
│  [Shield] Bearer Auth · 64KB Payload Limit · Injection Defense   │
│  [Cache L1] Sub-Millisecond Exact Hash Cache (<0.05ms)           │
│  [Session] Multi-Turn History & Slot Memory (Order IDs, items)   │
│  [Router] Script & Language Router (English, Hinglish, Manglish) │
│           ├── English      ──> ModernBERT Head (1024-d, 5ms)     │
│           └── Multilingual ──> mmBERT Head     (768-d,  6ms)     │
│  [Cache L2] Sub-Millisecond Semantic Cosine Cache (<0.8ms)       │
│  [Domain] Declarative Profiles (eCommerce, Food, Manufacturing)  │
│  [Shadow] Passive Traffic Mirroring & Savings Audit              │
│                                                                  │
│     ┌───────────────────────────────┴───────────────────────┐    │
│     │                                                       │    │
│  [PATH A: LOCAL RESOLUTION]              [PATH B: FRONTIER LLM]  │
│  - Intent confidence >= 70%              - Low confidence <70%   │
│  - Non-destructive actions               - Destructive actions   │
│  - Built-in Mock CRM Lookup              - Complex free-form gen │
│  - Trilingual Conversational Reply       - Gemini / GPT-4o       │
│  - SSE Typewriter Streaming (12ms/tok)   - Upstream SSE Pass-thru│
│  - Latency: ~30-60 ms (Cache: <1ms)      - Latency: ~800-2000ms  │
│  - Cost: $0.00                           - Cost: Upstream rate   │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚡ 2-Line Drop-in Integration

Pelid is **100% OpenAI-API compatible**. You do not need to rewrite your agent code or change prompt schemas. Simply update `base_url`:

```python
from openai import OpenAI

# Point your client to Pelid instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="your-pelid-or-upstream-key",
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Where is my order #9921?"}],
)

# If routine: Answered locally in 45ms for $0.00!
print(response.choices[0].message.content)
```

## 🌟 Enterprise Features

- **⚡ Sub-Millisecond Dual-Level Cache:** Level 1 Exact Hash cache (<0.05ms) paired with Level 2 Semantic Cosine similarity search (<0.8ms) across frozen embeddings. Dynamic entity isolation prevents cross-customer order collisions.
- **🌊 Native OpenAI SSE Streaming:** Full `stream: true` Server-Sent Events (`text/event-stream`) support for both local Path A answers and transparent passthrough for frontier LLM streams.
- **🧠 Multi-Turn Dialogue Context & Slot Memory:** Tracks persistent entities (e.g. `Order ID #89214`) across conversational turns so follow-up questions (*"When will it reach my house?"*, *"Can you cancel it?"*) retain context and intent.
- **🌐 Trilingual Vernacular Understanding:** 100% evaluated accuracy on English, Romanized Hindi (Hinglish), and Romanized Malayalam (Manglish) customer support queries.
- **📦 Multi-Domain Manifest (`pelid.yaml`):** Declarative profile support for `ecommerce`, `food_delivery`, and `b2b_manufacturing` with domain-specific destructive action rules.
- **🛡️ Security Shield:** Real-time prompt injection detection, 64KB DoS payload limits, and optional Bearer token authentication.
- **🔍 Passive Shadow Mode:** Mirror live production traffic asynchronously to generate zero-risk 48-hour cost savings audits before touching user traffic.
- **📊 Real-Time Executive Dashboard:** Embedded single-page web tester with live SQLite telemetry, sub-millisecond cache counters, and 1-click CSV audit exports.

---

## 🚀 Quickstart

### Option 1: Docker (Recommended)
Zero dependencies, zero setup. Starts in 5 seconds with pre-warmed models:
```bash
# Clone and enter repo
git clone https://github.com/your-org/pelid.git
cd pelid

# Run with Docker Compose
docker-compose up -d
```
Open **`http://localhost:8080`** in your browser to view the real-time executive telemetry dashboard.

---

### Option 2: Local Python Environment
```bash
git clone https://github.com/your-org/pelid.git
cd pelid

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -e .

# Launch Pelid Gateway
python -m uvicorn pelid.proxy:app --host 127.0.0.1 --port 8080
```

---

## 📊 Live Verification Benchmark (17/17 Passed — 100.0%)

Pelid natively parses Romanized vernacular queries (English, Hinglish, and Manglish) and responds in the user's exact dialect:

| Query | Dialect | Intent | Routed To | Decision Latency | Cost Saved |
|---|---|---|---|---|---|
| `"Where is my order #9921?"` | English | `order_status` | **PATH A** | **38.2 ms** | **100% ($0.00)** |
| `"ennte order evide?"` | Manglish | `order_status` | **PATH A** | **49.1 ms** | **100% ($0.00)** |
| `"aree mera package tuuta hua hai"` | Hinglish | `wrong_or_defective_item` | **PATH A** | **52.4 ms** | **100% ($0.00)** |
| `"Ente purchase-inte GST bill copy engane download cheyyaam?"` | Manglish | `invoice_or_receipt` | **PATH A** | **55.0 ms** | **100% ($0.00)** |
| `"how to book stuff?"` | English | `how_to_question` | **PATH A** | **42.1 ms** | **100% ($0.00)** |
| `"Placed order-inte delivery location maattaan pattumo?"` | Manglish | `shipping_address_change` | **PATH B** | *Frontier LLM* | *Safety Escalation* |
| `"Please delete my account and erase all my data"` | English | `delete_account` | **PATH B** | *Frontier LLM* | *Safety Escalation* |

Run the benchmark anytime:
```bash
python scripts/evaluate_dual_engine.py
```

---

## 🛡️ Enterprise Security & Guardrails

1. **Prompt Injection Shield:** Real-time regex and token signature analysis intercepting adversarial jailbreaks (`"ignore all previous instructions"`, `"DAN mode"`, etc.) before local heads execute.
2. **Context Budget Guardrail:** Requests exceeding 450 tokens bypass Tier 0 to ensure zero memory exhaustion.
3. **Destructive Action Interception:** Actions involving financial loss, refunds, address modification, or account deletion are **strictly forbidden** from local automation and unconditionally escalated to supervised frontier models.
4. **Bearer Token Authentication:** Optional `PELID_PROXY_API_KEY` prevents unauthorized proxy usage.

---

## 📈 Observability & Metrics

Pelid comes out of the box with enterprise observability:
- **Dashboard:** Interactive real-time metrics feed at `http://localhost:8080`.
- **Prometheus Metrics:** OpenMetrics endpoint at `/metrics` for Datadog / Grafana scraping.
- **Kubernetes Probes:** `/healthz` (liveness) and `/ready` (model warmness).
- **Audit Export:** Instant CSV export via `/api/logs/export`.

---

## 📜 License

Licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for details.
