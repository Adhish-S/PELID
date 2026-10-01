"""
Client test script for Pelid proxy.
Simulates an OpenAI API client sending requests through the proxy at localhost:8080.

Usage:
    1. Start the proxy in one terminal:
       python -m uvicorn pelid.proxy:app --port 8080

    2. Run this test script in another terminal:
       python scripts/test_proxy_client.py
"""

import httpx
import time

PROXY_URL = "http://127.0.0.1:8080/v1/chat/completions"

test_prompts = [
    "Where is my package #88921?",
    "App crashes when I tap checkout",
    "Please cancel my order #90211 immediately",
    "Thank you so much help ke liye",
    "Explain quantum computing in 2 paragraphs",  # Free-form generation -> Path B
]

print("=" * 65)
print("  Sending test requests through Pelid Proxy (localhost:8080)")
print("=" * 65)

for p in test_prompts:
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": p}
        ]
    }

    t0 = time.perf_counter()
    try:
        r = httpx.post(PROXY_URL, json=payload, timeout=30.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if r.status_code == 200:
            data = r.json()
            meta = data.get("pelid_metadata", {})
            choice = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            path = meta.get("path", "B (Upstream)")
            tokens = usage.get("total_tokens", 0)

            print(f"\nPrompt:      \"{p}\"")
            print(f"Path:        Path {path} | Time: {elapsed_ms:.1f} ms | Tokens: {tokens}")
            print(f"Response:    {choice[:80]}...")
        else:
            print(f"\nPrompt:      \"{p}\" -> Error {r.status_code}: {r.text[:120]}")
    except httpx.ConnectError:
        print(f"\n[!] Could not connect to {PROXY_URL}.")
        print("    Please start the proxy first: python -m uvicorn pelid.proxy:app --port 8080")
        break

print("\n" + "=" * 65)
