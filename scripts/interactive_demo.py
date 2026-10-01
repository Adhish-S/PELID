"""
Interactive Pelid Live Demo

Run this script to test any customer message interactively.
Type phrases in English, Hinglish, or Malayalam, or paste real agent prompts.

Usage:
    python scripts/interactive_demo.py
"""

import asyncio
import sys
import time

# Ensure UTF-8 output on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pelid.context_budget import count_tokens, is_within_budget
from pelid.decision import run_decision
from pelid.lang_router import detect_language
from pelid.responder import resolve_path_a_response


def print_banner():
    print("\n" + "=" * 70)
    print("  🚀  PELID -- AI DECISION PROXY (Interactive Live Demo)")
    print("=" * 70)
    print("  Test any query in English, Hinglish, or Malayalam.")
    print("  Type 'quit' or 'exit' to exit.")
    print("-" * 70)


async def evaluate_query(query: str):
    t0 = time.perf_counter()

    # 1. Context budget check
    tokens = count_tokens(query)
    within_budget = is_within_budget(query)

    # 2. Language Detection
    lang = detect_language(query)
    if lang == "ml":
        lang_name = "Malayalam / Manglish [ml]"
    elif lang == "hi":
        lang_name = "Hindi / Hinglish [hi]"
    else:
        lang_name = "English [en]"

    # 3. Decision Engine
    res = await run_decision(query, language=lang)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 4. Display Results
    print(f"\n  Query:         \"{query}\"")
    print(f"  Tokens:        {tokens} tokens (Context budget: {'Within limit [<=450]' if within_budget else 'OVER BUDGET'})")
    print(f"  Language:      {lang_name}")
    print(f"  Intent:        {res.label}")
    print(f"  Confidence:    {res.confidence * 100:.1f}%")
    print(f"  Latency:       {elapsed_ms:.1f} ms")

    # Routing determination
    if not within_budget:
        print(f"  --> ROUTE:     [PATH B - Frontier LLM] (Reason: Exceeded context budget of 450 tokens)")
    elif res.is_destructive:
        print(f"  --> ROUTE:     [PATH B - Frontier LLM] (Safety Rule: '{res.label}' is an irreversible action!)")
    elif res.should_use_path_a:
        reply = await resolve_path_a_response(query, res.label, language=lang)
        print(f"  --> ROUTE:     [PATH A - Local Bypass (FREE!)] (High confidence, resolved in {elapsed_ms:.0f}ms)")
        print(f"  --> RESPONSE:  \"{reply}\"")
    else:
        print(f"  --> ROUTE:     [PATH B - Frontier LLM] (Low confidence < 70%, escalates safely)")

    # Show top 3 candidates
    if res.all_probabilities:
        sorted_probs = sorted(res.all_probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        top_candidates = ", ".join([f"{k}: {v*100:.1f}%" for k, v in sorted_probs])
        print(f"  Top Probs:     {top_candidates}")
    print("-" * 70)


async def main():
    print_banner()

    # Preset showcase samples
    showcase = [
        "Where is my package #88921?",
        "ennte order evide?",
        "aree mera package tuuta hua hai",
        "Mera refund kab tak aayega bhai?",
        "Please cancel my order #90211 right now",
        "App open aakumpol crash aakunn",
        "Permanent delete my account and clear data",
        "What are your customer support operating hours today?",
    ]

    print("\n[Running Quick Showcase on 7 Sample Queries...]\n")
    for q in showcase:
        await evaluate_query(q)

    print("\n" + "=" * 70)
    print("  NOW TRY YOUR OWN QUERIES BELOW:")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter query > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Exiting demo. Goodbye!")
                break
            await evaluate_query(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo. Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())
