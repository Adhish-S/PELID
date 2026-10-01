"""
Pelid — Comprehensive Dual Decision Engine & Tri-Lingual Response Verification

Verifies:
1. Language classification: English ('en') vs Hinglish ('hi') vs Manglish ('ml')
2. Intent classification and calibrated confidence
3. Routing: Path A (Free Local Resolution) vs Path B (Destructive / Frontier LLM)
4. Ergonomic response matching the exact dialect of the user
"""

import asyncio
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pelid.lang_router import detect_language
from pelid.decision import run_decision
from pelid.responder import resolve_path_a_response

# (query, expected_lang, expected_intent, expected_path_a)
TEST_SUITE = [
    # 1. Manglish Tracking Query (User Report Case 1)
    ("ennte order evide?", "ml", "order_status", True),
    # 2. Hinglish Broken Package Query (User Report Case 2)
    ("aree mera package tuuta hua hai", "hi", "wrong_or_defective_item", True),
    # 3. English with Mock CRM Order ID lookup
    ("Where is my order #9921?", "en", "order_status", True),
    # 4. English with Apple MacBook Order ID lookup
    ("Where is my order #89214?", "en", "order_status", True),
    # 5. English with common keyboard typo ('wherre')
    ("wherre is my order", "en", "order_status", True),
    # 6. Manglish Broken / Defective Package
    ("ennte package pottiya vannath.", "ml", "wrong_or_defective_item", True),
    # 7. Hinglish with Typo & Slang ('oreder', 'kahaa')
    ("aree mera oreder kahaa hai?", "hi", "order_status", True),
    # 8. Manglish Slang & Informal Vocative ('eda', 'eedeya ulle')
    ("eda ennte package eedeya ulle?", "ml", "order_status", True),
    # 9. Destructive Refund Request (Must route to Path B)
    ("mera glass damaged aaya, enghana refund kittum?", "ml", "refund_request", False),
    # 10. English Greeting
    ("hi there, how are you?", "en", "general_chitchat_or_faq", True),
    # 11. Hinglish Polite Inquiry
    ("namaste, mera order kahan tak pahuncha?", "hi", "order_status", True),
    # 12. Destructive Account Deletion (Must route to Path B)
    ("Please delete my account and erase all my data", "en", "delete_account", False),
    # 13. Manglish GST Invoice Download (User Case)
    ("Ente purchase-inte GST bill copy engane download cheyyaam?", "ml", "invoice_or_receipt", True),
    # 14. English Booking How-To (User Case)
    ("how to book stuff?", "en", "how_to_question", True),
    # 15. Manglish Address Change (User Case - Destructive Action -> Path B)
    ("Placed order-inte delivery location maattaan pattumo?", "ml", "shipping_address_change", False),
    # 16. Manglish Missing Package Delivery (User Case)
    ("Appil delivered ennu kaanikunnu pakshe enikku item kittiyilla.", "ml", "order_delayed_or_missing", True),
    # 17. Hinglish Missing Package Delivery (User Case)
    ("App me delivered dikha raha hai par mujhe packet nahi mila.", "hi", "order_delayed_or_missing", True),
]


async def run_suite():
    print("=" * 85)
    print("  PELID DUAL-HEAD TRI-LINGUAL BENCHMARK (English / Hinglish / Manglish)")
    print("=" * 85)

    passed = 0
    total = len(TEST_SUITE)

    for query, exp_lang, exp_intent, exp_path_a in TEST_SUITE:
        t0 = time.perf_counter()
        lang = detect_language(query)
        result = await run_decision(query, language=lang)
        latency_ms = (time.perf_counter() - t0) * 1000

        route = "PATH A (LOCAL)" if result.should_use_path_a else "PATH B (FRONTIER)"
        lang_ok = (lang == exp_lang)
        intent_ok = (result.label == exp_intent)
        route_ok = (result.should_use_path_a == exp_path_a)

        status = "PASS" if (lang_ok and intent_ok and route_ok) else "FAIL"
        if status == "PASS":
            passed += 1

        print(f"\n[{status}] Query: \"{query}\"")
        print(f"       Lang: {lang} (Expected: {exp_lang}) | Intent: {result.label} (Expected: {exp_intent})")
        print(f"       Confidence: {result.confidence*100:.1f}% | Latency: {latency_ms:.1f}ms | Route: {route}")

        if result.should_use_path_a:
            reply = await resolve_path_a_response(query, result.label, language=lang)
            print(f"       --> Ergonomic Reply: \"{reply}\"")
        else:
            reason = "Destructive action override" if result.is_destructive else "Low confidence"
            print(f"       --> Escalation Reason: {reason}")

    print("\n" + "=" * 85)
    print(f"  BENCHMARK SUMMARY: {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print("=" * 85)


if __name__ == "__main__":
    asyncio.run(run_suite())
