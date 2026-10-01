"""
Stage 2 — Synthetic Dataset Generator

Reads hand-written seed examples from data/seed_examples.txt and uses
Gemini (free tier) to generate ~50 noisy variants per intent.

Output: data/training_data.csv (label, message)

Usage:
    python scripts/generate_training_data.py
"""

import csv
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ─── Setup ───────────────────────────────────────────────────

# Find the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GEMINI_API_KEY", "")
if not API_KEY:
    print("ERROR: GEMINI_API_KEY not found in .env file!")
    print("Make sure your .env file exists and contains: GEMINI_API_KEY=your-key-here")
    exit(1)

GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={API_KEY}"

SEED_FILE = PROJECT_ROOT / "data" / "seed_examples.txt"
OUTPUT_FILE = PROJECT_ROOT / "data" / "training_data.csv"

# Gemini free tier: 15 requests per minute -> wait 8 seconds between calls
RATE_LIMIT_DELAY = 8

# How many variants to generate per intent
VARIANTS_PER_INTENT = 50

# Retry settings for 503/429 errors
MAX_RETRIES = 3
RETRY_BASE_DELAY = 15  # seconds, doubles each retry


# ─── Read Seed Examples ──────────────────────────────────────

def read_seed_examples(filepath: Path) -> dict[str, list[str]]:
    """
    Parse seed_examples.txt and group messages by intent label.

    Returns:
        dict mapping intent_label -> list of example messages
    """
    intents: dict[str, list[str]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            # Parse "intent_label | message" format
            if " | " in line:
                label, message = line.split(" | ", 1)
                label = label.strip()
                message = message.strip()
                if label not in intents:
                    intents[label] = []
                intents[label].append(message)

    return intents


# ─── Generate Variants with Gemini ───────────────────────────

def build_prompt(intent_label: str, examples: list[str], count: int) -> str:
    """
    Build a prompt asking Gemini to generate noisy variants of the seed examples.
    """
    examples_text = "\n".join(f"  - {ex}" for ex in examples)

    return f"""You are a synthetic data generator for a customer support intent classifier.

TASK: Generate exactly {count} realistic customer messages that belong to the intent category "{intent_label}".

Here are real examples of this intent:
{examples_text}

RULES:
1. Each message should sound like a REAL customer typing quickly — include natural typos, abbreviations, and informal language.
2. Mix THREE languages randomly across the messages:
   - Pure English (~40% of messages)
   - Hindi in Roman script / Hinglish (~35% of messages)
   - Malayalam in Roman script / Manglish (~25% of messages)
3. Vary the message length — some very short ("refund kab milega"), some longer with context.
4. Include realistic details: random order numbers, product names, amounts in ₹, time references.
5. Do NOT repeat the exact seed examples I gave you — create genuinely new variations.
6. Do NOT add numbering, bullet points, or any formatting — just one message per line.
7. Do NOT add intent labels or any metadata — just the raw customer message text.

OUTPUT: Return ONLY the {count} messages, one per line, nothing else. No explanations, no headers."""


def call_gemini(prompt: str) -> str | None:
    """
    Call the Gemini API and return the generated text.
    Retries on 503 (overloaded) and 429 (rate limit) errors.
    Returns None if all retries fail.
    """
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 1.0,  # High temperature for diverse outputs
            "maxOutputTokens": 4096,
        }
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.post(
                GEMINI_URL,
                json=payload,
                timeout=60.0,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                print("  [!] No text in response")
                return None

            elif response.status_code in (503, 429):
                # Server overloaded or rate limited - retry with backoff
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                if attempt < MAX_RETRIES:
                    print(f"  [!] Error {response.status_code} (attempt {attempt + 1}/{MAX_RETRIES + 1}) - retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  [!] Error {response.status_code} - all {MAX_RETRIES + 1} attempts failed")
                    return None
            else:
                print(f"  [!] API error {response.status_code}: {response.text[:200]}")
                return None

        except httpx.TimeoutException:
            print("  [!] Request timed out")
            return None
        except Exception as e:
            print(f"  [!] Unexpected error: {e}")
            return None

    return None


def parse_variants(raw_text: str) -> list[str]:
    """
    Parse Gemini's output into individual messages.
    Filters out empty lines and any lines that look like formatting artifacts.
    """
    lines = raw_text.strip().split("\n")
    variants = []

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Remove any numbering like "1.", "1)", "- " that Gemini might add despite instructions
        if line and line[0].isdigit():
            # Remove patterns like "1. ", "1) ", "01. "
            for i, ch in enumerate(line):
                if ch in ".)" and i < 4:
                    line = line[i + 1:].strip()
                    break
                elif not ch.isdigit():
                    break
        if line.startswith("- "):
            line = line[2:].strip()
        if line.startswith("* "):
            line = line[2:].strip()
        # Skip if empty after cleanup
        if line:
            variants.append(line)

    return variants


# ─── Main ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Pelid -- Synthetic Training Data Generator (Stage 2)")
    print("=" * 60)

    # Read seed examples
    if not SEED_FILE.exists():
        print(f"ERROR: Seed file not found at {SEED_FILE}")
        exit(1)

    intents = read_seed_examples(SEED_FILE)
    print(f"\nLoaded {sum(len(v) for v in intents.values())} seed examples across {len(intents)} intents\n")

    # Generate variants for each intent
    all_training_data: list[tuple[str, str]] = []
    total_intents = len(intents)

    for i, (label, examples) in enumerate(intents.items(), 1):
        print(f"[{i}/{total_intents}] Generating {VARIANTS_PER_INTENT} variants for: {label}")
        print(f"  Seed examples: {len(examples)}")

        prompt = build_prompt(label, examples, VARIANTS_PER_INTENT)
        raw_text = call_gemini(prompt)

        if raw_text:
            variants = parse_variants(raw_text)
            print(f"  [OK] Generated {len(variants)} variants")

            for msg in variants:
                all_training_data.append((label, msg))
        else:
            print(f"  [FAIL] Failed to generate variants for {label}")

        # Rate limit: wait between API calls (free tier = 15 RPM)
        if i < total_intents:
            print(f"  Waiting {RATE_LIMIT_DELAY}s (rate limit)...")
            time.sleep(RATE_LIMIT_DELAY)

    # Save to CSV
    print(f"\n{'=' * 60}")
    print(f"Saving {len(all_training_data)} training samples to {OUTPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "message"])  # Header
        for label, message in all_training_data:
            writer.writerow([label, message])

    # Print summary
    print(f"\n{'-' * 60}")
    print("SUMMARY")
    print(f"{'-' * 60}")

    intent_counts: dict[str, int] = {}
    for label, _ in all_training_data:
        intent_counts[label] = intent_counts.get(label, 0) + 1

    for label, count in sorted(intent_counts.items()):
        print(f"  {label:40s} {count:4d} samples")

    print(f"{'-' * 60}")
    print(f"  {'TOTAL':40s} {len(all_training_data):4d} samples")
    print(f"\n[DONE] Training data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
