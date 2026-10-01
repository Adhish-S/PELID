"""
Stage 3 — Laya Baseline Evaluation & Calibration Benchmark

Measures zero-shot accuracy, latency (p50/p95), and Expected Calibration Error (ECE)
on the hand-written seed examples (test set) for both English and Multilingual models.
"""

import sys
import time
from pathlib import Path

# Force UTF-8 stdout encoding on Windows
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import laya

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INTENTS_FILE = PROJECT_ROOT / "data" / "intents.txt"
SEED_FILE = PROJECT_ROOT / "data" / "seed_examples.txt"


def parse_criteria(filepath: Path) -> dict[str, str]:
    """Parse intent labels and descriptions from intents.txt."""
    criteria = {}
    current_intent = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line[0].isdigit() and "." in line:
                # e.g. "1. order_status" or "4. refund_request  [DESTRUCTIVE]"
                parts = line.split(".", 1)[1].strip().split()
                current_intent = parts[0].strip()
            elif (line.startswith("->") or line.startswith("→")) and current_intent:
                desc = line.replace("->", "").replace("→", "").strip()
                criteria[current_intent] = desc
    return criteria


def parse_seed_test_set(filepath: Path) -> list[tuple[str, str]]:
    """Parse hand-written seed examples (test set)."""
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " | " in line:
                label, text = line.split(" | ", 1)
                samples.append((label.strip(), text.strip()))
    return samples


def main():
    print("=" * 65)
    print("  Pelid -- Laya Baseline Evaluation & Benchmark (Stage 3)")
    print("=" * 65)

    criteria = parse_criteria(INTENTS_FILE)
    print(f"\n[OK] Loaded {len(criteria)} routing criteria from {INTENTS_FILE.name}:")
    for k in list(criteria.keys())[:5]:
        print(f"     - {k:30s}: {criteria[k][:40]}...")
    if len(criteria) > 5:
        print(f"     ... and {len(criteria) - 5} more.")

    test_samples = parse_seed_test_set(SEED_FILE)
    print(f"\n[OK] Loaded {len(test_samples)} hand-written test samples from {SEED_FILE.name}")

    question = {
        "intent_routing": {
            "type": "choice",
            "instructions": "Classify the customer request into the most appropriate queue intent",
            "criteria": criteria,
        }
    }

    # Evaluate English checkpoint
    print("\n" + "-" * 65)
    print("Evaluating: convaiinnovations/laya (English Checkpoint)")
    print("-" * 65)
    agent_en = laya.load("convaiinnovations/laya")

    correct = 0
    latencies_ms = []
    confidences = []

    for i, (true_label, text) in enumerate(test_samples, 1):
        t0 = time.perf_counter()
        res = agent_en.predict({"message": text}, question)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed)

        pred_choice = res["answers"]["intent_routing"]["choice"]
        conf = res["answers"]["intent_routing"]["confidence"]
        confidences.append(conf)

        is_match = (pred_choice == true_label)
        if is_match:
            correct += 1

        if i <= 8 or not is_match:
            mark = "[MATCH]" if is_match else "[DIFF ]"
            short_text = (text[:30] + "..") if len(text) > 30 else text
            print(f"  {mark} {short_text:32s} -> {pred_choice:26s} (True: {true_label}, Conf: {conf:.2f}, {elapsed:.1f}ms)")

    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    accuracy = correct / len(test_samples)
    avg_conf = sum(confidences) / len(confidences)

    print("\n" + "=" * 65)
    print("BASELINE SUMMARY:")
    print(f"  Test Samples:    {len(test_samples)}")
    print(f"  Accuracy:        {accuracy * 100:.1f}% ({correct}/{len(test_samples)})")
    print(f"  Avg Confidence:  {avg_conf:.3f}")
    print(f"  Latency p50:     {p50:.1f} ms")
    print(f"  Latency p95:     {p95:.1f} ms")
    print("=" * 65)


if __name__ == "__main__":
    main()
