import json
import os
import time
from datetime import datetime

from langgraph_backend import is_query_appropriate

# seconds between classifier calls, to stay under the free-tier limit of 15 requests per minute
PAUSE_SECONDS = 4.5


# runs the real guardrail classifier on one text; returns "allow", "block", or "error" if every attempt fails
def classify(text, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return "allow" if is_query_appropriate(text) else "block"
        except Exception as e:
            print(f"   attempt {attempt}/{attempts} failed: {type(e).__name__}")
            time.sleep(20 * attempt)
    return "error"


# classifies every case in guardrail_cases.json, prints catch rate, correct-allow rate and misses by category, and saves the results
def main():
    with open("guardrail_cases.json") as f:
        cases = json.load(f)

    results = []
    for i, case in enumerate(cases, start=1):
        predicted = classify(case["text"])
        verdict = "ok" if predicted == case["expected"] else ("ERROR" if predicted == "error" else "WRONG")
        print(f"[{i}/{len(cases)}] expected={case['expected']:5} got={predicted:5} {verdict:5} | {case['text'][:60]}")
        results.append({**case, "predicted": predicted})
        time.sleep(PAUSE_SECONDS)

    scored = [r for r in results if r["predicted"] != "error"]
    errors = len(results) - len(scored)

    should_block = [r for r in scored if r["expected"] == "block"]
    should_allow = [r for r in scored if r["expected"] == "allow"]
    caught = sum(1 for r in should_block if r["predicted"] == "block")
    kept = sum(1 for r in should_allow if r["predicted"] == "allow")

    print("\n=== Guardrail results ===")
    print(f"Catch rate (should-block that were blocked): {caught}/{len(should_block)}")
    print(f"Allowed correctly (should-allow that were allowed): {kept}/{len(should_allow)}")
    print(f"Cases excluded because every attempt errored: {errors}")

    print("\nBy category:")
    for category in sorted({r["category"] for r in scored}):
        group = [r for r in scored if r["category"] == category]
        right = sum(1 for r in group if r["predicted"] == r["expected"])
        print(f"   {category:28} {right}/{len(group)} correct")

    wrong = [r for r in scored if r["predicted"] != r["expected"]]
    if wrong:
        print("\nMisclassified (look at each before judging):")
        for r in wrong:
            print(f"   expected {r['expected']}, got {r['predicted']} | [{r['category']}] {r['text'][:80]}")

    os.makedirs("eval_runs", exist_ok=True)
    out = f"eval_runs/guardrail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to", out)


if __name__ == "__main__":
    main()