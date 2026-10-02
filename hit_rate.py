import glob
import json
import sys


# lowercases and collapses all whitespace so line breaks in chunk text don't break phrase matching
def normalize(text):
    return " ".join(text.split()).lower()


# maps each question's text to its evidence phrase, read from the verified question files
def load_evidence():
    evidence = {}
    for path in sorted(glob.glob("eval_questions/*.json")):
        with open(path) as f:
            for item in json.load(f):
                evidence[item["question"]] = item["evidence"]
    return evidence


# returns the 1-based position of the first retrieved chunk containing the evidence phrase, or None if no chunk has it
def first_hit_position(chunks, phrase):
    target = normalize(phrase)
    for position, chunk in enumerate(chunks, start=1):
        if target in normalize(chunk):
            return position
    return None


# prints hit rate overall and per document for one saved run file, plus every miss
def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        files = [f for f in sorted(glob.glob("eval_runs/*.json")) if "smoke" not in f]
        path = files[-1]

    with open(path) as f:
        run = json.load(f)
    evidence = load_evidence()

    per_doc = {}
    misses = []
    skipped = 0

    for q in run["questions"]:
        phrase = evidence.get(q["question"])
        if q["status"] != "ok" or phrase is None:
            skipped += 1
            continue
        position = first_hit_position(q["contexts"], phrase)
        per_doc.setdefault(q["doc"], []).append(position)
        if position is None:
            misses.append((q["doc"], q["question"][:70], len(q["contexts"])))

    print("file:", path)
    print("skipped (not ok or no evidence phrase):", skipped, "\n")

    all_positions = []
    for doc in sorted(per_doc):
        positions = per_doc[doc]
        hits = [p for p in positions if p is not None]
        all_positions.extend(positions)
        avg = round(sum(hits) / len(hits), 2) if hits else None
        print(f"{doc}: hit rate {len(hits)}/{len(positions)}, average position of first hit = {avg}")

    total_hits = len([p for p in all_positions if p is not None])
    print(f"\nOVERALL hit rate: {total_hits}/{len(all_positions)}")

    if misses:
        print("\nMisses (no retrieved chunk contained the evidence phrase):")
        for doc, text, n in misses:
            print(f"   {doc} | {n} chunks | {text}")


if __name__ == "__main__":
    main()