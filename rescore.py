import json
import math
import sys

from evaluation import score_with_ragas, mean_ignoring_nan, METRIC_NAMES

# Usage: python rescore.py eval_runs/<run>.json
# Re-runs only the RAGAS judge for questions whose scores came back NaN (for example when the quota ran out
# mid-scoring). The chatbot is NOT asked again: the saved answers and retrieved chunks are scored as they are.


# True if a question was answered but at least one of its metric scores is missing
def needs_rescore(record):
    if record["status"] != "ok":
        return False
    scores = record.get("scores", {})
    return not scores or any(v is None or (isinstance(v, float) and math.isnan(v)) for v in scores.values())


# re-scores the gaps, recomputes the means over all questions, and writes <run>_rescored.json next to the original
def main(path):
    with open(path) as f:
        run = json.load(f)

    todo = [r for r in run["questions"] if needs_rescore(r)]
    print(f"{len(todo)} question(s) to re-score")
    if todo:
        score_with_ragas(todo)

    run["mean_scores"] = {
        name: mean_ignoring_nan([r["scores"][name] for r in run["questions"] if "scores" in r])
        for name in METRIC_NAMES
    }
    still_missing = sum(1 for r in run["questions"] if needs_rescore(r))
    run["meta"]["rescored"] = len(todo)
    run["meta"]["still_unscored"] = still_missing

    out_path = path.replace(".json", "_rescored.json")
    with open(out_path, "w") as f:
        json.dump(run, f, indent=2)
    print("Mean scores:", run["mean_scores"])
    print(f"Still unscored: {still_missing}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main(sys.argv[1])
