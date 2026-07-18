"""
Eval harness - running DistilBERT and Claude over the gold set and outputs the comparison 
numbers (accuracy, F1, latency, cost).

"""

import csv
import json 
import time
import statistics
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from src.predict import predict as distilbert_predict
from src.claude_classifier import classify as claude_classify

GOLD_PATH = "data/gold_set.csv"
RESULTS_DIR = Path("results")

# Claude Haiku 4.5 pricing. USD per Mtoken
CLAUDE_INPUT_RATE = 1.00 / 1_000_000
CLAUDE_OUTPUT_RATE = 5.00 / 1_000_000


def category_for(row_id):
    """Map gold-set id ranges to hard-case categories."""
    i = int(row_id)
    if 1 <= i <= 13:
        return "clear_positive"
    if 14 <= i <= 26:
        return "clear_negative"
    if 27 <= i <= 31:
        return "sarcasm"
    if 32 <= i <= 36:
        return "mixed"
    if 37 <= i <= 40:
        return "faint_praise"
    if 41 <= i <= 44:
        return "negation"
    if 45 <= i <= 47:
        return "terse"
    if 48 <= i <= 50:
        return "twist"
    return "other"

def load_gold(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
    

def timed_call(fn, text, retries=3):
    '''
    Call a classifier, timing only successful attempts. Retries transient API errors.
    '''
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            out = fn(text)
            return out, time.perf_counter() - t0
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

def run_model(fn, rows):
    timed_call(fn, rows[0]["text"])          # warm-up: exclude cold start from timing
    records = []
    for r in rows:
        out, latency = timed_call(fn, r["text"])
        records.append({
            "id": r["id"],
            "category": category_for(r["id"]),
            "true": r["label"].strip().lower(),   # normalize: gold labels may be capitalized
            "pred": out["label"],
            "latency_s": latency,
            "input_tokens": out.get("input_tokens"),
            "output_tokens": out.get("output_tokens"),
        })
    return records

def summarize(name, records):
    y_true = [r["true"] for r in records]
    y_pred = [r["pred"] for r in records]
    latencies = [r["latency_s"] for r in records]

    if records[0]["input_tokens"] is not None:
        cost_per_call = statistics.mean(
            r["input_tokens"] * CLAUDE_INPUT_RATE + r["output_tokens"] * CLAUDE_OUTPUT_RATE
            for r in records
        )
        cost_per_1k = cost_per_call * 1000
    else:
        cost_per_1k = 0.0

    cats = {}
    for r in records:
        cats.setdefault(r["category"], []).append(r["true"] == r["pred"])
    
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred, pos_label="positive", zero_division=0), 4),
        "latency_median_s": round(statistics.median(latencies), 4),
        "latency_mean_s": round(statistics.mean(latencies), 4),
        "cost_per_1k_usd": round(cost_per_1k, 4),
        "category_accuracy": {c: round(sum(v) / len(v), 3) for c, v in cats.items()},
    }
    

def main():
    rows = load_gold(GOLD_PATH)
    print(f"Loaded {len(rows)} gold examples\n")

    distil = run_model(distilbert_predict, rows)
    claude = run_model(claude_classify, rows)

    summaries = [summarize("DistilBERT", distil), summarize("Claude (Haiku 4.5)", claude)]

    RESULTS_DIR.mkdir(exist_ok = True)
    with open (RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(summaries, f, indent = 2)
    with open (RESULTS_DIR / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "true", "distilbert", "claude"])
        for d, c in zip(distil, claude):
            w.writerow([d["id"], d["category"], d["true"], d["pred"], c["pred"]])

    print(f"{'Model':<22}{'Acc':>7}{'F1':>8}{'Latency(s)':>13}{'Cost/1k$':>11}")
    for s in summaries:
        print(f"{s['model']:<22}{s['accuracy']:>7}{s['f1']:>8}"
              f"{s['latency_median_s']:>13}{s['cost_per_1k_usd']:>11}")
    print("\nPer-category accuracy:")
    for s in summaries:
        print(f"  {s['model']}:")
        for cat, acc in s["category_accuracy"].items():
            print(f"    {cat:<16}{acc}")


if __name__ == "__main__":
    main()
