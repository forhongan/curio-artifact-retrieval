"""Score only fully judged pooled top5; never infer full-gallery recall."""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    root = Path("artifacts/reference_eval")
    mapping = json.loads((root / "pool_private_mapping.json").read_text())
    judgments = []
    for name in ("grades_root.json", "grades_agent.json"):
        judgments.extend(json.loads((root / name).read_text(encoding="utf-8")))
    grades = {r["alias"]: r["grade"] for r in judgments}
    if len(grades) != len(judgments) or set(grades) != {r["alias"] for r in mapping}:
        raise ValueError("Duplicate or missing blind grades")
    if any(g not in (0, 1, 2) for g in grades.values()):
        raise ValueError("Grade must be 0/1/2")
    by_query = {}
    for row in mapping:
        by_query.setdefault(row["qid"], {})[row["item_id"]] = grades[row["alias"]]
    records = json.loads((root / "rankings.json").read_text())
    metrics = []
    for row in records:
        qrels = by_query[row["qid"]]
        ranking = row["ranking"][:5]
        if set(ranking) - qrels.keys():
            raise ValueError("Unjudged top5 result")
        actual = [qrels[key] for key in ranking]
        discount = np.log2(np.arange(2, 7))
        padded = np.pad(actual, (0, 5 - len(actual)))
        ideal = sorted(qrels.values(), reverse=True)[:5]
        idcg = sum((2 ** g - 1) / d for g, d in zip(ideal, discount))
        metrics.append({"qid": row["qid"], "method": row["method"], "status": row["status"],
                        "p@5": sum(g == 2 for g in actual) / 5,
                        "pooled_ndcg@5": float(np.sum((2. ** padded - 1) / discount) / idcg) if idcg else None,
                        "grades_top5": actual})
    methods = sorted({r["method"] for r in records})
    summary = {}
    rng = np.random.default_rng(6631)
    for method in methods:
        rows = [r for r in metrics if r["method"] == method]
        values = np.array([r["p@5"] for r in rows])
        boot = rng.choice(values, (10000, len(values)), replace=True).mean(1)
        summary[method] = {"p@5": float(values.mean()),
                           "p@5_bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
                           "pooled_ndcg@5": float(np.mean([r["pooled_ndcg@5"] for r in rows])),
                           "query_coverage": sum(r["status"] == "ok" for r in rows) / len(rows),
                           "intent_count": len(rows)}
        comparator = {r["qid"]: r["p@5"] for r in metrics if r["method"] == "siglip_single"}
        differences = [r["p@5"] - comparator[r["qid"]] for r in rows]
        summary[method]["wins_ties_losses_vs_single_reference"] = {
            "wins": sum(d > 1e-8 for d in differences), "ties": sum(abs(d) <= 1e-8 for d in differences),
            "losses": sum(d < -1e-8 for d in differences)}
    report = {"scope": "Exploratory 8-intent full4692 ranking, method-blind pooled top5 AI judgments",
              "judged_query_image_pairs": len(mapping), "distinct_judged_images": len({r["item_id"] for r in mapping}),
              "summary": summary, "per_query": metrics,
              "judges": "Root AI: meiping/gourd/blue_white/celadon. Independent agent AI: bronze/jade/lacquer/textile. One judge per image, no human gold or interrater estimate.",
              "limits": ["Eight purposively selected visual intents; no test-set generalization or full-gallery recall estimate.",
                         "References and methods fixed before blind grading; no weights tuned to these grades.",
                         "Reference IDs explicitly curated, so this isolates visual matching from automatic reference selection quality.",
                         "Partial/uncertain grade1 is not a verified positive. Similarity cannot verify period/authenticity/material.",
                         "Specialist abstentions count as empty results for end-to-end P@5; invalid masks are not negative relevance labels.",
                         "Pooled nDCG ideal is restricted to observed pool. Wide bootstrap intervals and many comparisons: descriptive only."],
              "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [root / "protocol.json", root / "rankings.json", root / "grades_root.json", root / "grades_agent.json"]}}
    (root / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for method, score in summary.items():
        print(method, json.dumps(score))


if __name__ == "__main__":
    main()
