"""Paired intent bootstrap and transparent result-pool inspection summary."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/relevance_v5"


def bootstrap(left, right):
    difference = np.asarray(right) - np.asarray(left)
    rng = np.random.default_rng(5523)
    samples = difference[rng.integers(0, len(difference), size=(10000, len(difference)))].mean(
        axis=1
    )
    return {
        "delta": float(difference.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "unit": "20 semantic intents, paired EN/ZH average; 10000 bootstrap samples",
    }


def main():
    records = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["records"]
    lookup = {(r["model"], r["method"], r["language"]): r for r in records}

    def values(model, method, metric):
        a = [lookup[model, method, lang]["per_query"] for lang in ["en", "zh"]]
        return [(x[metric] + y[metric]) / 2 for x, y in zip(*a)]

    comparisons = []
    for name, model, method in [
        ("teammate checkpoint", "team_comp_lora", "dense"),
        ("raw HSV fusion", "production_lora", "aux_raw_color"),
        ("GrabCut HSV fusion", "production_lora", "aux_gc_color"),
        ("DINO graph .1", "production_lora", "dino_graph_0.1"),
    ]:
        comparisons.append(
            {
                "comparison": name,
                "metric": "ndcg@10",
                **bootstrap(
                    values("production_lora", "dense", "ndcg@10"), values(model, method, "ndcg@10")
                ),
            }
        )
    comparisons.append(
        {
            "comparison": "logistic vs adaptive GMM",
            "metric": "set_f1",
            **bootstrap(
                values("production_lora", "adaptive_gmm", "set_f1"),
                values("production_lora", "calibrated", "set_f1"),
            ),
        }
    )
    comparisons.append(
        {
            "comparison": "HSV threshold vs dense threshold",
            "metric": "set_f1",
            **bootstrap(
                values("production_lora", "dense", "set_f1"),
                values("production_lora", "aux_raw_color", "set_f1"),
            ),
        }
    )
    (OUT / "paired_bootstrap.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    manifest = json.loads(
        (ROOT / "research/teammate_review/source/docs/resources_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    entries = manifest.get("assets", manifest.get("packages", []))
    # Manifest schema also provides named asset maps in some release versions.
    if isinstance(entries, dict):
        entries = list(entries.values())
    verification = []
    for path in (ROOT / "research/teammate_review/downloads").glob("*.zip"):
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        matching = [r for r in entries if path.name in str(r)]
        verification.append(
            {
                "file": path.name,
                "sha256": digest,
                "manifest_match": any(r.get("sha256") == digest for r in matching),
            }
        )
    (OUT / "download_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
