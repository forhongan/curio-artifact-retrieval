"""Replay previously exposed gallery failures without fitting or threshold changes."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from curio.evaluate.targeted_boundary import policy_metrics
from curio.retrieve.factor_boundary import FactorEvidence, QUERY_ATOMS, and_decision
from curio.retrieve.joint_vlm import LocalVQA

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "eval/boundary_factors_v11"
    out = ROOT / "artifacts/factor_boundary_v11"
    protocol = json.loads((base / "stress_protocol.json").read_text())
    path = base / "stress_labels.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["labels_sha256"]:
        raise ValueError("Frozen stress labels changed")
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    bundle = json.loads((out / "heads.json").read_text())
    heads = bundle["heads"]
    runtime = LocalVQA(ROOT)
    if bundle["vlm_identity"] != runtime.identity:
        raise ValueError("Scoring model changed")
    scorer = FactorEvidence(runtime)
    thresholds = {a: h["threshold"] for a, h in heads.items()}
    results = []
    try:
        with (out / "stress_scores.jsonl").open("w") as stream:
            for n, r in enumerate(rows):
                if (
                    hashlib.sha256((ROOT / r["image_path"]).read_bytes()).hexdigest()
                    != r["image_sha256"]
                ):
                    raise ValueError("Reviewed image changed")
                values, scores = {}, {}
                for atom in QUERY_ATOMS[r["qid"]]:
                    s = scorer.score(r["image_path"], atom)
                    scores[atom] = s
                    h = heads[atom]
                    x = (np.array(s["features"]) - h["mean"]) / h["scale"]
                    probability = float(expit(x @ np.array(h["coefficient"]) + h["intercept"]))
                    values[atom] = {
                        "score": 0.0 if s["color_excluded"] else probability,
                        "color_excluded": s["color_excluded"],
                    }
                results.append({**r, **and_decision(r["qid"], values, thresholds)})
                stream.write(json.dumps({"key": r["key"], "components": scores}) + "\n")
                stream.flush()
                if (n + 1) % 10 == 0 or n + 1 == len(rows):
                    print(f"Gallery regression {n + 1}/{len(rows)}", flush=True)
    finally:
        runtime.close()
    report = {
        "protocol": protocol,
        "test": policy_metrics(results, [r["yes"] for r in results]),
        "per_query": {
            q: policy_metrics(
                [r for r in results if r["qid"] == q], [r["yes"] for r in results if r["qid"] == q]
            )
            for q in sorted({r["qid"] for r in results})
        },
        "release_approved": False,
    }
    (base / "stress_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    (base / "stress_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
