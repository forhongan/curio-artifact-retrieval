"""Constrained follow-up on cached evidence; keeps v11 results unchanged."""

import hashlib
import json
from pathlib import Path

import numpy as np

from curio.evaluate.monotone_evidence import fit_monotone, predict_monotone
from curio.evaluate.targeted_boundary import calibrate, policy_metrics, validate_partitions
from curio.retrieve.factor_boundary import COLOR_ATOMS, QUERY_ATOMS, and_decision

ROOT = Path(__file__).resolve().parents[1]


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def features(score):
    if score["color_excluded"]:
        return [0.0, 0.0]
    return [score["positive"]["yes_no_logit"], -score["counter"]["yes_no_logit"]]


def guard(score, atom):
    return score["color_excluded"] or (
        atom in COLOR_ATOMS
        and (score["positive"]["yes_no_logit"] <= 0 or score["counter"]["yes_no_logit"] >= 0)
    )


def main():
    old = ROOT / "eval/boundary_factors_v11"
    old_out = ROOT / "artifacts/factor_boundary_v11"
    base = ROOT / "eval/boundary_monotone_v12"
    out = ROOT / "artifacts/monotone_boundary_v12"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((base / "protocol.json").read_text())
    for name, digest in protocol["inputs"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen input changed: " + name)
    components = readl(old / "components.jsonl")
    validate_partitions(components)
    scores = {r["key"]: r for r in readl(old_out / "component_scores.jsonl")}
    heads, calibrated, predicted = {}, {}, {}
    for atom in sorted({r["atom"] for r in components}):
        rows = [r for r in components if r["atom"] == atom]
        evidence = [scores[r["key"]] for r in rows]
        x = np.array([features(s) for s in evidence])
        fit = np.array([r["split"] == "train" and r["grade"] >= 0 for r in rows])
        head = fit_monotone(x[fit], [r["decision"] for r, use in zip(rows, fit) if use])
        p = predict_monotone(head, x)
        p[[guard(s, atom) for s in evidence]] = 0
        dev = np.array([r["split"] == "dev" for r in rows])
        cal = calibrate([r for r, use in zip(rows, dev) if use], p[dev])
        calibrated[atom] = cal
        heads[atom] = {
            **head,
            "threshold": cal["threshold"],
            "fit_keys": [r["key"] for r, use in zip(rows, fit) if use],
        }
        predicted.update(
            {
                r["key"]: {"score": float(value), "color_excluded": s["color_excluded"]}
                for r, s, value in zip(rows, evidence, p)
            }
        )
    thresholds = {a: h["threshold"] for a, h in heads.items()}
    results = []
    for row in readl(old / "labels.jsonl"):
        values = {a: predicted[a + ":" + row["item_id"]] for a in QUERY_ATOMS[row["qid"]]}
        results.append({**row, **and_decision(row["qid"], values, thresholds)})
    stress_scores = {r["key"]: r["components"] for r in readl(old_out / "stress_scores.jsonl")}
    stress = []
    for row in readl(old / "stress_labels.jsonl"):
        values = {}
        for atom, s in stress_scores[row["key"]].items():
            p = float(predict_monotone(heads[atom], [features(s)])[0])
            values[atom] = {
                "score": 0.0 if guard(s, atom) else p,
                "color_excluded": s["color_excluded"],
            }
        stress.append({**row, **and_decision(row["qid"], values, thresholds)})
    test = [r for r in results if r["split"] == "test"]
    report = {
        "protocol": protocol,
        "release_approved": False,
        "calibration": calibrated,
        "test": policy_metrics(test, [r["yes"] for r in test]),
        "stress": policy_metrics(stress, [r["yes"] for r in stress]),
        "per_query": {
            q: {
                name: policy_metrics(
                    [r for r in subset if r["qid"] == q],
                    [r["yes"] for r in subset if r["qid"] == q],
                )
                for name, subset in [("test", test), ("stress", stress)]
            }
            for q in sorted(QUERY_ATOMS)
        },
    }
    (out / "heads.json").write_text(
        json.dumps({"release_approved": False, "heads": heads, "protocol": protocol}) + "\n"
    )
    (base / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    (base / "stress_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in stress)
    )
    (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {"test": report["test"], "stress": report["stress"], "per_query": report["per_query"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
