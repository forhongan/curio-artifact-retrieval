"""Frozen fresh-holdout comparison of learned factor evidence and whole-query VQA."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from curio.evaluate.targeted_boundary import calibrate, policy_metrics, validate_partitions
from curio.retrieve.factor_boundary import FactorEvidence, and_decision, PROBES, VERSION
from curio.retrieve.joint_vlm import LocalVQA

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "eval/boundary_factors_v11"
    out = ROOT / "artifacts/factor_boundary_v11"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((base / "review_manifest.json").read_text())
    for name, digest in manifest["hashes"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen reviewed data changed")
    components = [json.loads(s) for s in (base / "components.jsonl").read_text().splitlines()]
    rows = [json.loads(s) for s in (base / "labels.jsonl").read_text().splitlines()]
    validate_partitions(components)
    validate_partitions(rows)
    runtime = LocalVQA(ROOT)
    scorer = FactorEvidence(runtime)
    scores, raw_scores = [], []
    try:
        with (out / "component_scores.jsonl").open("w") as stream:
            for n, r in enumerate(components):
                if (
                    hashlib.sha256((ROOT / r["image_path"]).read_bytes()).hexdigest()
                    != r["image_sha256"]
                ):
                    raise ValueError("Reviewed image changed")
                score = scorer.score(r["image_path"], r["atom"])
                scores.append(score)
                stream.write(json.dumps({"key": r["key"], **score}) + "\n")
                stream.flush()
                if (n + 1) % 20 == 0 or n + 1 == len(components):
                    print(f"Component evidence {n + 1}/{len(components)}", flush=True)
        for n, r in enumerate(rows):
            plan = runtime.plan(r["query_en"])
            if runtime.plan(r["query_zh"]).key != plan.key:
                raise ValueError("Bilingual plan mismatch")
            score = runtime.score(r["image_path"], plan)
            raw_scores.append(score)
            if (n + 1) % 20 == 0:
                print(f"Whole-query baseline {n + 1}/{len(rows)}", flush=True)
        (out / "raw_scores.jsonl").write_text(
            "".join(json.dumps({"key": r["key"], **s}) + "\n" for r, s in zip(rows, raw_scores))
        )
    finally:
        runtime.close()
    predictions, heads, component_report, thresholds = {}, {}, {}, {}
    for atom in PROBES:
        positions = [i for i, r in enumerate(components) if r["atom"] == atom]
        subset = [components[i] for i in positions]
        x = np.array([scores[i]["features"] for i in positions])
        fit = np.array([r["split"] == "train" and r["grade"] >= 0 for r in subset])
        scaler = StandardScaler().fit(x[fit])
        model = LogisticRegression(C=1, class_weight="balanced", max_iter=1000).fit(
            scaler.transform(x[fit]), [r["decision"] for r, use in zip(subset, fit) if use]
        )
        p = model.predict_proba(scaler.transform(x))[:, 1]
        p[[scores[i]["color_excluded"] for i in positions]] = 0
        dev = np.array([r["split"] == "dev" for r in subset])
        test = np.array([r["split"] == "test" for r in subset])
        cal = calibrate([r for r, use in zip(subset, dev) if use], p[dev])
        thresholds[atom] = cal["threshold"]
        component_report[atom] = {
            "calibration": cal,
            "test": policy_metrics(
                [r for r, use in zip(subset, test) if use], p[test] >= cal["threshold"]
            ),
        }
        heads[atom] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficient": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "threshold": cal["threshold"],
            "fit_keys": [r["key"] for r, use in zip(subset, fit) if use],
        }
        for j, i in enumerate(positions):
            predictions[components[i]["key"]] = {
                "score": float(p[j]),
                "color_excluded": scores[i]["color_excluded"],
            }
    results = []
    baseline_thresholds = {}
    for qid in sorted({r["qid"] for r in rows}):
        subset = [
            (r, s) for r, s in zip(rows, raw_scores) if r["qid"] == qid and r["split"] == "dev"
        ]
        baseline_thresholds[qid] = calibrate(
            [r for r, _ in subset], [s["score"] for _, s in subset]
        )
    for r, s in zip(rows, raw_scores):
        values = {
            atom: predictions[atom + ":" + r["item_id"]]
            for atom in PROBES
            if atom + ":" + r["item_id"] in predictions
        }
        decision = and_decision(r["qid"], values, thresholds)
        results.append(
            {
                **r,
                **decision,
                "raw_yes": s["score"] >= 0.5,
                "calibrated_raw_yes": s["score"] >= baseline_thresholds[r["qid"]]["threshold"],
            }
        )
    test = [r for r in results if r["split"] == "test"]
    report = {
        "version": VERSION,
        "release_approved": False,
        "bilingual_same_plan": True,
        "probes": PROBES,
        "components": component_report,
        "baseline_calibration": baseline_thresholds,
        "test": {
            m: policy_metrics(test, [r[key] for r in test])
            for m, key in [
                ("factors", "yes"),
                ("raw_vqa", "raw_yes"),
                ("calibrated_vqa", "calibrated_raw_yes"),
            ]
        },
        "per_query": {
            q: {
                m: policy_metrics(
                    [r for r in test if r["qid"] == q], [r[key] for r in test if r["qid"] == q]
                )
                for m, key in [
                    ("factors", "yes"),
                    ("raw_vqa", "raw_yes"),
                    ("calibrated_vqa", "calibrated_raw_yes"),
                ]
            }
            for q in sorted({r["qid"] for r in test})
        },
    }
    bundle = {
        "version": VERSION,
        "release_approved": False,
        "heads": heads,
        "probes": PROBES,
        "feature_code_sha256": hashlib.sha256(
            (ROOT / "src/curio/retrieve/factor_boundary.py").read_bytes()
        ).hexdigest(),
        "label_sha256": manifest["hashes"]["components.jsonl"],
        "vlm_identity": runtime.identity,
    }
    (out / "heads.json").write_text(json.dumps(bundle) + "\n")
    (base / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
    )
    (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
