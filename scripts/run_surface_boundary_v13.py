"""Learn red from object pixels, preserving other frozen v11 decisions."""

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from curio.evaluate.monotone_evidence import predict_monotone
from curio.evaluate.targeted_boundary import calibrate, policy_metrics
from curio.represent.object_surface import extract_surface
from curio.retrieve.factor_boundary import and_decision
from run_monotone_boundary_v12 import features, guard

ROOT = Path(__file__).resolve().parents[1]


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    old = ROOT / "eval/boundary_factors_v11"
    base = ROOT / "eval/boundary_surface_v13"
    out = ROOT / "artifacts/surface_boundary_v13"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((base / "protocol.json").read_text())
    qa = json.loads((base / "train_mask_review.json").read_text())
    component_rows = [r for r in readl(old / "components.jsonl") if r["atom"] == "red"]
    component_scores = {
        r["key"]: r for r in readl(ROOT / "artifacts/factor_boundary_v11/component_scores.jsonl")
    }
    stress_rows = [r for r in readl(old / "stress_labels.jsonl") if r["qid"] == "red_bowl"]
    stress_scores = {
        r["key"]: r for r in readl(ROOT / "artifacts/factor_boundary_v11/stress_scores.jsonl")
    }
    red_head = json.loads((ROOT / "artifacts/monotone_boundary_v12/heads.json").read_text())[
        "heads"
    ]["red"]
    surfaces = {}
    unique = {r["item_id"]: r for r in component_rows + stress_rows}
    for n, r in enumerate(unique.values()):
        s = extract_surface(ROOT / r["image_path"])
        surfaces[r["item_id"]] = {k: v for k, v in s.items() if k not in {"rgb", "mask"}}
        if (n + 1) % 30 == 0:
            print(f"Object surfaces {n + 1}/{len(unique)}", flush=True)
    fit = [
        r
        for r in component_rows
        if r["split"] == "train"
        and r["grade"] >= 0
        and surfaces[r["item_id"]]["valid"]
        and r["item_id"] not in qa["exclude_from_surface_fit"]
    ]
    model = LogisticRegression(C=10, class_weight="balanced", max_iter=1000).fit(
        np.sqrt([surfaces[r["item_id"]]["features"] for r in fit]), [r["decision"] for r in fit]
    )

    def red_score(ident, semantic):
        surface = surfaces[ident]
        if guard(semantic, "red") or not surface["valid"]:
            return 0.0
        visual = model.predict_proba(np.sqrt([surface["features"]]))[0, 1]
        meaning = predict_monotone(red_head, [features(semantic)])[0]
        return float(min(visual, meaning))

    red_prob = {
        r["item_id"]: red_score(r["item_id"], component_scores[r["key"]]) for r in component_rows
    }
    dev = [r for r in component_rows if r["split"] == "dev"]
    calibration = calibrate(dev, [red_prob[r["item_id"]] for r in dev])
    outputs = {}
    for name in ["predictions", "stress_predictions"]:
        results = []
        for r in readl(old / (name + ".jsonl")):
            if r["qid"] == "red_bowl":
                semantic = component_scores.get("red:" + r["item_id"])
                if semantic is None:
                    semantic = stress_scores[r["key"]]["components"]["red"]
                values = {
                    e["atom"]: {
                        "score": e["score"],
                        "color_excluded": e["reason"] == "color_not_observable",
                    }
                    for e in r["evidence"]
                }
                thresholds = {e["atom"]: e["threshold"] for e in r["evidence"]}
                values["red"] = {
                    "score": red_score(r["item_id"], semantic),
                    "color_excluded": semantic["color_excluded"],
                }
                thresholds["red"] = calibration["threshold"]
                r = {**r, **and_decision("red_bowl", values, thresholds)}
            results.append(r)
        outputs[name] = results
        (base / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        )
    test = [r for r in outputs["predictions"] if r["split"] == "test"]
    stress = outputs["stress_predictions"]
    report = {
        "protocol": protocol,
        "release_approved": False,
        "red_calibration": calibration,
        "surface_fit_count": len(fit),
        "area_check_valid": sum(s["valid"] for s in surfaces.values()),
        "surfaces": len(surfaces),
        "test": policy_metrics(test, [r["yes"] for r in test]),
        "stress": policy_metrics(stress, [r["yes"] for r in stress]),
        "red_bowl": {
            name: policy_metrics(
                [r for r in rs if r["qid"] == "red_bowl"],
                [r["yes"] for r in rs if r["qid"] == "red_bowl"],
            )
            for name, rs in [("test", test), ("stress", stress)]
        },
    }
    (out / "surface_features.json").write_text(json.dumps(surfaces) + "\n")
    (out / "red_surface_head.json").write_text(
        json.dumps(
            {
                "coefficient": model.coef_[0].tolist(),
                "intercept": float(model.intercept_[0]),
                "fit_keys": [r["key"] for r in fit],
                "threshold": calibration["threshold"],
                "release_approved": False,
            }
        )
        + "\n"
    )
    (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
