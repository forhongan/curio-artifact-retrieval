"""Frozen pilot: existing reviewed labels, grouped fit/calibration/test separation.

The existing labels were AI reviewed with museum records. This is an exploratory
comparison, NOT a new independently human-labelled test or a deployment certificate.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from curio.retrieve.joint_vlm import LocalVQA

ROOT = Path(__file__).resolve().parents[1]
QIDS = [
    "v5_porcelain",
    "v5_bottle",
    "v5_bowl",
    "v5_blue_white",
    "v5_green_vase",
    "v5_red_vase",
    "v5_dragon",
    "v5_lotus",
]


def readl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics(rows, accepted):
    known = np.array([r["grade"] >= 0 for r in rows])
    positive = np.array([r["grade"] == 3 for r in rows])
    accepted = np.asarray(accepted, bool)
    tp, fp, fn = (
        int((accepted & positive).sum()),
        int((accepted & known & ~positive).sum()),
        int((~accepted & positive).sum()),
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "unknown_accepted": int((accepted & ~known).sum()),
    }


def main():
    out = ROOT / "artifacts/joint_vlm_v9"
    out.mkdir(parents=True, exist_ok=True)
    base = ROOT / "eval/relevance_v5"
    freeze = json.loads((base / "freeze.json").read_text())
    for name, expected in freeze["hashes"].items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == expected
    queries = {r["qid"]: r for r in readl(base / "queries.jsonl")}
    items = {r["item_id"]: r for r in readl(ROOT / "data/snapshots/collection_v2.jsonl")}
    core = readl(base / "items.jsonl")
    # Split dev by object group, not by query-image row or language.
    groups = sorted(
        {r["object_group"] for r in core if r["split"] == "dev"},
        key=lambda s: hashlib.sha256(("joint-v9/" + s).encode()).hexdigest(),
    )
    fit_groups = set(groups[::2])
    partitions = {
        r["item_id"]: "test"
        if r["split"] == "test"
        else "fit"
        if r["object_group"] in fit_groups
        else "calibration"
        for r in core
    }
    cases = [
        dict(r, partition=partitions[r["item_id"]])
        for r in readl(base / "qrels.jsonl")
        if r["qid"] in QIDS
    ]
    protocol = {
        "qids": QIDS,
        "gallery": 4692,
        "core_images": 200,
        "rows": len(cases),
        "partitions": partitions,
        "label_source": "Existing v5 root-AI review with museum records; not independent human labels",
        "selection": "8 preselected varied intents; no model-score based sample selection",
        "calibration_goal": "empirical precision >= .90, recall >= .50, at least 5 true positives; not a statistical guarantee",
        "failure_policy": "if no threshold meets goal, retain .5 for experiment and mark goal unmet; never promote default",
        "freeze_sha256": hashlib.sha256((base / "freeze.json").read_bytes()).hexdigest(),
    }
    protocol_path = out / "pilot_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Pilot protocol changed; use a new experiment directory")
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    runtime = LocalVQA(ROOT)
    results, started = [], time.monotonic()
    try:
        plans = {qid: runtime.plan(queries[qid]["text_en"]) for qid in QIDS}
        for qid in QIDS:
            assert runtime.plan(queries[qid]["text_zh"]).key == plans[qid].key, qid
        with (out / "pilot_scores.jsonl").open("w") as stream:
            for index, case in enumerate(cases):
                score = runtime.score(items[case["item_id"]]["image_path"], plans[case["qid"]])
                result = {**case, **score}
                results.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                if (index + 1) % 40 == 0:
                    print(
                        f"Scored {index + 1}/{len(cases)}, elapsed {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
        fit = [
            r
            for r in results
            if r["partition"] == "fit" and r["grade"] >= 0 and r["reason"] != "color_not_observable"
        ]
        model = LogisticRegression(C=1.0).fit(
            np.array([[r["yes_no_logit"]] for r in fit]), [r["grade"] == 3 for r in fit]
        )

        def probability(rows):
            raw = np.array(
                [[r["yes_no_logit"] if r["yes_no_logit"] is not None else -50] for r in rows]
            )
            p = model.predict_proba(raw)[:, 1]
            p[[r["reason"] == "color_not_observable" for r in rows]] = 0
            return p

        calibration = [r for r in results if r["partition"] == "calibration"]
        p = probability(calibration)
        candidates = []
        for t in np.unique(p[p > 0]):
            m = metrics(calibration, p >= t)
            if m["tp"] >= 5 and m["precision"] >= 0.9 and m["recall"] >= 0.5:
                candidates.append((m["recall"], m["precision"], float(t)))
        threshold = max(candidates)[2] if candidates else 0.5
        test = [r for r in results if r["partition"] == "test"]
        v7 = json.loads((ROOT / "artifacts/set_boundary_v7/selected_ids.json").read_text())
        old = {
            lang: metrics(test, [r["item_id"] in v7[f"conditions/{lang}"][r["qid"]] for r in test])
            for lang in ["zh", "en"]
        }
        report = {
            "protocol": protocol,
            "model": runtime.identity,
            "test": {
                "v7": old,
                "vqa_raw_half": metrics(test, [r["score"] >= 0.5 for r in test]),
                "vqa_calibrated": metrics(test, probability(test) >= threshold),
            },
            "threshold": threshold,
            "calibration_goal_met": bool(candidates),
            "calibration_metrics": metrics(calibration, p >= threshold),
            "total_seconds": time.monotonic() - started,
            "per_query": {},
        }
        for qid in QIDS:
            rows = [r for r in test if r["qid"] == qid]
            report["per_query"][qid] = metrics(rows, probability(rows) >= threshold)
        bundle = {
            "schema": "joint-calibration-v1",
            "release_approved": False,
            "model": runtime.identity,
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "threshold": threshold,
            "scope": "Exploratory global head fitted on existing AI/source-reviewed labels; not validated for unseen concepts",
            "query_keys": [plans[qid].key for qid in QIDS],
            "calibration_goal_met": bool(candidates),
        }
        (out / "calibration.json").write_text(json.dumps(bundle, indent=2) + "\n")
        (out / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["test"], indent=2), flush=True)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
