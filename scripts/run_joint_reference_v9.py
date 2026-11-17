"""Fit-only positive/hard-negative exemplars; evaluate on the same frozen pilot.

This is an adaptive research comparison on already inspected test data, not a
fresh independent test. No test labels or images become reference exemplars.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from curio.retrieve.joint_vlm import LocalVQA
from run_joint_vlm_v9 import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "artifacts/joint_vlm_v9"
    source = [json.loads(s) for s in (out / "pilot_scores.jsonl").read_text().splitlines()]
    queries = {
        r["qid"]: r
        for r in map(
            json.loads, (ROOT / "eval/relevance_v5/queries.jsonl").read_text().splitlines()
        )
    }
    items = {
        r["item_id"]: r
        for r in map(
            json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
        )
    }
    qids = json.loads((out / "pilot_protocol.json").read_text())["qids"]
    scorer = LocalVQA(ROOT)
    bank = {}
    plans = {}
    started = time.monotonic()
    results = []
    try:
        for qid in qids:
            plan = scorer.plan(queries[qid]["text_en"])
            plans[qid] = plan
            fit = [
                r
                for r in source
                if r["qid"] == qid
                and r["partition"] == "fit"
                and r["reason"] != "color_not_observable"
            ]
            pos = sorted(
                [r for r in fit if r["grade"] == 3], key=lambda r: (-r["score"], r["item_id"])
            )
            neg = sorted(
                [r for r in fit if 0 <= r["grade"] < 3], key=lambda r: (-r["score"], r["item_id"])
            )
            if pos and neg:
                bank[plan.key] = [
                    {
                        "item_id": r["item_id"],
                        "image_path": items[r["item_id"]]["image_path"],
                        "yes": yes,
                    }
                    for r, yes in [(pos[0], True), (neg[0], False)]
                ]
        bank_value = {
            "selection": "highest-score positive and false-positive from fit object groups only",
            "label_source": "existing AI/source-reviewed v5 labels",
            "references": bank,
        }
        bank_path = out / "reference_bank.json"
        if bank_path.exists() and json.loads(bank_path.read_text()) != bank_value:
            raise RuntimeError("Reference selection changed; use a new experiment")
        bank_path.write_text(json.dumps(bank_value, indent=2) + "\n")
        with (out / "reference_scores.jsonl").open("w") as stream:
            for i, case in enumerate(source):
                item = items[case["item_id"]]
                plan = plans[case["qid"]]

                def same(ref):
                    other = items[ref["item_id"]]
                    return ref["item_id"] == item["item_id"] or any(
                        item.get(k) and item[k] == other.get(k)
                        for k in ["object_group", "image_sha256"]
                    )

                refs = [r for r in bank.get(plan.key, []) if not same(r)]
                result = {
                    **case,
                    **scorer.score(item["image_path"], plan, references=refs),
                    "reference_ids": [r["item_id"] for r in refs],
                }
                results.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                if (i + 1) % 40 == 0:
                    print(
                        f"References {i + 1}/{len(source)}, {time.monotonic() - started:.1f}s",
                        flush=True,
                    )
        fit = [
            r
            for r in results
            if r["partition"] == "fit" and r["grade"] >= 0 and r["reason"] != "color_not_observable"
        ]
        model = LogisticRegression(C=1).fit(
            [[r["yes_no_logit"]] for r in fit], [r["grade"] == 3 for r in fit]
        )

        def probs(rows):
            p = model.predict_proba(
                [[r["yes_no_logit"] if r["yes_no_logit"] is not None else -50] for r in rows]
            )[:, 1]
            p[[r["reason"] == "color_not_observable" for r in rows]] = 0
            return p

        cal = [r for r in results if r["partition"] == "calibration"]
        p = probs(cal)
        choices = []
        for t in np.unique(p[p > 0]):
            m = metrics(cal, p >= t)
            if m["tp"] >= 5 and m["precision"] >= 0.9 and m["recall"] >= 0.5:
                choices.append((m["recall"], m["precision"], float(t)))
        threshold = max(choices)[2] if choices else 0.5
        test = [r for r in results if r["partition"] == "test"]
        report = {
            "status": "exploratory reference adaptation, not independent human validation",
            "model": scorer.identity,
            "calibration_goal_met": bool(choices),
            "threshold": threshold,
            "calibration": metrics(cal, p >= threshold),
            "test_raw": metrics(test, [r["score"] >= 0.5 for r in test]),
            "test_calibrated": metrics(test, probs(test) >= threshold),
            "per_query": {
                q: metrics(
                    [r for r in test if r["qid"] == q],
                    probs([r for r in test if r["qid"] == q]) >= threshold,
                )
                for q in qids
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        bundle = {
            "schema": "joint-calibration-v1",
            "release_approved": False,
            "model": scorer.identity,
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
            "threshold": threshold,
            "query_keys": [p.key for p in plans.values()],
            "calibration_goal_met": bool(choices),
            "reference_bank_sha256": hashlib.sha256(bank_path.read_bytes()).hexdigest(),
            "scope": "Existing AI/source-reviewed labels; fit-only exemplars; not validated for new concepts",
        }
        (out / "reference_calibration.json").write_text(json.dumps(bundle, indent=2) + "\n")
        (out / "reference_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
