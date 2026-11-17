"""One predeclared hybrid head: VQA logit plus frozen contrastive features.

Exploratory follow-up after inspecting earlier tests; no hyperparameter search.
Fit/calibration object groups remain unchanged and test images never fit the head.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_set_boundary_v7 import load_data, text_evidence
from run_joint_vlm_v9 import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "artifacts/joint_vlm_v9"
    protocol = {
        "version": "joint-fusion-research-v1",
        "features": "raw VQA logit clipped to [-30,30] + 9 existing English contrastive features",
        "fit": "existing pilot fit partition only; StandardScaler and LogisticRegression(C=0.1)",
        "calibration": "existing separate object groups; precision>=.9, recall>=.5, TP>=5; otherwise .5 and failed goal",
        "test_scope": "adaptive research on already inspected AI/source-reviewed test, not independent human validation",
    }
    pp = out / "fusion_protocol.json"
    if pp.exists() and json.loads(pp.read_text()) != protocol:
        raise RuntimeError("Protocol changed; use a new experiment")
    pp.write_text(json.dumps(protocol, indent=2) + "\n")
    queries, _, ids, _, _, _, _, frozen = load_data(ROOT)
    features, _ = text_evidence(ROOT, ROOT / "artifacts/set_boundary_v7", queries, ids, frozen)
    imap = {ident: i for i, ident in enumerate(ids)}
    qmap = {q["qid"]: i for i, q in enumerate(queries)}
    rows = [json.loads(s) for s in (out / "pilot_scores.jsonl").read_text().splitlines()]
    x = np.array(
        [
            [
                np.clip(r["yes_no_logit"] if r["yes_no_logit"] is not None else -30, -30, 30),
                *features["en"][imap[r["item_id"]], qmap[r["qid"]]].tolist(),
            ]
            for r in rows
        ]
    )
    observed = np.array([r["reason"] != "color_not_observable" for r in rows])
    fit = np.array([r["partition"] == "fit" and r["grade"] >= 0 for r in rows]) & observed
    y = np.array([r["grade"] == 3 for r in rows])
    scaler = StandardScaler().fit(x[fit])
    model = LogisticRegression(C=0.1, max_iter=1000).fit(scaler.transform(x[fit]), y[fit])
    p = model.predict_proba(scaler.transform(x))[:, 1]
    p[~observed] = 0
    cal = np.array([r["partition"] == "calibration" for r in rows])
    test = np.array([r["partition"] == "test" for r in rows])
    calibration = [r for r in rows if r["partition"] == "calibration"]
    evaluation = [r for r in rows if r["partition"] == "test"]
    choices = []
    for t in np.unique(p[cal & (p > 0)]):
        m = metrics(calibration, p[cal] >= t)
        if m["tp"] >= 5 and m["precision"] >= 0.9 and m["recall"] >= 0.5:
            choices.append((m["recall"], m["precision"], float(t)))
    threshold = max(choices)[2] if choices else 0.5
    report = {
        "protocol": protocol,
        "calibration_goal_met": bool(choices),
        "threshold": threshold,
        "calibration": metrics(calibration, p[cal] >= threshold),
        "test": metrics(evaluation, p[test] >= threshold),
        "per_query": {
            q: metrics(
                [r for r in evaluation if r["qid"] == q],
                p[test & np.array([r["qid"] == q for r in rows])] >= threshold,
            )
            for q in sorted({r["qid"] for r in rows})
        },
    }
    bundle = {
        "release_approved": False,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "threshold": threshold,
        "source_scores_sha256": hashlib.sha256(
            (out / "pilot_scores.jsonl").read_bytes()
        ).hexdigest(),
    }
    (out / "fusion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "fusion_head.json").write_text(json.dumps(bundle, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
