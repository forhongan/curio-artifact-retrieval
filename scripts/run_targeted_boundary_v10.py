"""Fit visual concept heads and compare VQA on frozen, targeted AI labels."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from curio.evaluate.targeted_boundary import calibrate, policy_metrics, validate_partitions
from curio.retrieve.joint_query import compile_known
from curio.retrieve.joint_vlm import LocalVQA

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "eval/boundary_targeted_v10"
    out = ROOT / "artifacts/targeted_boundary_v10"
    out.mkdir(parents=True, exist_ok=True)
    labels_path = base / "labels.jsonl"
    rows = [json.loads(s) for s in labels_path.read_text().splitlines()]
    manifest = json.loads((base / "review_manifest.json").read_text())
    protocol = json.loads((base / "protocol.json").read_text())
    for path, expected in [
        (labels_path, manifest["labels_sha256"]),
        (base / "candidates.jsonl", protocol["candidate_sha256"]),
        (ROOT / "data/snapshots/collection_v2.jsonl", protocol["gallery_sha256"]),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input changed: " + str(path))
    validate_partitions(rows)
    ids = json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text())
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    index = {ident: i for i, ident in enumerate(ids)}
    x = vectors[[index[r["item_id"]] for r in rows]]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    runtime = LocalVQA(ROOT)
    scores = []
    try:
        with (out / "scores.jsonl").open("w") as stream:
            for n, r in enumerate(rows):
                zh, en = compile_known(r["query_zh"]), compile_known(r["query_en"])
                if zh is None or en is None or zh.key != en.key:
                    raise ValueError("Bilingual concept mismatch: " + r["qid"])
                if (
                    hashlib.sha256((ROOT / r["image_path"]).read_bytes()).hexdigest()
                    != r["image_sha256"]
                ):
                    raise ValueError("Reviewed image changed")
                score = runtime.score(r["image_path"], en)
                scores.append(score)
                stream.write(json.dumps({"key": r["key"], **score}) + "\n")
                stream.flush()
                if (n + 1) % 20 == 0 or n + 1 == len(rows):
                    print(f"Scored {n + 1}/{len(rows)}", flush=True)
    finally:
        runtime.close()
    raw = np.array([s["score"] for s in scores])
    observable = np.array([s["reason"] != "color_not_observable" for s in scores])
    visual = np.zeros(len(rows))
    heads = {}
    for qid in protocol["queries"]:
        mask = np.array([r["qid"] == qid for r in rows])
        fit = mask & np.array([r["split"] == "train" and r["grade"] >= 0 for r in rows])
        y = np.array([r["decision"] for r in rows])
        if len(set(y[fit])) != 2:
            raise ValueError("Insufficient training classes for " + qid)
        head = LogisticRegression(C=10, class_weight="balanced", max_iter=1000).fit(x[fit], y[fit])
        visual[mask] = head.predict_proba(x[mask])[:, 1]
        heads[qid] = {
            "coefficient": head.coef_[0].tolist(),
            "intercept": float(head.intercept_[0]),
            "fit_keys": [r["key"] for r, use in zip(rows, fit) if use],
        }
    visual[~observable] = 0
    all_scores = {"raw_vqa": raw, "visual_head": visual, "equal_fusion": (raw + visual) / 2}
    report = {
        "protocol": protocol,
        "labels_sha256": manifest["labels_sha256"],
        "model": runtime.identity,
        "rows": len(rows),
        "images": len({r["item_id"] for r in rows}),
        "label_counts": dict(Counter(r["reason"] for r in rows)),
        "bilingual_same_plan": True,
        "release_approved": False,
        "methods": {},
    }
    predictions = []
    testmask = np.array([r["split"] == "test" for r in rows])
    testrows = [r for r in rows if r["split"] == "test"]
    for method, p in all_scores.items():
        method_report = {"per_query": {}}
        accepted = np.zeros(len(rows), dtype=bool)
        for qid in protocol["queries"]:
            dev = np.array([r["qid"] == qid and r["split"] == "dev" for r in rows])
            test = np.array([r["qid"] == qid and r["split"] == "test" for r in rows])
            calibration = calibrate([r for r, use in zip(rows, dev) if use], p[dev])
            mask = np.array([r["qid"] == qid for r in rows])
            accepted[mask] = p[mask] >= calibration["threshold"]
            method_report["per_query"][qid] = {
                "calibration": calibration,
                "test": policy_metrics([r for r, use in zip(rows, test) if use], accepted[test]),
                "test_at_0_5": policy_metrics(
                    [r for r, use in zip(rows, test) if use], p[test] >= 0.5
                ),
            }
        method_report["test"] = policy_metrics(testrows, accepted[testmask])
        method_report["test_at_0_5"] = policy_metrics(testrows, p[testmask] >= 0.5)
        report["methods"][method] = method_report
        predictions.extend(
            {
                "key": r["key"],
                "split": r["split"],
                "method": method,
                "expected": r["decision"],
                "score": float(p[i]),
                "accepted": bool(accepted[i]),
            }
            for i, r in enumerate(rows)
        )
    bundle = {
        "release_approved": False,
        "protocol": protocol,
        "heads": heads,
        "labels_sha256": manifest["labels_sha256"],
        "index_manifest_sha256": hashlib.sha256(
            (ROOT / "artifacts/siglip_adapted/manifest.json").read_bytes()
        ).hexdigest(),
    }
    (out / "visual_heads.json").write_text(json.dumps(bundle) + "\n")
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions))
    (base / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({m: r["test"] for m, r in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
