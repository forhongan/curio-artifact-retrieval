"""Adaptive follow-up: reviewed background negatives and positive support gate."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from curio.evaluate.targeted_boundary import (
    calibrate,
    policy_metrics,
    positive_support_threshold,
    validate_partitions,
)
from prepare_boundary_followup import sheet

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "eval/boundary_targeted_v10"
    out = ROOT / "artifacts/targeted_boundary_v10/support_followup"
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((base / "followup_protocol.json").read_text())
    paths = [
        (base / "labels.jsonl", "parent_labels_sha256"),
        (base / "gallery_diagnostic_labels.jsonl", "diagnostic_labels_sha256"),
    ]
    for path, key in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            raise ValueError("Frozen follow-up labels changed")
    original, diagnostic = [
        [json.loads(s) for s in path.read_text().splitlines()] for path, _ in paths
    ]
    validate_partitions(original)
    held_groups = {r["object_group"] for r in original if r["split"] != "train"}
    held_hashes = {r["image_sha256"] for r in original if r["split"] != "train"}
    fitrows, seen = [], set()
    for r in original + diagnostic:
        pair = (r["qid"], r["object_group"])
        if r["split"] != "train" or r["grade"] < 0 or pair in seen:
            continue
        if r["object_group"] in held_groups or r["image_sha256"] in held_hashes:
            raise ValueError("Held-out object cannot enter training")
        fitrows.append(r)
        seen.add(pair)
    ids = json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text())
    index = {ident: i for i, ident in enumerate(ids)}
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    gallery = {
        r["item_id"]: r
        for r in map(
            json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
        )
    }
    previous = json.loads(
        (ROOT / "artifacts/joint_vlm_v9/full_gallery_blue_white.json").read_text()
    )
    if not previous["complete"] or {r["item_id"] for r in previous["decisions"]} != set(ids):
        raise ValueError("Incomplete cached gallery color check")
    gray = {r["item_id"] for r in previous["decisions"] if r["reason"] == "color_not_observable"}
    seen_groups = {r["object_group"] for r in original + diagnostic}
    report = {"protocol": protocol, "release_approved": False, "per_query": {}}
    models, samples, testrows, decisions = {}, [], [], []
    for qid in sorted({r["qid"] for r in original}):
        fit = [r for r in fitrows if r["qid"] == qid]
        x = vectors[[index[r["item_id"]] for r in fit]]
        y = np.array([r["decision"] for r in fit])
        model = LogisticRegression(C=10, class_weight="balanced", max_iter=1000).fit(x, y)
        support_threshold = positive_support_threshold(x[y])
        support = (vectors @ x[y].T).max(axis=1)
        p = model.predict_proba(vectors)[:, 1]
        supported = support >= support_threshold
        p[~supported] = 0
        if qid in ("red_bowl", "blue_white"):
            p[[index[ident] for ident in gray]] = 0
        dev = [r for r in original if r["qid"] == qid and r["split"] == "dev"]
        calibration = calibrate(dev, [p[index[r["item_id"]]] for r in dev])
        accepted = p >= calibration["threshold"]
        test = [r for r in original if r["qid"] == qid and r["split"] == "test"]
        test_yes = [bool(accepted[index[r["item_id"]]]) for r in test]
        testrows += test
        decisions += test_yes
        report["per_query"][qid] = {
            "train": len(fit),
            "train_positive": int(y.sum()),
            "support_threshold": support_threshold,
            "calibration": calibration,
            "test": policy_metrics(test, test_yes),
            "full_gallery": {
                "total": len(ids),
                "yes": int(accepted.sum()),
                "no": int((~accepted).sum()),
            },
        }
        models[qid] = {
            "coefficient": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "positive_item_ids": [r["item_id"] for r in fit if r["decision"]],
            "fit_keys": [r["key"] for r in fit],
            "support_threshold": support_threshold,
            "threshold": calibration["threshold"],
        }
        all_rows = [
            {
                "item_id": ident,
                "score": float(p[i]),
                "support": float(support[i]),
                "decision": bool(accepted[i]),
            }
            for i, ident in enumerate(ids)
        ]
        (out / f"{qid}_gallery.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
        candidates = [
            r
            for r in all_rows
            if r["decision"] and gallery[r["item_id"]]["object_group"] not in seen_groups
        ]
        candidates.sort(
            key=lambda r: hashlib.sha256(
                ("support-audit-v10/" + qid + gallery[r["item_id"]]["object_group"]).encode()
            ).hexdigest()
        )
        sample, groups = [], set()
        for r in candidates:
            g = gallery[r["item_id"]]
            if g["object_group"] in groups:
                continue
            groups.add(g["object_group"])
            sample.append({**g, **r, "qid": qid, "caption": f"score={r['score']:.3f}"})
            if len(sample) == 16:
                break
        if sample:
            sheet(sample, out / f"{qid}_audit.png", qid + " | background + support")
        samples.extend(
            {
                k: r[k]
                for k in (
                    "qid",
                    "item_id",
                    "object_group",
                    "image_path",
                    "title",
                    "medium",
                    "source_url",
                    "score",
                )
            }
            for r in sample
        )
    report["test"] = policy_metrics(testrows, decisions)
    (out / "models.json").write_text(
        json.dumps({"release_approved": False, "models": models, "protocol": protocol}) + "\n"
    )
    (out / "audit_sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    (base / "followup_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
