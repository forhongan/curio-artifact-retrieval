"""Fit on train, calibrate on dev, freeze heads, then report new and old cases separately."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from curio.evaluate.targeted_boundary import calibrate, policy_metrics, validate_partitions
from curio.represent.object_surface import extract_surface
from curio.retrieve.factor_boundary import FactorEvidence
from curio.retrieve.joint_vlm import LocalVQA
from curio.retrieve.shape_boundary import (
    VERSION,
    PROBES,
    GEOMETRY_NAMES,
    extra_plan,
    geometry_features,
    feature_vector,
    fit_shape,
    predict_shape,
    decide_shape,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/boundary_shape_v14"
OUT = ROOT / "artifacts/shape_boundary_v14"


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-only", action="store_true", help="Reuse frozen image evidence")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BASE / "review_manifest.json").read_text())
    for name, digest in manifest["hashes"].items():
        if sha(BASE / name) != digest:
            raise ValueError("Reviewed data changed")
    protocol = json.loads((BASE / "protocol.json").read_text())
    rows = readl(BASE / "components.jsonl")
    validate_partitions(rows)
    gallery = {r["item_id"]: r for r in readl(ROOT / "data/snapshots/collection_v2.jsonl")}
    legacy = []
    for file, scope in [
        ("predictions.jsonl", "old_test"),
        ("stress_predictions.jsonl", "old_failures"),
        ("gallery_audit_labels.jsonl", "old_gallery_audit"),
    ]:
        for r in readl(ROOT / "eval/boundary_surface_v13" / file):
            if r["qid"] == "blue_white" or (scope == "old_test" and r["split"] != "test"):
                continue
            legacy.append(
                {
                    **r,
                    "atom": "bowl" if r["qid"] == "red_bowl" else r["qid"],
                    "scope": scope,
                    "baseline_yes": r["yes"],
                }
            )
    unique = {(r["atom"], r["item_id"]): r for r in rows + legacy}
    runtime = LocalVQA(ROOT)
    old_scorer = FactorEvidence(runtime)
    heads13 = json.loads((ROOT / "artifacts/factor_boundary_v11/heads.json").read_text())
    if runtime.identity != heads13["vlm_identity"]:
        raise ValueError("VLM identity changed")
    frozen = {
        "version": VERSION,
        "code_sha256": sha(ROOT / "src/curio/retrieve/shape_boundary.py"),
        "surface_code_sha256": sha(ROOT / "src/curio/represent/object_surface.py"),
        "protocol_sha256": sha(BASE / "protocol.json"),
        "labels_sha256": sha(BASE / "components.jsonl"),
        "vlm_identity": runtime.identity,
    }
    identity_path = OUT / "run_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != frozen:
        raise ValueError("Scoring code or protocol changed during frozen experiment")
    identity_path.write_text(json.dumps(frozen, indent=2) + "\n")
    scores, surfaces = {}, {}
    if args.fit_only:
        cached = readl(OUT / "scores.jsonl")
        scores = {tuple(r["key"].split(":", 1)): r for r in cached}
        if len(scores) != len(cached) or set(scores) != set(unique):
            raise ValueError("Incomplete frozen shape evidence")
        for name in ["report.json", "predictions.jsonl", "regression_predictions.jsonl"]:
            if (BASE / name).exists() and not (OUT / ("initial_" + name)).exists():
                (OUT / ("initial_" + name)).write_bytes((BASE / name).read_bytes())
    try:
        with (OUT / ("reuse.log" if args.fit_only else "scores.jsonl")).open("w") as stream:
            for n, ((atom, ident), r) in enumerate(unique.items()):
                if args.fit_only:
                    continue
                item = gallery[ident]
                if sha(ROOT / item["image_path"]) != item["image_sha256"]:
                    raise ValueError("Image hash changed")
                old = old_scorer.score(item["image_path"], atom)
                extra = runtime.score(item["image_path"], extra_plan(atom))
                if ident not in surfaces:
                    s = extract_surface(ROOT / item["image_path"])
                    surfaces[ident] = geometry_features(s["mask"], s["valid"])
                vector = feature_vector(old, extra, surfaces[ident])
                h = heads13["heads"][atom]
                baseline = float(
                    expit(
                        ((np.array(old["features"]) - h["mean"]) / h["scale"])
                        @ np.array(h["coefficient"])
                        + h["intercept"]
                    )
                )
                score = {
                    "key": atom + ":" + ident,
                    "features": vector,
                    "extra": extra,
                    "old": old,
                    "baseline_score": baseline,
                    "baseline_yes": baseline >= h["threshold"],
                }
                scores[(atom, ident)] = score
                stream.write(json.dumps(score) + "\n")
                stream.flush()
                if (n + 1) % 20 == 0 or n + 1 == len(unique):
                    print(f"Shape evidence {n + 1}/{len(unique)}", flush=True)
    finally:
        runtime.close()
    heads, calibration = {}, {}
    for atom in PROBES:
        subset = [r for r in rows if r["atom"] == atom]
        fit = [r for r in subset if r["split"] == "train" and r["grade"] >= 0]
        if any(r["source_split"] != "train" for r in fit):
            raise ValueError("Non-train source used for fitting")
        head = fit_shape(
            [scores[(atom, r["item_id"])]["features"] for r in fit], [r["decision"] for r in fit]
        )
        dev = [r for r in subset if r["split"] == "dev"]
        probs = predict_shape(head, [scores[(atom, r["item_id"])]["features"] for r in dev])
        # The primary visual path must not use candidate catalogue text during
        # calibration. Catalogue veto is a separate diagnostic at this point.
        for i, r in enumerate(dev):
            if atom == "gourd" and scores[(atom, r["item_id"])]["extra"]["yes_no_logit"] <= 0:
                probs[i] = 0
        cal = calibrate(dev, probs)
        heads[atom] = {
            **head,
            "threshold": cal["threshold"],
            "fit_keys": [r["key"] for r in fit],
            "release_approved": False,
        }
        calibration[atom] = {
            **cal,
            "dev_keys": [r["key"] for r in dev],
            "reused_old_dev": sum(r["provenance"] == "reused_v11_gourd_dev" for r in dev),
        }
    # Heads are frozen before test scoring/report inspection; no test optimization.
    (OUT / "heads.json").write_text(
        json.dumps(
            {
                **frozen,
                "heads": heads,
                "probes": PROBES,
                "geometry_names": GEOMETRY_NAMES,
                "release_approved": False,
            }
        )
        + "\n"
    )
    predictions = []
    for r in rows:
        if r["provenance"] != "fresh_v14":
            continue
        s = scores[(r["atom"], r["item_id"])]
        d = decide_shape(
            r["atom"], heads[r["atom"]], s["features"], s["extra"], gallery[r["item_id"]]
        )
        predictions.append(
            {
                **r,
                **d,
                "label_reason": r["reason"],
                "baseline_yes": s["baseline_yes"],
                "baseline_score": s["baseline_score"],
            }
        )
    regression = []
    for r in legacy:
        s = scores[(r["atom"], r["item_id"])]
        atom = r["atom"]
        d = decide_shape(atom, heads[atom], s["features"], s["extra"], gallery[r["item_id"]])
        unchanged_color = all(e["yes"] for e in r["evidence"] if e["atom"] != atom)
        regression.append(
            {
                **r,
                **d,
                "yes": d["yes"] and unchanged_color,
                "visual_yes": d["visual_yes"] and unchanged_color,
                "unchanged_color_passed": unchanged_color,
            }
        )
    for name, data in [("predictions", predictions), ("regression_predictions", regression)]:
        (BASE / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        )

    def metrics(data):
        return {
            m: policy_metrics(data, [r[k] for r in data])
            for m, k in [
                ("v13_baseline", "baseline_yes"),
                ("v14_visual_only", "visual_yes"),
                ("v14_catalogue_veto", "yes"),
            ]
        }

    test = [r for r in predictions if r["split"] == "test"]
    report = {
        "protocol": protocol,
        "primary_path": "Visual thresholds do not use catalogue veto; catalogue-assisted output is an auxiliary diagnostic, not the production image-only algorithm.",
        "calibration": calibration,
        "test": metrics(test),
        "per_atom_test": {a: metrics([r for r in test if r["atom"] == a]) for a in PROBES},
        "regression": {
            scope: {
                q: metrics([r for r in regression if r["scope"] == scope and r["qid"] == q])
                for q in ["red_bowl", "meiping", "gourd"]
            }
            for scope in ["old_test", "old_failures", "old_gallery_audit"]
        },
        "release_approved": False,
        "default_changed": False,
    }
    (BASE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"test": report["test"], "per_atom_test": report["per_atom_test"]}, indent=2))


if __name__ == "__main__":
    main()
