"""Offline full-gallery cascade; failed verification never produces a complete set."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from curio.evaluate.monotone_evidence import predict_monotone
from curio.represent.object_surface import extract_surface
from curio.retrieve.factor_boundary import FactorEvidence, QUERY_ATOMS, and_decision, query_id
from curio.retrieve.joint_vlm import LocalVQA
from prepare_boundary_followup import sheet
from run_monotone_boundary_v12 import features, guard

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "eval/boundary_surface_v13"
    out = ROOT / "artifacts/surface_boundary_v13/gallery"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/targeted_boundary_v10/support_followup"
    gallery = {
        r["item_id"]: r
        for r in map(
            json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
        )
    }
    names = {
        "red_bowl": ("红色碗", "a red bowl"),
        "blue_white": ("青花瓷", "blue and white porcelain"),
        "meiping": ("梅瓶", "meiping vase"),
        "gourd": ("葫芦瓶", "double gourd vase"),
    }
    protocol = {
        "version": "factor-surface-gallery-v13",
        "prefilter": "Frozen v10 support-head decisions over all 4692 images, then v13 verification only on its positives. Prefilter misses are not recovered; full-gallery recall is unknown.",
        "candidate_hashes": {
            q: hashlib.sha256((source / f"{q}_gallery.jsonl").read_bytes()).hexdigest()
            for q in names
        },
        "gallery_sha256": hashlib.sha256(
            (ROOT / "data/snapshots/collection_v2.jsonl").read_bytes()
        ).hexdigest(),
        "release_approved": False,
        "default_changed": False,
    }
    protocol_path = base / "gallery_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Gallery protocol changed")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    bundle = json.loads((ROOT / "artifacts/factor_boundary_v11/heads.json").read_text())
    heads = bundle["heads"]
    red_semantic = json.loads((ROOT / "artifacts/monotone_boundary_v12/heads.json").read_text())[
        "heads"
    ]["red"]
    red_surface = json.loads(
        (ROOT / "artifacts/surface_boundary_v13/red_surface_head.json").read_text()
    )
    thresholds = {a: h["threshold"] for a, h in heads.items()}
    thresholds["red"] = red_surface["threshold"]
    runtime = LocalVQA(ROOT)
    if runtime.identity != bundle["vlm_identity"]:
        raise ValueError("Model changed since component fitting")
    scorer = FactorEvidence(runtime)
    summary, samples = {}, []
    reviewed_groups = set()
    for directory, names_to_read in [
        (
            ROOT / "eval/boundary_targeted_v10",
            ["labels.jsonl", "gallery_diagnostic_labels.jsonl", "support_diagnostic_labels.jsonl"],
        ),
        (ROOT / "eval/boundary_factors_v11", ["labels.jsonl", "stress_labels.jsonl"]),
    ]:
        for name in names_to_read:
            reviewed_groups.update(
                r["object_group"]
                for r in map(json.loads, (directory / name).read_text().splitlines())
            )
    try:
        for qid, (zh, en) in names.items():
            assert query_id(zh) == query_id(en) == qid
            candidates = [
                json.loads(s) for s in (source / f"{qid}_gallery.jsonl").read_text().splitlines()
            ]
            if len(candidates) != len(gallery) or {r["item_id"] for r in candidates} != set(
                gallery
            ):
                raise ValueError("Incomplete candidate decisions")
            decisions, verified = [], 0
            for r in candidates:
                item = gallery[r["item_id"]]
                if not r["decision"]:
                    decisions.append(
                        {
                            "item_id": r["item_id"],
                            "yes": False,
                            "reason": "prefilter_insufficient_support",
                            "evidence": [],
                        }
                    )
                    continue
                values = {}
                for atom in QUERY_ATOMS[qid]:
                    s = scorer.score(item["image_path"], atom)
                    if atom == "red":
                        p = 0.0
                        if not guard(s, atom):
                            surface = extract_surface(ROOT / item["image_path"])
                            if surface["valid"]:
                                visual = float(
                                    expit(
                                        np.sqrt(surface["features"])
                                        @ np.array(red_surface["coefficient"])
                                        + red_surface["intercept"]
                                    )
                                )
                                p = min(
                                    visual, float(predict_monotone(red_semantic, [features(s)])[0])
                                )
                    else:
                        h = heads[atom]
                        x = (np.array(s["features"]) - h["mean"]) / h["scale"]
                        p = float(expit(x @ np.array(h["coefficient"]) + h["intercept"]))
                    values[atom] = {
                        "score": 0.0 if s["color_excluded"] else p,
                        "color_excluded": s["color_excluded"],
                    }
                d = and_decision(qid, values, thresholds)
                decisions.append(
                    {
                        "item_id": r["item_id"],
                        **d,
                        "reason": "supported" if d["yes"] else "component_rejected",
                    }
                )
                verified += 1
                if verified % 20 == 0:
                    print(qid, "verified", verified, flush=True)
            target = out / f"{qid}.json"
            result = {
                "qid": qid,
                "queries": [zh, en],
                "complete": True,
                "processed": len(decisions),
                "verified_candidates": verified,
                "yes": sum(r["yes"] for r in decisions),
                "decisions": decisions,
                "experimental": True,
                "release_approved": False,
            }
            target.write_text(json.dumps(result, ensure_ascii=False) + "\n")
            summary[qid] = {k: v for k, v in result.items() if k != "decisions"}
            accepted = [
                r
                for r in decisions
                if r["yes"] and gallery[r["item_id"]]["object_group"] not in reviewed_groups
            ]
            accepted.sort(
                key=lambda r: hashlib.sha256(
                    ("v13-gallery-audit/" + qid + gallery[r["item_id"]]["object_group"]).encode()
                ).hexdigest()
            )
            sample, seen = [], set()
            for r in accepted:
                item = gallery[r["item_id"]]
                if item["object_group"] in seen:
                    continue
                seen.add(item["object_group"])
                sample.append({**item, **r, "qid": qid, "caption": "v13 accepted | not reviewed"})
                if len(sample) == 12:
                    break
            if sample:
                sheet(sample, out / f"{qid}_audit.png", qid + " | v13 cascade")
            samples.extend(
                {
                    k: r[k]
                    for k in [
                        "item_id",
                        "qid",
                        "object_group",
                        "image_path",
                        "title",
                        "medium",
                        "source_url",
                    ]
                }
                for r in sample
            )
            print(qid, "complete:", result["yes"], "Yes /", len(decisions), flush=True)
    finally:
        runtime.close()
    (out / "audit_sample.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    (base / "gallery_summary.json").write_text(
        json.dumps({"protocol": protocol, "queries": summary}, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
