"""Replay a frozen base with explicit counterevidence over every gourd candidate."""

import hashlib
import json
from pathlib import Path

from curio.evaluate.targeted_boundary import policy_metrics
from curio.retrieve.factor_boundary import query_id
from curio.retrieve.joint_vlm import LocalVQA
from curio.retrieve.shape_counterevidence import PROBES, apply_counterevidence, counter_plan
from prepare_boundary_followup import sheet

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/boundary_shape_v14"
OUT = ROOT / "artifacts/shape_boundary_v14"


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    gallery = {r["item_id"]: r for r in readl(ROOT / "data/snapshots/collection_v2.jsonl")}
    base_path = ROOT / "artifacts/surface_boundary_v13/gallery/gourd.json"
    old = json.loads(base_path.read_text())
    # Include all 59 coarse candidates, including 18 previously rejected by v13.
    candidates = [r for r in old["decisions"] if r["reason"] != "prefilter_insufficient_support"]
    replay = [
        r
        for file in ["predictions.jsonl", "regression_predictions.jsonl"]
        for r in readl(BASE / file)
        if r["atom"] == "gourd" and r.get("split") != "train"
    ]
    candidate_ids = {r["item_id"] for r in candidates}
    targets = {r["item_id"] for r in candidates + replay}
    protocol = {
        "version": "v14b-gourd-counterevidence",
        "base": "Frozen v13 visual gourd head; do not use regressed v14 gourd/bowl heads.",
        "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "probes": PROBES,
        "policy": "Reject if either independent counter-question has yes_no_logit >= 0; no threshold search or item-ID exceptions.",
        "scope": "Adaptive old-case replay and full frozen candidate scan; v14 tests already inspected, so NOT independent new validation. Prefilter recall remains unknown.",
        "target_count": len(targets),
        "candidate_count": len(candidates),
        "release_approved": False,
        "default_changed": False,
    }
    protocol_path = BASE / "countergate_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Frozen countergate protocol changed")
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    runtime = LocalVQA(ROOT)
    evidence = {}
    try:
        with (OUT / "countergate_scores.jsonl").open("w") as stream:
            for n, ident in enumerate(sorted(targets)):
                evidence[ident] = {
                    k: runtime.score(gallery[ident]["image_path"], counter_plan(k)) for k in PROBES
                }
                apply_counterevidence(False, evidence[ident])  # Validate before completion.
                stream.write(json.dumps({"item_id": ident, "scores": evidence[ident]}) + "\n")
                stream.flush()
                if (n + 1) % 20 == 0 or n + 1 == len(targets):
                    print(f"Counterevidence {n + 1}/{len(targets)}", flush=True)
    finally:
        runtime.close()
    decisions = []
    for r in old["decisions"]:
        if r["item_id"] not in candidate_ids:
            decisions.append(r)
        else:
            decisions.append(
                {
                    **r,
                    **apply_counterevidence(r["yes"], evidence[r["item_id"]]),
                    "counterevidence": evidence[r["item_id"]],
                }
            )
    assert query_id("葫芦瓶") == query_id("double gourd vase") == "gourd"
    result = {
        "qid": "gourd",
        "queries": ["葫芦瓶", "double gourd vase"],
        "complete": True,
        "processed": len(decisions),
        "verified_candidates": len(candidates),
        "yes": sum(r["yes"] for r in decisions),
        "decisions": decisions,
        "experimental": True,
        "release_approved": False,
    }
    (OUT / "countergate_gallery.json").write_text(json.dumps(result) + "\n")
    reruns = []
    for r in replay:
        scope = r.get("scope", "v14_" + r["split"] + "_replay")
        reruns.append(
            {
                **r,
                **apply_counterevidence(r["baseline_yes"], evidence[r["item_id"]]),
                "scope": scope,
                "counterevidence": evidence[r["item_id"]],
            }
        )
    (BASE / "countergate_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reruns)
    )
    report = {
        "protocol": protocol,
        "gallery_yes": result["yes"],
        "baseline_gallery_yes": old["yes"],
        "rejected_previous_yes": [
            r["item_id"] for r in decisions if r.get("baseline_yes") and not r["yes"]
        ],
        "replay": {
            s: {
                "v13": policy_metrics(
                    [r for r in reruns if r["scope"] == s],
                    [r["baseline_yes"] for r in reruns if r["scope"] == s],
                ),
                "countergate": policy_metrics(
                    [r for r in reruns if r["scope"] == s],
                    [r["yes"] for r in reruns if r["scope"] == s],
                ),
            }
            for s in sorted({r["scope"] for r in reruns})
        },
        "release_approved": False,
        "default_changed": False,
    }
    (BASE / "countergate_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    accepted = [
        dict(gallery[r["item_id"]], caption="countergate Yes | review next")
        for r in decisions
        if r["yes"]
    ]
    for start in range(0, len(accepted), 16):
        sheet(
            accepted[start : start + 16],
            OUT / f"gourd_final_{start // 16 + 1}.png",
            "gourd | countergate accepted",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
