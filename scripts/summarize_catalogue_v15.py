"""Validate saved real runs and export a compact, auditable release receipt."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from curio.retrieve.catalogue_visual import CatalogueVisualScorer
from curio.retrieve.joint_pixels import color_observability
from curio.retrieve.joint_vlm import LocalVQA
from verify_catalogue_v15 import QUERIES

ROOT = Path(__file__).resolve().parents[1]


def main():
    gallery = list(
        map(json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines())
    )
    by_id = {r["item_id"]: r for r in gallery}
    service = SimpleNamespace(root=ROOT, items=gallery, by_id=by_id)
    scorer = CatalogueVisualScorer(service, LocalVQA(ROOT))
    identity = scorer.identity
    reports, decisions, summaries = {}, {}, {}
    directory = ROOT / "artifacts/catalogue_visual_v15/verification"
    for name in QUERIES:
        path = directory / f"{name}.json"
        report = json.loads(path.read_text())
        assert report["complete"] and report["model"] == identity, (name, "stale model/policy")
        assert report["bilingual_same_job"]
        assert len(report["decisions"]) == len(gallery)
        mapping = {r["item_id"]: r for r in report["decisions"]}
        assert set(mapping) == set(by_id)
        assert all(type(r["yes"]) is bool for r in mapping.values())
        yes = {r["item_id"] for r in mapping.values() if r["yes"]}
        assert yes == set(report["matched_item_ids"])
        assert report["matched_count"] + report["rejected_count"] == len(gallery)
        assert report["uncertain_count"] == 0
        assert report["plan"]["key"] == scorer.plan(QUERIES[name][0]).key
        excluded = []
        if report["plan"]["requires_color"]:
            excluded = [
                i
                for i in sorted(yes)
                if not color_observability(ROOT / by_id[i]["image_path"])["observable"]
            ]
            assert not excluded, (name, "unobservable accepted color", excluded)
        summaries[name] = {
            "queries": QUERIES[name],
            "job_id": report["job_id"],
            "yes": len(yes),
            "no": len(gallery) - len(yes),
            "binary_complete": True,
            "same_bilingual_job": True,
            "accepted_with_unobservable_color": len(excluded)
            if report["plan"]["requires_color"]
            else None,
            "decision_reasons": dict(Counter(r["reason"] for r in mapping.values())),
            "seconds": report["latency_ms"] / 1000,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        reports[name], decisions[name] = report, mapping
    generic = set(reports["gourd"]["matched_item_ids"])
    assert set(reports["gourd_double"]["matched_item_ids"]) <= generic
    assert set(reports["gourd_triple"]["matched_item_ids"]) <= generic
    assert set(reports["green_narrow"]["matched_item_ids"]) <= set(
        reports["green_vase"]["matched_item_ids"]
    )
    expectations = {
        "blue_white": {"met_42303": False},
        "green_narrow": {
            "met_46546": False,
            "met_46957": False,
            "met_48346": False,
            "met_46989": False,
            "met_46953": False,
            "met_42228": True,
        },
        "red_bowl": {
            "cma_120216": False,
            "met_46998": False,
            "met_460692": True,
            "met_47312": True,
            "cma_154693": False,
            "met_460694": False,
            "met_49220": False,
            "met_49348": False,
            "met_49349": False,
            "met_76119": True,
        },
        "meiping": {"met_48254": True},
        "gourd": {
            "cma_101724": False,
            "cma_147392": False,
            "met_50037": True,
            "cma_97660": True,
            "met_42358": True,
            "cma_140244": False,
        },
        "gourd_double": {"met_42358": False},
        "gourd_triple": {"met_42358": True},
    }
    regression = []
    for name, cases in expectations.items():
        for ident, expected in cases.items():
            actual = decisions[name][ident]["yes"]
            regression.append(
                {
                    "query": name,
                    "item_id": ident,
                    "expected": expected,
                    "actual": actual,
                    "passed": actual == expected,
                    "reason": decisions[name][ident]["reason"],
                }
            )
    # Precision guards are required. Known positive cases remain visible as
    # recovery probes: strict filtering is allowed to miss them, not to hide
    # the miss by changing an expected label after the run.
    assert all(r["passed"] for r in regression if not r["expected"]), regression
    assert all(summaries[q]["yes"] > 0 for q in QUERIES)
    for row in reports["green_narrow"]["results"]:
        evidence = next(
            e for e in row["condition_evidence"] if e["condition"] == "shape/narrow_mouth"
        )
        assert 0 < evidence["mouth_body_ratio"] <= evidence["ratio_limit"] == 0.35
    worker_sha = hashlib.sha256((ROOT / "scripts/joint_vlm_worker.py").read_bytes()).hexdigest()
    assert worker_sha == "4005f48b0c6c46d87716f644841e5989966c3cbc15119696b9368badb1bac8ab"
    result = {
        "version": "catalogue-visual-v15",
        "gallery_count": len(gallery),
        "model": identity,
        "queries": summaries,
        "gourd_subtypes_are_subsets": True,
        "narrow_mouth_is_subset_of_green_vases": True,
        "known_case_regressions": regression,
        "known_positive_misses": [r for r in regression if r["expected"] and not r["actual"]],
        "frozen_worker_unchanged": True,
        "quality_scope": "Engineering checks and previously seen AI-assisted cases; not independent accuracy or gallery recall.",
    }
    target = ROOT / "eval/boundary_catalogue_v15/verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {k: {c: r[c] for c in ["yes", "no", "seconds"]} for k, r in summaries.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
