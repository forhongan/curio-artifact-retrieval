"""Freeze fresh object groups before factor-model scoring or label inspection."""

import hashlib
import json
from pathlib import Path
import re

from curio.retrieve.strict_boundary import pixel_evidence
from prepare_boundary_followup import sheet
from prepare_targeted_boundary_v10 import likely_positive, eligible_distractor, QUERIES

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/boundary_factors_v11"


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    out = ROOT / "artifacts/factor_boundary_v11"
    out.mkdir(parents=True, exist_ok=True)
    gallery = [
        json.loads(s)
        for s in (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
    ]
    used_ids = set()
    for name in [
        "relevance_v5/items.jsonl",
        "boundary_assisted_v10/labels.jsonl",
        "boundary_targeted_v10/labels.jsonl",
        "boundary_targeted_v10/gallery_diagnostic_labels.jsonl",
        "boundary_targeted_v10/support_diagnostic_labels.jsonl",
    ]:
        used_ids.update(
            r["item_id"] for r in map(json.loads, (ROOT / "eval" / name).read_text().splitlines())
        )
    used_ids.update(
        r["item_id"]
        for r in json.loads((ROOT / "artifacts/joint_vlm_v9/specialty_report.json").read_text())[
            "rows"
        ]
    )
    used_groups = {r["object_group"] for r in gallery if r["item_id"] in used_ids}
    used_hashes = {r["image_sha256"] for r in gallery if r["item_id"] in used_ids}
    fresh = []
    for r in sorted(gallery, key=lambda r: r["item_id"]):
        if r["object_group"] in used_groups or r["image_sha256"] in used_hashes:
            continue
        fresh.append(r)
        used_groups.add(r["object_group"])
        used_hashes.add(r["image_sha256"])
    # Rare red bowls: look for visible red pixels to discover candidates whose
    # catalogue medium does not mention red. This detector does not label them.
    red_scores = {}
    for r in fresh:
        if re.search(r"\bbowl\b|\bcup\b", r["title"], re.I):
            evidence = pixel_evidence(ROOT / r["image_path"])
            red_scores[r["item_id"]] = evidence["color_fractions"].get("red", 0)
    selected, selected_groups = [], set()
    for qid, (zh, en) in QUERIES.items():
        available = [r for r in fresh if r["object_group"] not in selected_groups]
        if qid == "red_bowl":
            pos = sorted(
                [r for r in available if r["item_id"] in red_scores],
                key=lambda r: (-red_scores[r["item_id"]], r["item_id"]),
            )[:12]
        else:
            pos = sorted(
                [r for r in available if likely_positive(r, qid)],
                key=lambda r: hashlib.sha256(
                    ("factor-positive-v11/" + r["item_id"]).encode()
                ).hexdigest(),
            )[:12]
        pos_ids = {r["item_id"] for r in pos}
        neg = sorted(
            [r for r in available if r["item_id"] not in pos_ids and eligible_distractor(r, qid)],
            key=lambda r: hashlib.sha256(
                ("factor-negative-v11/" + qid + r["item_id"]).encode()
            ).hexdigest(),
        )[:12]
        for selection, pool in [("positive_candidate", pos), ("nearby_category_candidate", neg)]:
            for n, r in enumerate(pool):
                selected_groups.add(r["object_group"])
                selected.append(
                    {
                        "key": qid + ":" + r["item_id"],
                        "qid": qid,
                        "query_zh": zh,
                        "query_en": en,
                        "selection": selection,
                        "source_split": r["split"],
                        "partition": "dev" if n % 2 == 0 else "test",
                        **{
                            k: r[k]
                            for k in [
                                "item_id",
                                "object_group",
                                "image_sha256",
                                "image_path",
                                "title",
                                "medium",
                                "source_url",
                            ]
                        },
                    }
                )
        subset = [r for r in selected if r["qid"] == qid]
        for offset in range(0, len(subset), 12):
            sheet(
                [
                    dict(r, caption=r["partition"] + " | NOT LABELED")
                    for r in subset[offset : offset + 12]
                ],
                out / f"{qid}_{offset // 12 + 1}.png",
                qid + " | new holdout",
            )
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    path = BASE / "candidates.jsonl"
    if path.exists() and path.read_text() != data:
        raise ValueError("Frozen candidates changed")
    path.write_text(data)
    protocol = {
        "version": "factor-boundary-v11",
        "candidates_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "queries": QUERIES,
        "fresh_rows": len(selected),
        "holdout": "Unreviewed object groups, disjoint from v5/v9/v10 reviewed groups and hashes. Source split retained; source-train images may be held out from the new head, but were not necessarily unseen by the old image encoder.",
        "training": "Only previously reviewed source-train objects; component labels assigned from root visual review before scoring. No new holdout group may enter fitting.",
        "algorithm": "Per-component positive and counter-description VQA logits plus frozen pixel evidence; StandardScaler and LogisticRegression(C=1, balanced). Red bowl requires both red-surface and bowl components. Other concepts each have their own component.",
        "thresholds": "Each component calibrated only on fresh dev: empirical precision>=.9, recall>=.5, TP>=2; fallback .5 marked failed. Queries use AND. No test-based feature or threshold changes.",
        "limitations": "Small targeted AI-reviewed diagnostic, not independent human gold or full-gallery recall. Binary No for insufficient visible support; no pending results.",
        "release_approved": False,
    }
    (BASE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(len(selected), "fresh pairs;", len(selected_groups), "objects")


if __name__ == "__main__":
    main()
