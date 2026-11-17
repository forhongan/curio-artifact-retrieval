"""Mine candidates, never labels, for a targeted AI visual review.

Museum text identifies likely positives. Frozen image embeddings select similar
vessel distractors. Neither is accepted as ground truth without image review.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from prepare_boundary_followup import sheet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/boundary_targeted_v10"
QUERIES = {
    "red_bowl": ("红色碗", "a red bowl"),
    "blue_white": ("青花瓷", "blue and white porcelain"),
    "meiping": ("梅瓶", "meiping vase"),
    "gourd": ("葫芦瓶", "double gourd vase"),
}


def likely_positive(row, qid):
    title, medium = row["title"].lower(), row["medium"].lower()
    if qid == "red_bowl":
        return bool(
            re.search(r"\bbowl\b", title)
            and re.search(r"\bred\b|ox.blood|sang.de.boeuf|cinnabar", medium)
        )
    if qid == "blue_white":
        return (
            "porcelain" in medium
            and bool(
                re.search(
                    r"cobalt blue|underglaze blue|under.glaze.*blue|blue and white decoration",
                    medium,
                )
            )
            and not re.search(r"polychrome|overglaze|copper red|yellow glaze", medium)
        )
    if qid == "meiping":
        return "meiping" in title
    return (
        "gourd" in title and bool(re.search(r"vase|bottle|flask", title)) and "potter" not in title
    )


def eligible_distractor(row, qid):
    title = row["title"].lower()
    pattern = (
        r"bowl|dish|plate|vase"
        if qid == "red_bowl"
        else r"vase|bottle|flask|jar|bowl|dish|plate|cup"
    )
    if qid in ("meiping", "gourd"):
        pattern = r"vase|bottle|flask|jar"
    return bool(re.search(pattern, title)) and not likely_positive(row, qid)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pictures = ROOT / "artifacts/targeted_boundary_v10"
    pictures.mkdir(parents=True, exist_ok=True)
    gallery = [
        json.loads(s)
        for s in (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
    ]
    by_id = {r["item_id"]: r for r in gallery}
    previous = [
        json.loads(s) for s in (ROOT / "eval/relevance_v5/items.jsonl").read_text().splitlines()
    ]
    previous += json.loads((ROOT / "artifacts/set_boundary_followup/new_items.json").read_text())
    previous += [
        {"item_id": r["item_id"]}
        for r in json.loads((ROOT / "artifacts/joint_vlm_v9/specialty_report.json").read_text())[
            "rows"
        ]
    ]
    excluded_groups = {by_id[r["item_id"]]["object_group"] for r in previous}
    excluded_hashes = {by_id[r["item_id"]]["image_sha256"] for r in previous}
    # A single representative per physical object and exact image, globally.
    unique, seen_groups, seen_hashes = [], set(excluded_groups), set(excluded_hashes)
    for r in sorted(gallery, key=lambda r: r["item_id"]):
        if r["object_group"] in seen_groups or r["image_sha256"] in seen_hashes:
            continue
        unique.append(r)
        seen_groups.add(r["object_group"])
        seen_hashes.add(r["image_sha256"])
    ids = json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text())
    index = {ident: i for i, ident in enumerate(ids)}
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    selected = []
    for qid, (zh, en) in QUERIES.items():
        train_pool = [r for r in unique if r["split"] == "train" and likely_positive(r, qid)]
        centroid = vectors[[index[r["item_id"]] for r in train_pool]].mean(axis=0)
        for split in ("train", "dev", "test"):
            pool = [r for r in unique if r["split"] == split]
            positives = sorted(
                [r for r in pool if likely_positive(r, qid)],
                key=lambda r: hashlib.sha256(
                    ("targeted-v10/" + qid + r["item_id"]).encode()
                ).hexdigest(),
            )
            count = 10 if split == "train" else 4
            if qid == "red_bowl":
                count = len(positives)  # Rare concept: inspect every metadata candidate.
            negatives = sorted(
                [r for r in pool if eligible_distractor(r, qid)],
                key=lambda r: (-float(vectors[index[r["item_id"]]] @ centroid), r["item_id"]),
            )[: 10 if split == "train" else 6]
            subset = [(r, "catalogue_candidate") for r in positives[:count]] + [
                (r, "similar_vessel_candidate") for r in negatives
            ]
            for r, selection in subset:
                selected.append(
                    {
                        "key": qid + ":" + r["item_id"],
                        "qid": qid,
                        "query_zh": zh,
                        "query_en": en,
                        **{
                            k: r[k]
                            for k in (
                                "item_id",
                                "object_group",
                                "image_sha256",
                                "image_path",
                                "split",
                                "title",
                                "medium",
                                "source_url",
                            )
                        },
                        "selection": selection,
                    }
                )
        subset = [r for r in selected if r["qid"] == qid]
        for offset in range(0, len(subset), 20):
            sheet(
                [
                    dict(r, caption=r["split"] + " | NOT LABELED")
                    for r in subset[offset : offset + 20]
                ],
                pictures / f"{qid}_{offset // 20 + 1}.png",
                qid + " | candidate review",
            )
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    path = OUT / "candidates.jsonl"
    if path.exists() and path.read_text() != data:
        raise RuntimeError("Frozen candidates changed; use a new dataset version")
    path.write_text(data)
    protocol = {
        "schema": "curio-targeted-boundary-v10",
        "queries": QUERIES,
        "candidate_sha256": hashlib.sha256(data.encode()).hexdigest(),
        "gallery_sha256": hashlib.sha256(
            (ROOT / "data/snapshots/collection_v2.jsonl").read_bytes()
        ).hexdigest(),
        "selection": "catalogue candidates plus visually similar vessels; training-only candidate centroids; labels require root AI image inspection",
        "split_policy": "preserve source train/dev/test; exclude v5, v10 random review and v9 specialty objects and image hashes",
        "experiment": "per-concept L2 LogisticRegression(C=10, class_weight=balanced) on frozen normalized image embeddings; compare raw VQA and equal-weight mean of the two probabilities",
        "calibration": "dev only: precision>=0.9, recall>=0.5, at least 2 TP per concept; fallback threshold .5 with goal_met=false; never approve release from this small targeted sample",
        "limits": "AI labels, targeted nonrepresentative sample; not full-gallery precision or recall, no general free-query claim",
        "release_approved": False,
        "counts": dict(Counter(r["qid"] for r in selected)),
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pairs": len(selected),
                "images": len({r["item_id"] for r in selected}),
                "counts": protocol["counts"],
            }
        )
    )


if __name__ == "__main__":
    main()
