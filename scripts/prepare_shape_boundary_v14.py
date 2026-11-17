"""Freeze unseen shape candidates, preserving previous holdouts and image groups."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from prepare_boundary_followup import sheet

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/boundary_shape_v14"
OUT = ROOT / "artifacts/shape_boundary_v14"


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def order(row, salt):
    return hashlib.sha256(("shape-v14/" + salt + row["object_group"]).encode()).hexdigest()


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    gallery = readl(ROOT / "data/snapshots/collection_v2.jsonl")
    used = set()
    sources = [
        "relevance_v5/items.jsonl",
        "boundary_assisted_v10/labels.jsonl",
        "boundary_targeted_v10/labels.jsonl",
        "boundary_targeted_v10/gallery_diagnostic_labels.jsonl",
        "boundary_targeted_v10/support_diagnostic_labels.jsonl",
        "boundary_factors_v11/labels.jsonl",
        "boundary_factors_v11/components.jsonl",
        "boundary_factors_v11/stress_labels.jsonl",
        "boundary_surface_v13/gallery_audit_labels.jsonl",
    ]
    for name in sources:
        used.update(r["item_id"] for r in readl(ROOT / "eval" / name))
    used.update(
        r["item_id"]
        for r in json.loads((ROOT / "artifacts/joint_vlm_v9/specialty_report.json").read_text())[
            "rows"
        ]
    )
    excluded_groups = {r["object_group"] for r in gallery if r["item_id"] in used}
    excluded_hashes = {r["image_sha256"] for r in gallery if r["item_id"] in used}
    groups, hashes, fresh = set(excluded_groups), set(excluded_hashes), []
    for r in sorted(gallery, key=lambda r: r["item_id"]):
        if r["object_group"] in groups or r["image_sha256"] in hashes:
            continue
        fresh.append(r)
        groups.add(r["object_group"])
        hashes.add(r["image_sha256"])
    index = {
        ident: i
        for i, ident in enumerate(
            json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text())
        )
    }
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    old = readl(ROOT / "eval/boundary_factors_v11/components.jsonl")
    chosen, used_new = [], set()

    def add(rows, atom, split, selection):
        for r in rows:
            if r["object_group"] in used_new:
                raise ValueError("Duplicate object group")
            used_new.add(r["object_group"])
            chosen.append(
                {
                    **{
                        k: r[k]
                        for k in [
                            "item_id",
                            "object_group",
                            "image_sha256",
                            "image_path",
                            "title",
                            "medium",
                            "object_name",
                            "source_url",
                        ]
                    },
                    "key": atom + ":" + r["item_id"],
                    "atom": atom,
                    "qid": atom,
                    "source_split": r["split"],
                    "split": split,
                    "selection": selection,
                }
            )

    def available():
        return [r for r in fresh if r["object_group"] not in used_new]

    # Named candidates are discovery aids, not labels. Retain every remaining
    # meiping candidate; only source-train objects may enter fitting.
    named = sorted(
        [r for r in available() if "meiping" in r["title"].lower()],
        key=lambda r: order(r, "meiping_named"),
    )
    train = [r for r in named if r["split"] == "train"][:6]
    add(train, "meiping", "train", "catalogue_candidate")
    rest = [r for r in named if r["object_group"] not in used_new]
    for n, r in enumerate(rest):
        add(
            [r],
            "meiping",
            "test" if r["split"] == "test" or n % 2 else "dev",
            "catalogue_candidate",
        )
    for atom in ["meiping", "gourd"]:
        positives = [
            r for r in old if r["atom"] == atom and r["split"] == "train" and r["grade"] == 3
        ]
        centroid = vectors[[index[r["item_id"]] for r in positives]].mean(axis=0)
        # Gourd candidates include any nearby object, not just named bottles.
        pool = available()
        if atom == "meiping":
            pool = [r for r in pool if re.search(r"vase|bottle|flask|jar", r["title"], re.I)]
        pool = sorted(
            pool, key=lambda r: (-float(vectors[index[r["item_id"]]] @ centroid), r["item_id"])
        )
        if atom == "gourd":
            hard = [
                r
                for r in pool
                if re.search(r"double vase|double.sided teapot", r["title"], re.I)
                and r["split"] == "train"
            ]
            add(hard, atom, "train", "parallel_container_candidate")
        target = 8 if atom == "meiping" else 12
        count_train = sum(
            r["atom"] == atom
            and r["split"] == "train"
            and r["selection"] == "parallel_container_candidate"
            for r in chosen
        )
        train = [r for r in pool if r["object_group"] not in used_new and r["split"] == "train"]
        add(train[: target - count_train], atom, "train", "similar_unreviewed_object")
        rest = [r for r in pool if r["object_group"] not in used_new][: target * 2]
        for n, r in enumerate(rest):
            add([r], atom, "dev" if n % 2 == 0 else "test", "similar_unreviewed_object")
    for category, pattern, n in [
        ("bowl", r"\bbowl\b", 8),
        ("cup", r"\bcup\b", 4),
        ("plate", r"\bplate\b|\bdish\b", 4),
    ]:
        pool = sorted(
            [r for r in available() if re.search(pattern, r["title"], re.I)],
            key=lambda r: order(r, category),
        )
        add(
            [r for r in pool if r["split"] == "train"][:n], "bowl", "train", category + "_candidate"
        )
        rest = [r for r in pool if r["object_group"] not in used_new][: n * 2]
        for i, r in enumerate(rest):
            add([r], "bowl", "dev" if i % 2 == 0 else "test", category + "_candidate")
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen)
    target = BASE / "candidates.jsonl"
    if target.exists() and target.read_text() != text:
        raise ValueError("Frozen candidates changed")
    target.write_text(text)
    for atom in ["meiping", "gourd", "bowl"]:
        rows = [r for r in chosen if r["atom"] == atom]
        for start in range(0, len(rows), 12):
            sheet(
                [dict(r, caption=r["split"] + " | UNLABELED") for r in rows[start : start + 12]],
                OUT / f"{atom}_{start // 12 + 1}.png",
                atom + " | fresh shape candidates",
            )
    manifest = {
        "candidates_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prior_review_sources": sources,
        "fresh_pairs": len(chosen),
        "fresh_object_groups": len(used_new),
        "prior_reviewed_groups": len(excluded_groups),
        "source_train_heldout_caveat": "Some new-head holdouts were source-train for old encoders; not necessarily unseen by every pretrained/earlier model.",
        "human_confirmed": False,
        "release_approved": False,
    }
    (BASE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                a: {
                    s: sum(r["atom"] == a and r["split"] == s for r in chosen)
                    for s in ["train", "dev", "test"]
                }
                for a in ["meiping", "gourd", "bowl"]
            }
        )
    )


if __name__ == "__main__":
    main()
