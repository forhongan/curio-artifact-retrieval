"""Freeze a fresh, pixel-only attribute benchmark before any model selection."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/cv_research_20261104"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readrows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    if (OUT / "protocol.json").exists():
        raise FileExistsError(
            "Frozen benchmark exists; use a separate directory for a new protocol"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    catalog_path = ROOT / "data/snapshots/collection_v2.jsonl"
    old_path = ROOT / "eval/visual_labels/annotations.jsonl"
    catalog, old = readrows(catalog_path), readrows(old_path)
    by_id = {r["item_id"]: r for r in catalog}
    excluded = {by_id[r["item_id"]]["object_group"] for r in old}
    previous_pool = ROOT / "artifacts/reference_eval/pool_private_mapping.json"
    if previous_pool.exists():
        excluded.update(
            by_id[r["item_id"]]["object_group"]
            for r in json.loads(previous_pool.read_text(encoding="utf8"))
        )
    seed = "curio-cv-pixels-20261104-v1"
    selected = []
    for split, count in (("dev", 48), ("test", 96)):
        order = sorted(
            (r for r in catalog if r["split"] == split and r["object_group"] not in excluded),
            key=lambda r: hashlib.sha256((seed + r["item_id"]).encode()).hexdigest(),
        )
        chosen, groups = [], set()
        for source in ("met", "cleveland"):
            for r in order:
                if r["source"] == source and r["object_group"] not in groups:
                    chosen.append(r)
                    groups.add(r["object_group"])
                    if sum(x["source"] == source for x in chosen) == count // 2:
                        break
        if len(chosen) != count:
            raise ValueError("Insufficient disjoint families")
        chosen.sort(
            key=lambda r: hashlib.sha256((seed + "alias" + r["item_id"]).encode()).hexdigest()
        )
        for i, r in enumerate(chosen):
            selected.append(
                {
                    "alias": ("D" if split == "dev" else "V") + f"{i + 1:03d}",
                    "item_id": r["item_id"],
                    "split": split,
                    "object_group": r["object_group"],
                    "image_path": r["image_path"],
                    "image_sha256": r["image_sha256"],
                    "source": r["source"],
                }
            )
    train = [r for r in old if r["split"] == "train"]
    assert len(train) == 218
    all_groups = {by_id[r["item_id"]]["object_group"] for r in train}
    assert not all_groups.intersection(r["object_group"] for r in selected)
    (OUT / "blind_map.json").write_text(json.dumps(selected, indent=2), encoding="utf8")
    # The extractor sees only identifiers, pixels and hashes, never titles or labels.
    pixels = [
        {"item_id": r["item_id"], "image_path": r["image_path"], "image_sha256": r["image_sha256"]}
        for r in catalog
    ]
    (OUT / "pixels.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in pixels), encoding="utf8"
    )
    for person in ("a", "b"):
        directory = OUT / f"annotator_{person}"
        directory.mkdir(exist_ok=True)
        batch = [r for i, r in enumerate(selected) if i % 2 == (0 if person == "a" else 1)]
        for start in range(0, len(batch), 6):
            sheet = Image.new("RGB", (1200, 1220), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            for slot, r in enumerate(batch[start : start + 6]):
                path = ROOT / r["image_path"]
                if sha(path) != r["image_sha256"]:
                    raise ValueError("Image hash mismatch")
                with Image.open(path) as im:
                    tile = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (560, 355))
                x, y = (slot % 2) * 600, (slot // 2) * 405
                sheet.paste(tile, (x + (600 - tile.width) // 2, y + 35 + (355 - tile.height) // 2))
                draw.text((x + 12, y + 8), r["alias"], fill="black", font_size=22)
            sheet.save(directory / f"sheet_{start // 6 + 1:02d}.jpg", quality=94)
        (directory / "aliases.json").write_text(
            json.dumps([r["alias"] for r in batch]), encoding="utf8"
        )
    queries = [
        ("vase", "瓶形器物", "a vase", [["object_type", "vase", 1]]),
        ("bowl", "碗形器物", "a bowl", [["object_type", "bowl", 1]]),
        ("plate", "盘形器物", "a plate", [["object_type", "plate", 1]]),
        (
            "figure",
            "人物或动物立体造像",
            "a three dimensional figurine",
            [["object_type", "figure", 1]],
        ),
        ("cup", "杯形器物", "a cup", [["object_type", "cup", 1]]),
        ("textile", "平面的织物", "a flat textile", [["object_type", "textile", 1]]),
        ("green", "绿色器物", "a green object", [["colors", "green", 1]]),
        ("blue", "带蓝色的器物", "an object with blue color", [["colors", "blue", 1]]),
        ("white", "带白色的器物", "an object with white color", [["colors", "white", 1]]),
        ("red", "红色器物", "a red object", [["colors", "red", 1]]),
        ("brown", "褐色器物", "a brown object", [["colors", "brown", 1]]),
        (
            "handles",
            "有可见把手或耳的器物",
            "an object with visible handles",
            [["shape", "handles", 1]],
        ),
        (
            "narrow",
            "窄口细颈器物",
            "a vessel with a narrow mouth or neck",
            [["shape", "narrow_mouth", 1]],
        ),
        ("wide", "敞口器物", "a vessel with a wide open mouth", [["shape", "wide_mouth", 1]]),
        ("round", "有圆鼓器腹的器物", "a vessel with a rounded body", [["shape", "round_body", 1]]),
        (
            "green_narrow",
            "绿色窄口器物",
            "a green vessel with a narrow mouth",
            [["colors", "green", 1], ["shape", "narrow_mouth", 1]],
        ),
        (
            "blue_white",
            "蓝白双色器物",
            "a blue and white object",
            [["colors", "blue", 1], ["colors", "white", 1]],
        ),
        (
            "handled_round",
            "有耳且器腹圆鼓的器物",
            "a rounded vessel with visible handles",
            [["shape", "handles", 1], ["shape", "round_body", 1]],
        ),
        (
            "vase_green",
            "绿色瓶形器物",
            "a green vase",
            [["object_type", "vase", 1], ["colors", "green", 1]],
        ),
        (
            "bowl_white",
            "白色碗形器物",
            "a white bowl",
            [["object_type", "bowl", 1], ["colors", "white", 1]],
        ),
        (
            "narrow_nohandle",
            "窄口且没有可见把手的器物",
            "a narrow mouth vessel without visible handles",
            [["shape", "narrow_mouth", 1], ["shape", "handles", 0]],
        ),
    ]
    (OUT / "queries.json").write_text(
        json.dumps(
            [dict(qid=q, zh=z, en=e, atoms=a) for q, z, e, a in queries],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    protocol = {
        "version": "cv-pixels-20261104-v1",
        "seed": seed,
        "catalog_sha256": sha(catalog_path),
        "old_annotations_sha256": sha(old_path),
        "mapping_sha256": sha(OUT / "blind_map.json"),
        "queries_sha256": sha(OUT / "queries.json"),
        "train": 218,
        "dev": 48,
        "test": 96,
        "sampling": "hash order, 50/50 sources, one per official family, exclude all prior AI annotation families and reference pool families",
        "supervision": "existing218 AI pixel-only training annotations; fresh independent GPT-6.1 Terra blind-image dev/test labels",
        "label_limits": "AI attribute labels, not human gold; no material, chronology, or semantic fine motif claims",
        "candidate_inputs": "image bytes only, opaque IDs; no gallery metadata/captions/labels used for prediction or ranking",
        "evaluation": "21 fixed intents, relevance=all query attributes explicitly observed; any known contradiction negative, unknown otherwise; report unknown and zero-positive intents; no dropping after seeing method results",
        "models": [
            "raw HOG LR",
            "raw HSV LR",
            "raw LBP LR",
            "raw concatenation LR",
            "GrabCut concatenation LR",
            "DINOv2 image-only LR",
            "SigLIP image LR control",
            "SigLIP zero-shot dot baseline",
        ],
        "selection": "LR balanced binary per attribute with train>=5 positive and5negative, C in[0.1,1,10] selected by dev macro average precision; no test loading during fitting",
        "retrieval": "query atoms -> mean log attribute probability, score uses predictions only; unsupported query atoms explicitly fail coverage",
        "workflow": "predeclare DINO attribute recall K in[10,25,50] then equal RRF classical attribute rerank; fusion weight[0,0.25,0.5,0.75,1] dev only; evaluate default and selected; no promotion of default from this small audit",
        "timing": "single CPU thread scoring, warmup then repeated online query parse+score vs text encoder+score; feature build/loading/training separate; throughput not HTTP latency",
        "robustness": "heldout test images fixed brightness0.65, JPEG quality25, background white compositing only when mask usable; original labels assumed invariant, no artifact used for training or choosing settings",
        "artifacts": "new directory, original production and frozen reports preserved",
        "source_counts": dict(Counter(r["source"] for r in selected)),
    }
    (OUT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        json.dumps(
            {
                "output": str(OUT),
                "fresh": len(selected),
                "train": len(train),
                "protocol_sha256": sha(OUT / "protocol.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
