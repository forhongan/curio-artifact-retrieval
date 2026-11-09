"""Publish compact, traceable experiment statistics and CC0 gallery thumbnails."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/public/assets"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts/relevance_v5/results.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    records = raw["records"]
    models = []
    for key, label in [
        ("clip", "CLIP"),
        ("siglip2", "SigLIP2"),
        ("production_lora", "SigLIP2 + LoRA"),
        ("team_comp_lora", "双语池化头 LoRA"),
    ]:
        row = {"key": key, "label": label}
        for lang in ["en", "zh"]:
            r = next(
                x
                for x in records
                if x["model"] == key and x["method"] == "dense" and x["language"] == lang
            )
            row[lang] = r["metrics"]
        models.append(row)
    sets = []
    for key, label in [
        ("adaptive_gmm", "自适应 GMM"),
        ("dense", "共享分数边界"),
        ("calibrated", "综合分数边界"),
        ("aux_raw_color", "颜色辅助边界"),
    ]:
        row = {"key": key, "label": label}
        for lang in ["en", "zh"]:
            row[lang] = next(
                x
                for x in records
                if x["model"] == "production_lora" and x["method"] == key and x["language"] == lang
            )["metrics"]
        sets.append(row)
    data = {
        "version": "v5.1",
        "evaluated_images": 140,
        "dev_images": 60,
        "intents": 20,
        "gallery": 4692,
        "models": models,
        "sets": sets,
        "timing": [
            {"label": a, "seconds": b}
            for a, b in [
                ("CLIP", 126.4),
                ("SigLIP", 113.9),
                ("SigLIP2", 104.4),
                ("LoRA", 102.5),
                ("DINO", 122.0),
            ]
        ],
        "image_components": json.loads(
            (ROOT / "artifacts/relevance_v5/image_components/results.json").read_text(
                encoding="utf-8"
            )
        )["summary"],
        "provenance": {
            "source": "artifacts/relevance_v5/results.json",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "protocol": "20 bilingual intents, 60 dev / 140 test reviewed images, v5.1; known-only ranking; full-gallery decisions, test-core set truth",
            "timing_source": "docs/CV_BACKBONES_20261104.md",
            "timing_protocol": "RTX3070, batch16, 4692 images, includes IO/preprocessing/encoding, excludes model loading/saving",
        },
    }
    (OUT / "research-v6.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    catalog = {
        r["item_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    ids = ["cma_122448", "met_52024", "met_52976", "met_42144", "met_52765"]
    for key in ["gourd", "meiping", "moonflask", "liao", "celadon", "dragon"]:
        path = ROOT / f"artifacts/reference_scenarios_v6/{key}/comparison.json"
        if path.exists():
            case = json.loads(path.read_text(encoding="utf-8"))
            for ranking in case.get("rankings", {}).values():
                ids.extend(ranking[:5])
    images = OUT / "showcase"
    images.mkdir(exist_ok=True)
    attribution = []
    for item_id in dict.fromkeys(ids):
        row = catalog[item_id]
        with Image.open(ROOT / row["image_path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((650, 650))
            im.save(images / f"{item_id}.jpg", quality=88)
        attribution.append(
            {
                k: row.get(k)
                for k in ["item_id", "title", "source", "source_url", "license", "credit_line"]
            }
        )
    (images / "attribution.json").write_text(
        json.dumps(attribution, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(len(attribution), "CC0 gallery thumbnails; metrics from", source)


if __name__ == "__main__":
    main()
