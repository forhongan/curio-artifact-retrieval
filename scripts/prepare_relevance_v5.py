"""Freeze a model-independent core for image/source relevance review."""

import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval/relevance_v5"


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def prepare():
    OUT.mkdir(exist_ok=True)
    catalogue = readl(ROOT / "data/snapshots/collection_v2.jsonl")
    facts = {
        r["item_id"]: r
        for r in readl(
            ROOT / "artifacts/cv_v4_20261108/catalogue_v4_semantic_review/labels_catalogue_v2.jsonl"
        )
    }
    by_id = {r["item_id"]: r for r in catalogue}
    dev = json.loads((ROOT / "eval/visual_dev/gallery_ids.json").read_text())
    test = json.loads((ROOT / "eval/visual_test/gallery_ids.json").read_text())
    groups = {by_id[i]["object_group"] for i in dev + test}
    candidates = sorted(
        (r for r in catalogue if r["split"] == "test" and r["object_group"] not in groups),
        key=lambda r: hashlib.sha256(("relevance-v5|" + r["item_id"]).encode()).hexdigest(),
    )
    # Round-robin by broad material, for coverage rather than gallery prevalence.
    bins = {}
    for r in candidates:
        key = tuple(facts[r["item_id"]]["derived_ontology"]["broad_material"])
        bins.setdefault(key, []).append(r)
    while len(test) < 140:
        for key in sorted(bins):
            while bins[key] and bins[key][0]["object_group"] in groups:
                bins[key].pop(0)
            if bins[key] and len(test) < 140:
                r = bins[key].pop(0)
                test.append(r["item_id"])
                groups.add(r["object_group"])
    rows = []
    for split, ids in [("dev", dev), ("test", test)]:
        for item in ids:
            r, f = by_id[item], facts[item]
            rows.append(
                {
                    "review_id": f"R{len(rows) + 1:03}",
                    "item_id": item,
                    "split": split,
                    "image_path": r["image_path"],
                    "image_sha256": r["image_sha256"],
                    "object_group": r["object_group"],
                    "official_catalog": f["official_catalog"],
                    "catalogue_facets": f["derived_ontology"],
                }
            )
    (OUT / "items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for start in range(0, len(rows), 10):
        sheet = Image.new("RGB", (1750, 920), "white")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(rows[start : start + 10]):
            x, y = (j % 5) * 350, (j // 5) * 460
            with Image.open(ROOT / r["image_path"]) as im:
                im = ImageOps.contain(im.convert("RGB"), (340, 340))
                sheet.paste(im, (x + (350 - im.width) // 2, y))
            title = r["official_catalog"]["title"]
            medium = (
                r["official_catalog"].get("medium") or r["official_catalog"].get("technique") or ""
            )
            label = (
                f"{r['review_id']} {r['item_id']}\n"
                + "\n".join(textwrap.wrap(title + " | " + medium, 37))[:180]
            )
            draw.text((x + 6, y + 342), label, fill="black", font=font)
        sheet.save(OUT / f"review_{start // 10 + 1:02}.jpg", quality=92)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "version": "relevance-v5",
                "dev": len(dev),
                "test": len(test),
                "sampling": "Original 60 dev + 60 test; 80 deterministic material-stratified test additions; one new object/group.",
                "scope": "AI image plus complete official record review; historical holdouts reused; no claim of fresh blind test.",
                "items_sha256": hashlib.sha256((OUT / "items.jsonl").read_bytes()).hexdigest(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(len(rows))


if __name__ == "__main__":
    prepare()
