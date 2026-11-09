"""Freeze a 24-image, pixel-level real-object segmentation QA set.

Truth is manually outlined from pixels by the primary agent. Image identity,
truth type and source hashes are frozen before any v4 parameter comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IDS = [
    "met_47346", "cma_140206", "met_53844", "cma_95094",
    "met_48364", "met_46753", "met_44358", "cma_135586",
    "cma_156948", "met_460684", "cma_101146", "cma_101150",
    "cma_101151", "cma_101152", "cma_101153", "cma_101156",
    "cma_101159", "cma_101163", "cma_101165", "cma_101168",
    "cma_101175", "cma_101185", "cma_101192", "cma_101198",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="freeze truth files")
    args = parser.parse_args()
    catalog = {r["item_id"]: r for r in rows(ROOT / "data/snapshots/collection_v2.jsonl")}
    output = ROOT / "eval/background_truth_20261108"
    if args.write and output.exists():
        existing = [p for p in output.iterdir() if p.name != "truth_contact_sheet.jpg" and not p.name.endswith(".png")]
        if existing:
            raise FileExistsError("Frozen truth exists; use a separate directory")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    sheet = Image.new("RGB", (1200, 1600), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    for i, item_id in enumerate(IDS):
        row = catalog[item_id]
        path = ROOT / row["image_path"]
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        # All test images are <=1024 source pixels; preserve native coordinates.
        mask = Image.new("L", image.size, 0)
        brush = ImageDraw.Draw(mask)
        width, height = image.size
        if i < 10:
            if item_id == "met_47346":
                brush.ellipse((126, 271, 627, 706), fill=255)
                brush.rectangle((178, 245, 216, 388), fill=255)
                brush.rectangle((535, 242, 580, 388), fill=255)
                brush.rectangle((192, 635, 282, 847), fill=255)
                brush.rectangle((341, 635, 432, 809), fill=255)
                brush.rectangle((455, 635, 548, 847), fill=255)
            elif item_id == "cma_140206":
                brush.polygon([(90, 202), (935, 202), (855, 516), (685, 609), (340, 609), (161, 516)], fill=255)
                brush.rectangle((333, 603, 689, 663), fill=255)
            elif item_id == "met_53844":
                brush.ellipse((295, 156, 728, 626), fill=255)
                brush.rectangle((250, 76, 316, 385), fill=255)
                brush.rectangle((704, 72, 768, 385), fill=255)
                brush.ellipse((434, 599, 532, 746), fill=255)
            elif item_id == "cma_95094":
                brush.polygon([(294, 278), (695, 258), (754, 336), (733, 709), (299, 720), (270, 337)], fill=255)
            elif item_id == "met_48364":
                brush.polygon([(94, 258), (591, 258), (590, 757), (489, 806), (130, 774)], fill=255)
                brush.rectangle((92, 245, 594, 277), fill=255)
                brush.ellipse((235, 794, 407, 845), fill=255)
            elif item_id == "met_46753":
                brush.polygon([(268, 218), (757, 218), (772, 656), (651, 747), (372, 747), (253, 656)], fill=255)
                brush.rectangle((267, 202, 758, 243), fill=255)
            elif item_id == "met_44358":
                brush.polygon([(323, 64), (368, 64), (388, 159), (455, 185), (469, 334), (484, 724), (519, 901), (154, 901), (188, 725), (211, 334), (218, 185), (289, 159)], fill=255)
            elif item_id == "cma_135586":
                brush.polygon([(112, 243), (650, 263), (637, 890), (624, 976), (202, 963), (131, 916)], fill=255)
                brush.ellipse((350, 31, 420, 105), fill=255)
            elif item_id == "cma_156948":
                brush.ellipse((126, 73, 588, 1009), fill=255)
                brush.rectangle((260, 826, 516, 979), fill=255)
            elif item_id == "met_460684":
                brush.ellipse((204, 192, 600, 704), fill=255)
                brush.rectangle((230, 104, 284, 310), fill=255)
                brush.rectangle((519, 104, 574, 310), fill=255)
                brush.rectangle((360, 96, 434, 220), fill=255)
                brush.polygon([(204, 569), (167, 655), (198, 821), (352, 821), (391, 655), (436, 569), (574, 569), (574, 808), (539, 890), (248, 890), (204, 808)], fill=255)
        else:
            # Planar truth is the material boundary, not a generic full-frame box.
            boundaries = {
                "cma_101146": (40, 40, 1000, 740), "cma_101150": (0, 30, 900, 950),
                "cma_101151": (20, 40, 990, 950), "cma_101152": (10, 45, 940, 940),
                "cma_101153": (35, 35, 325, 950), "cma_101156": (0, 25, 940, 950),
                "cma_101159": (110, 40, 300, 940), "cma_101163": (30, 35, 1000, 850),
                "cma_101165": (70, 50, 310, 950), "cma_101168": (30, 20, 960, 950),
                "cma_101175": (5, 120, 1000, 430), "cma_101185": (25, 35, 990, 760),
                "cma_101192": (0, 30, 1024, 750), "cma_101198": (15, 20, 990, 920),
            }
            brush.rectangle(boundaries[item_id], fill=255)
        truth = np.asarray(mask, np.uint8) > 0
        # A simple visible compositing QA, not a substitute for boundary review.
        thumb = image.copy()
        thumb.thumbnail((280, 225))
        preview = image.copy()
        overlay = np.asarray(preview).copy()
        overlay[truth] = (overlay[truth] * 0.55 + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
        preview = Image.fromarray(overlay)
        preview.thumbnail((280, 225))
        x, y = (i % 4) * 300, (i // 4) * 250
        sheet.paste(thumb, (x, y + 22))
        sheet.paste(preview, (x + 285 - min(140, preview.width), y + 22))
        draw.text((x + 4, y + 3), item_id, font=font, fill="black")
        records.append({
            "item_id": item_id, "image_path": row["image_path"], "image_sha256": sha(path),
            "truth_type": "object" if i < 10 else "planar", "width": width, "height": height,
            "annotation_method": "primary-agent pixel manual contour/rectangle; frozen before v4 comparison",
        })
        if args.write:
            mask.save(output / f"{item_id}.png")
    if args.write:
        (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf8")
        (output / "truth_contact_sheet.jpg").write_bytes(b"")
        sheet.save(output / "truth_contact_sheet.jpg", quality=92)
        (output / "manifest.json").write_text(json.dumps({
            "records_sha256": sha(output / "records.json"),
            "contact_sheet_sha256": sha(output / "truth_contact_sheet.jpg"),
            "masks": {r["item_id"]: {"sha256": sha(output / f"{r['item_id']}.png"), "bytes": (output / f"{r['item_id']}.png").stat().st_size} for r in records},
        }, indent=2), encoding="utf8")
    print(json.dumps({"count": len(records), "objects": 10, "planar": 14, "written": args.write}, indent=2))


if __name__ == "__main__":
    main()
