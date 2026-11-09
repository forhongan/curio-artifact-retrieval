"""Compare masks on the dedicated 24-image real museum-photo QA set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curio.represent.background_v4 import _background_probability, _border_mask, _open, _lab, isolate_object
from eval_background_v4 import metrics

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "eval/background_truth_20261108"
OUT = ROOT / "artifacts/bg_methods_20261108"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grabcut(image):
    rgb = _open(image)
    height, width = rgb.shape[:2]
    mask = np.full((height, width), cv2.GC_BGD, np.uint8)
    margin = 0.05
    seed = int.from_bytes(hashlib.sha256(rgb.tobytes()).digest()[:4], "little") & 0x7FFFFFFF
    cv2.setRNGSeed(seed)
    try:
        cv2.grabCut(rgb, mask, (round(width * margin), round(height * margin),
                               max(1, width - round(width * margin * 2)),
                               max(1, height - round(height * margin * 2))),
                    None, None, 3, cv2.GC_INIT_WITH_RECT)
        foreground = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
        return foreground, {"status": "completed", "method": "grabcut-v1"}
    except cv2.error:
        return np.zeros((height, width), bool), {"status": "abstain", "method": "grabcut-v1"}


def border_component(image):
    rgb = _open(image)
    lab = _lab(rgb)
    border, thickness = _border_mask(lab.shape, .04)
    bgprob, _ = _background_probability(lab, border)
    foreground = bgprob < .12
    from scipy import ndimage
    foreground = ndimage.binary_opening(foreground, iterations=1)
    foreground = ndimage.binary_closing(foreground, iterations=1)
    labels, count = ndimage.label(foreground)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        foreground = labels == sizes.argmax()
    return foreground, {"status": "completed", "method": "border-connected-component"}


def evaluate_method(name, predict, row):
    start = time.perf_counter()
    output = predict(ROOT / row["image_path"])
    if isinstance(output, dict):
        prediction, diagnostics = output["mask_binary"], output
    else:
        prediction, diagnostics = output
    elapsed = time.perf_counter() - start
    with Image.open(TRUTH / f"{row['item_id']}.png") as loaded:
        truth = np.asarray(loaded, bool)
    prediction = cv2.resize(prediction.astype(np.uint8), truth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
    score = metrics(prediction, truth)
    return {"method": name, "item_id": row["item_id"], "truth_type": row["truth_type"],
            "status": diagnostics["status"], "seconds": elapsed, **score,
            "Dice": 2 * score["precision"] * score["recall"] / (score["precision"] + score["recall"])
            if score["precision"] + score["recall"] else 0.0,
            "predicted_area": float(prediction.mean()), "truth_area": float(truth.mean())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((TRUTH / "records.json").read_text(encoding="utf8"))
    methods = {"old_grabcut": grabcut, "v4": isolate_object, "border_cc": border_component}
    results = []
    for name, predict in methods.items():
        for row in records:
            results.append(evaluate_method(name, predict, row))
    summaries = {}
    for name in methods:
        subset = [r for r in results if r["method"] == name]
        summaries[name] = {
            "all_mean_IoU": float(np.mean([r["IoU"] for r in subset])),
            "all_mean_Dice": float(np.mean([r["Dice"] for r in subset])),
            "all_mean_boundary_F1": float(np.mean([r["boundary_f1"] for r in subset])),
            "coverage": sum(r["status"] == "accepted" or r["status"] == "completed" for r in subset) / len(subset),
            "mean_seconds": float(np.mean([r["seconds"] for r in subset])),
            "by_truth_type": {kind: {
                "IoU": float(np.mean([r["IoU"] for r in subset if r["truth_type"] == kind])),
                "Dice": float(np.mean([r["Dice"] for r in subset if r["truth_type"] == kind])),
                "boundary_F1": float(np.mean([r["boundary_f1"] for r in subset if r["truth_type"] == kind])),
            } for kind in ("object", "planar")},
        }
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    sheet = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), "24-mask validation; rows: original / truth / old GrabCut / v4 / border+CC", fill="black", font=font)
    for col, row in enumerate(records[:8]):
        with Image.open(ROOT / row["image_path"]) as loaded:
            image = loaded.convert("RGB")
        with Image.open(TRUTH / f"{row['item_id']}.png") as loaded:
            truth = np.asarray(loaded, bool)
        predictions = [grabcut(ROOT / row["image_path"])[0], isolate_object(ROOT / row["image_path"])["mask_binary"],
                       border_component(ROOT / row["image_path"])[0]]
        tiles = [image, Image.fromarray(truth.astype(np.uint8) * 255).convert("RGB")]
        for mask in predictions:
            resized = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST).astype(bool)
            tiles.append(Image.fromarray(resized.astype(np.uint8) * 255).convert("RGB"))
        for j, tile in enumerate(tiles):
            tile = tile.copy()
            tile.thumbnail((180, 130))
            sheet.paste(tile, (col * 200, 30 + j * 136))
        draw.text((col * 200, 700), row["item_id"], fill="black", font=font)
    sheet.save(OUT / "comparison_grid.jpg", quality=92)
    payload = {"truth": "eval/background_truth_20261108 (24 approximate primary-agent pixel masks; not independent human gold)",
               "summaries": summaries, "results": results}
    (OUT / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
    (OUT / "manifest.json").write_text(json.dumps({p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}, indent=2), encoding="utf8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
