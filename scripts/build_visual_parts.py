"""Build deterministic classical visual specialists over the frozen catalog."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from curio.represent.visual_parts import FEATURE_DIMENSIONS, PARAMETERS, VERSION, extract_visual_parts


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sheet(examples, output, root):
    width, height = 300, 180
    sheet = Image.new("RGB", (width * 4, height * ((len(examples) + 3) // 4)), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(examples):
        x, y = index % 4 * width, index // 4 * height
        with Image.open(root / row["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((140, 140))
            sheet.paste(image, (x + 2, y + 2))
        mask = Image.open(output / row["mask_path"]).convert("L")
        mask.thumbnail((140, 140))
        sheet.paste(mask.convert("RGB"), (x + 150, y + 2))
        draw.text((x + 3, y + 144), row["item_id"], fill="black")
        reason = ",".join(row["diagnostics"]["reasons"]) or "accepted heuristic mask"
        draw.text((x + 3, y + 160), reason[:42], fill="black")
    return sheet


def build_visual_parts(data_path="data/snapshots/collection_v2.jsonl",
                       output="artifacts/visual_parts", *, root=".", workers=8):
    root = Path(root).resolve()
    data_path = Path(data_path)
    data_path = data_path if data_path.is_absolute() else root / data_path
    output = Path(output)
    output = output if output.is_absolute() else root / output
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["item_id"]) for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Catalog must be nonempty and item IDs unique")
    output.mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(exist_ok=True)
    started = time.monotonic()

    def process(row):
        path = root / row["image_path"]
        digest = _sha(path)
        expected = row.get("image_sha256")
        if expected and digest != expected:
            raise ValueError(f"Frozen processed image changed: {row['item_id']}")
        result = extract_visual_parts(path)
        # Derived mask names use a hash so unexpected IDs cannot escape the directory.
        mask_path = f"masks/{hashlib.sha256(row['item_id'].encode()).hexdigest()[:24]}.png"
        Image.fromarray(result["mask"]).save(output / mask_path)
        diagnostic = {"item_id": row["item_id"], "image_path": row["image_path"],
                      "image_sha256": digest, "mask_path": mask_path,
                      "mask_sha256": _sha(output / mask_path), "diagnostics": result["diagnostics"]}
        return result, diagnostic

    arrays = {feature: [] for feature in FEATURE_DIMENSIONS}
    validity = {feature: [] for feature in FEATURE_DIMENSIONS}
    diagnostics = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (result, diagnostic) in enumerate(pool.map(process, rows)):
            for feature in FEATURE_DIMENSIONS:
                arrays[feature].append(result[feature])
                validity[feature].append(result["valid"][feature])
            diagnostics.append(diagnostic)
            if (index + 1) % 250 == 0:
                print(f"Processed {index + 1}/{len(rows)}", flush=True)
    payload = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
    payload.update({f"valid_{key}": np.asarray(value, dtype=bool) for key, value in validity.items()})
    np.savez_compressed(output / "features.npz", **payload)
    (output / "item_ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "diagnostics.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in diagnostics), encoding="utf-8")
    for accepted, label in ((True, "accepted"), (False, "unknown")):
        choices = [row for row in diagnostics if row["diagnostics"]["valid"]["shape"] == accepted]
        if choices:
            chosen = [choices[int(i)] for i in np.linspace(0, len(choices) - 1, min(48, len(choices)), dtype=int)]
            _sheet(chosen, output, root).save(output / f"mask_qa_{label}.jpg", quality=92)
    reason_counts = Counter(reason for row in diagnostics for reason in row["diagnostics"]["reasons"])
    manifest = {"version": VERSION, "method": "classical robust border RGB plane + connected components",
                "trained_segmentation": False, "uses_sam": False, "uses_dino": False,
                "source_catalog": str(data_path), "data_sha256": _sha(data_path),
                "items": len(ids), "parameters": PARAMETERS,
                "feature_dimensions": FEATURE_DIMENSIONS,
                "implementation_sha256": _sha(Path(__file__).resolve().parents[1] / "src/curio/represent/visual_parts.py"),
                "builder_sha256": _sha(__file__),
                "normalization": "l2 per valid feature; invalid features are zeros",
                "feature_metadata": {"shape": "32-bin row/column silhouette width and center profiles plus 4 geometry values",
                                     "color": "foreground HSV 12x4x4 square-root histogram",
                                     "texture": "10 rotation-invariant uniform LBP bins +16 gradient orientation +8 gradient magnitude bins"},
                "valid_counts": {key: int(sum(values)) for key, values in validity.items()},
                "rejection_reasons": dict(reason_counts),
                "image_manifest_sha256": hashlib.sha256(json.dumps([(row["item_id"], row["image_sha256"]) for row in diagnostics]).encode()).hexdigest(),
                "artifact_sha256": {name: _sha(output / name) for name in ("features.npz", "item_ids.json", "diagnostics.jsonl")},
                "created_at": datetime.now(timezone.utc).isoformat(), "seconds": time.monotonic() - started}
    manifest["index_sha256"] = hashlib.sha256(json.dumps(manifest["artifact_sha256"], sort_keys=True).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--output", default="artifacts/visual_parts")
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build_visual_parts(args.data, args.output, root=args.root, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
