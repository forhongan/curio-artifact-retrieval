"""Evaluate an image multi-crop ensemble without changing the production index.

Each gallery image is encoded as the full image plus five overlapping square
crops. The normalized crop embeddings are averaged and normalized again. This
is a research-only ablation for the restricted visual gallery; it never writes
to ``artifacts/siglip_adapted``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageOps

from curio.evaluate.metrics import evaluate_ranking
from curio.evaluate.stats import bootstrap_ci
from curio.query import parse_query
from curio.represent.encoders import Encoder
from curio.represent.train import read_items


def crop_images(path: str | Path, mode: str) -> list[Image.Image]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if mode == "full":
        return [image]
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    center = image.crop((left, top, left + side, top + side))
    margin_x = max(0, int(width * 0.15))
    margin_y = max(0, int(height * 0.15))
    boxes = [
        (0, 0, min(width, side + margin_x), min(height, side + margin_y)),
        (max(0, width - side - margin_x), 0, width, min(height, side + margin_y)),
        (0, max(0, height - side - margin_y), min(width, side + margin_x), height),
        (max(0, width - side - margin_x), max(0, height - side - margin_y), width, height),
    ]
    return [image, center, *(image.crop(box) for box in boxes)]


def encode_pil(encoder: Encoder, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
    rows = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            with encoder.autocast():
                features = encoder.model.get_image_features(**encoder.image_inputs(batch))
            features = torch.nn.functional.normalize(features.float(), dim=-1)
            rows.append(features.cpu().numpy())
    return np.concatenate(rows).astype(np.float32)


def gallery_embeddings(encoder: Encoder, items: list[dict], gallery_ids: set[str], mode: str) -> tuple[np.ndarray, list[str], float]:
    started = time.perf_counter()
    vectors, ids = [], []
    for item in items:
        item_id = str(item["item_id"])
        if item_id not in gallery_ids:
            continue
        crops = crop_images(item["image_path"], mode)
        encoded = encode_pil(encoder, crops)
        vector = encoded.mean(axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        vectors.append(vector)
        ids.append(item_id)
    return np.asarray(vectors, dtype=np.float32), ids, (time.perf_counter() - started) * 1000


def run(args) -> dict:
    data_path = Path(args.data)
    queries_path = Path(args.queries)
    qrels_path = Path(args.qrels)
    gallery_path = Path(args.gallery_ids)
    items = read_items(data_path)
    queries = read_items(queries_path)
    qrels = {row["qid"]: row["grades"] for row in read_items(qrels_path)}
    gallery_ids = set(json.loads(gallery_path.read_text(encoding="utf-8")))
    manifest = json.loads((Path(args.index) / "manifest.json").read_text(encoding="utf-8"))
    adapter = manifest.get("adapter_path")
    encoder = Encoder(args.encoder, args.device, adapter_path=adapter)
    encoder.encode_texts(["warm up"])
    image_vectors, image_ids, encode_ms = gallery_embeddings(encoder, items, gallery_ids, args.mode)
    records = []
    for query in queries:
        text = query[f"text_{args.language}"]
        parsed = parse_query(text)
        query_vector = encoder.encode_texts([parsed.rewritten])[0]
        query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
        scores = image_vectors @ query_vector
        ranking = [image_ids[idx] for idx in np.argsort(-scores, kind="stable")[:20]]
        records.append({"qid": query["qid"], **evaluate_ranking(ranking, qrels[query["qid"]]), "ranking": ranking})
    metrics = {key: bootstrap_ci([row[key] for row in records]) for key in ("ndcg@10", "ap@10", "precision@5", "recall@20", "mrr@10")}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "language": args.language,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "qrels_sha256": hashlib.sha256(qrels_path.read_bytes()).hexdigest(),
        "gallery_sha256": hashlib.sha256(gallery_path.read_bytes()).hexdigest(),
        "index_manifest_sha256": hashlib.sha256((Path(args.index) / "manifest.json").read_bytes()).hexdigest(),
        "model": encoder.metadata,
        "gallery_count": len(image_ids),
        "query_count": len(records),
        "image_encode_ms": round(encode_ms, 2),
        "metrics": metrics,
        "warning": "Research-only multi-crop ablation on the AI-labeled restricted gallery; not human gold and not production evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--index", default="artifacts/siglip_adapted")
    parser.add_argument("--encoder", default="siglip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--queries", default="eval/visual_dev/queries.jsonl")
    parser.add_argument("--qrels", default="eval/visual_dev/qrels.jsonl")
    parser.add_argument("--gallery-ids", default="eval/visual_dev/gallery_ids.json")
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--mode", choices=("full", "multicrop"), default="multicrop")
    parser.add_argument("--output", default="artifacts/evaluation/multicrop_visual_dev_en.json")
    args = parser.parse_args()
    report = run(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
