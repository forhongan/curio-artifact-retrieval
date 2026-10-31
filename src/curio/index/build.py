from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from curio.retrieve.engine import load_items


def build_index(data_path="data/snapshots/collection_v2.jsonl", output="artifacts/clip", encoder_name="clip", *, encoder=None, adapter_path=None, root=".", batch_size=16):
    root = Path(root).resolve()
    data_path = Path(data_path)
    if not data_path.is_absolute():
        data_path = root / data_path
    items = load_items(data_path)
    if not items:
        raise ValueError("No catalog items; collect data before building an index")
    paths = [Path(item["image_path"]) for item in items]
    paths = [path if path.is_absolute() else root / path for path in paths]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} catalog images missing; first: {missing[0]}")
    if encoder is None:
        from curio.represent.encoders import Encoder
        encoder = Encoder(encoder_name, adapter_path=adapter_path)
    vectors = np.asarray(encoder.encode_images([str(path) for path in paths], batch_size=batch_size), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(items) or not np.isfinite(vectors).all():
        raise ValueError("Encoder produced invalid embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Encoder produced zero vectors")
    vectors /= norms
    ids = [str(item["item_id"]) for item in items]
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    index_sha = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    metadata = getattr(encoder, "metadata", {"encoder": encoder_name})
    manifest = {"index_version": index_sha[:16], "encoder": encoder_name, "encoder_metadata": metadata,
                "model_version": metadata.get("revision") or metadata.get("model_id") or encoder_name,
                "adapter_path": metadata.get("adapter_path"), "adapter_hash": metadata.get("adapter_hash"),
                "data_sha256": data_sha, "index_sha256": index_sha, "items": len(ids), "dimension": vectors.shape[1],
                "catalog_sha256": hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "normalized": True, "similarity": "cosine", "adapted": bool(metadata.get("adapter_path")),
                "created_at": datetime.now(timezone.utc).isoformat()}
    output = Path(output)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "image_embeddings.tmp.npy", vectors)
    (output / "image_embeddings.tmp.npy").replace(output / "image_embeddings.npy")
    for name, value in (("item_ids.json", ids), ("manifest.json", manifest)):
        temp = output / (name + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(output / name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build exact cosine index over museum images")
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--encoder", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--adapter", default=None, help="Optional PEFT image-tower adapter directory")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(build_index(args.data, args.output or f"artifacts/{args.encoder}", args.encoder, adapter_path=args.adapter, batch_size=args.batch_size), indent=2))


if __name__ == "__main__":
    main()
