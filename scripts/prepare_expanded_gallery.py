"""Index a separately acquired shard with the production image tower.

Reuse frozen gallery vectors, encode only new photographs, and keep the old
snapshot/index and test protocol unchanged. New rows are retrieval-only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from curio.index.build import build_index
from curio.represent.encoders import Encoder
from curio.retrieve.engine import SearchService, load_items


def main():
    base_path = Path("data/snapshots/collection_v2.jsonl")
    extra_path = Path("data/snapshots/expansion_v3.jsonl")
    base, extra = load_items(base_path), load_items(extra_path)
    existing = {r["item_id"] for r in base}
    if existing.intersection(r["item_id"] for r in extra):
        raise ValueError("Expansion contains a frozen gallery ID")
    if any(row.get("split") != "unassigned" for row in extra):
        raise ValueError("Expansion must not silently become training/test data")
    backend = SearchService(base, index_dir="artifacts/siglip_adapted", encoder_name="siglip")
    if backend.embeddings is None or len(backend.vector_rows) != len(base):
        raise ValueError("Frozen base image index failed validation")
    if backend.manifest["data_sha256"] != hashlib.sha256(base_path.read_bytes()).hexdigest():
        raise ValueError("Frozen snapshot changed")
    for row in extra:
        if hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() != row["image_sha256"]:
            raise ValueError(f"Expansion image changed: {row['item_id']}")
    encoder = Encoder("siglip", adapter_path=backend.manifest["adapter_path"])
    vectors = np.load("artifacts/siglip_adapted/image_embeddings.npy", allow_pickle=False)
    ids = json.loads(Path("artifacts/siglip_adapted/item_ids.json").read_text())
    lookup = {key: i for i, key in enumerate(ids)}
    vectors = vectors[[lookup[row["item_id"]] for row in base]]
    combined = Path("data/snapshots/collection_v3_expanded.jsonl")
    combined.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in base + extra), encoding="utf-8")

    class ReuseEncoder:
        metadata = encoder.metadata

        def encode_images(self, paths, batch_size=16):
            new = encoder.encode_images(paths[len(base):], batch_size=batch_size)
            return np.concatenate([vectors, new])

    manifest = build_index(combined, "artifacts/siglip_expanded", "siglip", encoder=ReuseEncoder())
    manifest.update(parent_index_sha256=backend.index_sha256,
                    expansion_data_sha256=hashlib.sha256(extra_path.read_bytes()).hexdigest(),
                    evaluation_status="Retrieval-only expansion; original v2 metrics do not apply",
                    original_items=len(base), expansion_items=len(extra))
    Path("artifacts/siglip_expanded/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
