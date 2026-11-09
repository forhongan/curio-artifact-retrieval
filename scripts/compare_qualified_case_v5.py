"""Same-query text controls versus checked external reference aggregation."""

import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.query import parse_query  # noqa: E402
from curio.represent.encoders import Encoder  # noqa: E402
from curio.retrieve.dynamic_reference_runtime import make_sheet  # noqa: E402


def main():
    out = ROOT / "artifacts/qualified_reference_v5/qianlong"
    existing = json.loads((out / "result.json").read_text(encoding="utf-8"))
    query = "Qianlong famille rose vase"
    ids = json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    texts = [query, parse_query(query).rewritten]
    encoder = Encoder("siglip")  # unchanged text tower for vision-only adapters
    scores = vectors @ encoder.encode_texts(texts).T
    rankings = {k: v[:5] for k, v in existing["rankings"].items()}
    for i, name in enumerate(["production_raw_text", "production_rewritten_text"]):
        rankings[name] = [ids[j] for j in np.argsort(-scores[:, i])[:5]]
    catalog = {
        r["item_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    pool = sorted({v for ranking in rankings.values() for v in ranking})
    for key, ranking in rankings.items():
        make_sheet(
            out / f"comparison_{key}.jpg",
            query + " | " + key,
            [(v, ROOT / catalog[v]["image_path"]) for v in ranking],
            columns=5,
        )
    manifest = {
        "query": query,
        "text_controls": texts,
        "rankings": rankings,
        "review_pool": [
            {
                k: catalog[v].get(k)
                for k in [
                    "item_id",
                    "title",
                    "period",
                    "year_from",
                    "year_to",
                    "medium",
                    "source_url",
                    "image_path",
                ]
            }
            for v in pool
        ],
        "evaluation": "exploratory same-query pool; view images and source fields before assigning strict professional relevance",
    }
    (out / "comparison.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(len(pool), "pooled targets; gallery vectors reused")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
