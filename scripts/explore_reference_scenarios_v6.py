"""Real external-reference scenarios; cache gallery vectors and retain all runs."""

import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from curio.retrieve.dynamic_reference_runtime import DynamicReferenceRuntime, make_sheet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/reference_scenarios_v6"
CASES = [
    ("gourd", "Chinese double gourd vase", "葫芦瓶"),
    ("meiping", "Chinese meiping vase", "梅瓶"),
    ("moonflask", "Chinese blue and white moon flask", "青花抱月瓶"),
    ("liao", "Liao dynasty leather flask", "辽代皮囊壶"),
    ("celadon", "Chinese celadon vase", "青瓷瓶"),
    ("dragon", "Chinese blue and white dragon dish", "青花龙纹盘"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = DynamicReferenceRuntime(ROOT)
    catalog = {
        r["item_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    ids = json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text())
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    rows = []
    for key, query, label in CASES:
        folder = OUT / key
        start = time.perf_counter()
        result = runtime.run_qualified(
            query,
            folder,
            cache_dir=OUT / "cache",
            reference_limit=6,
            top_k=10,
            sources=("vam", "mia"),
            aggregation="diverse",
        )
        record = {
            "key": key,
            "query": query,
            "label": label,
            "status": result["status"],
            "references": result["reference_count"],
            "elapsed_s": time.perf_counter() - start,
            "acquisition": result["acquisition"],
        }
        if result["reference_count"]:
            text = runtime._siglip.encode_texts([query])[0]
            ranking = [ids[i] for i in np.argsort(-(vectors @ text))[:10]]
            rankings = {
                "production_text": ranking,
                **{k: v[:10] for k, v in result["rankings"].items()},
            }
            pool = sorted({v for ranks in rankings.values() for v in ranks[:5]})
            record["rankings"] = rankings
            record["review_pool"] = [
                {
                    k: catalog[v].get(k)
                    for k in [
                        "item_id",
                        "title",
                        "medium",
                        "period",
                        "description",
                        "source_url",
                        "image_path",
                    ]
                }
                for v in pool
            ]
            make_sheet(
                folder / "text_top5.jpg",
                query + " | production text",
                [(i, ROOT / catalog[i]["image_path"]) for i in ranking[:5]],
                columns=5,
            )
            make_sheet(
                folder / "reference_top5.jpg",
                query + " | reference diverse",
                [(i, ROOT / catalog[i]["image_path"]) for i in rankings["diverse"][:5]],
                columns=5,
            )
            make_sheet(
                folder / "pooled_review.jpg",
                label + " | top-5 pool across all methods",
                [(i, ROOT / catalog[i]["image_path"]) for i in pool],
                columns=5,
            )
        (folder / "comparison.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(record)
        (OUT / "summary.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(key, result["status"], result["reference_count"], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
