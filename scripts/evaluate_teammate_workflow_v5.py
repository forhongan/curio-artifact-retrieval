"""Run the teammate's unmodified retrieval engine in an isolated process.

Use its actual query rewriting, probes and compositional implementation, with
released gallery caches. Added unjudged images remain unknown, never negative.
"""

from pathlib import Path
import importlib.util
import json
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/teammate_review/source"
RES = ROOT / "research/teammate_review/resources"
sys.path.insert(0, str(SOURCE / "src"))
from curio.represent.encoders import Encoder  # noqa: E402
from curio.retrieve.engine import SearchService, load_items  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "local_relevance", ROOT / "src/curio/evaluate/relevance.py"
)
relevance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relevance)


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def main():
    out = ROOT / "artifacts/relevance_v5/team_workflow"
    out.mkdir(parents=True, exist_ok=True)
    queries = readl(ROOT / "eval/relevance_v5/queries.jsonl")
    test = [
        r["item_id"] for r in readl(ROOT / "eval/relevance_v5/items.jsonl") if r["split"] == "test"
    ]
    qrels = {
        (r["qid"], r["item_id"]): r["grade"] for r in readl(ROOT / "eval/relevance_v5/qrels.jsonl")
    }
    records = []
    for model, adapter in [
        ("production_lora", ROOT / "artifacts/training/siglip_lora_random/best_adapter"),
        ("team_comp_lora", RES / "artifacts/training/siglip_lora_comp/best_adapter"),
    ]:
        encoder = Encoder("siglip", adapter_path=adapter)
        encode = encoder.encode_texts
        cache = {}

        def cached(texts):
            key = tuple(texts)
            if key not in cache:
                cache[key] = encode(texts)
            return cache[key]

        encoder.encode_texts = cached
        for size in [4692, 10200]:
            comp = model == "team_comp_lora"
            index = (
                ("siglip_10k_comp_lora" if comp else "siglip_10k")
                if size == 10200
                else ("siglip_comp_lora" if comp else "siglip_adapted")
            )
            probe = (
                ("siglip_10k_comp_lora" if comp else "siglip_10k")
                if size == 10200
                else ("vision_comp_lora" if comp else "vision_metadata_adapted")
            )
            items = load_items(
                RES / f"data/snapshots/collection_{'10k' if size == 10200 else 'v2'}.jsonl"
            )
            service = SearchService(
                items,
                index_dir=RES / "artifacts" / index,
                encoder=encoder,
                encoder_name="siglip",
                root=ROOT,
                probes_path=RES / "artifacts/probes" / probe,
                probe_weight=0.15,
            )
            if service.embeddings is None or service.probe_warning:
                raise ValueError((service.index_warning, service.probe_warning))
            for variant in ["image_only", "subquery_and", "compositional"]:
                for lang in ["en", "zh"]:
                    per = []
                    start = time.perf_counter()
                    for q in queries:
                        result = service.search(q["text_" + lang], top_k=size, variant=variant)
                        if result["effective_variant"] == "unavailable":
                            raise ValueError(result["warnings"])
                        ranking = [r["item_id"] for r in result["results"]]
                        lookup = {key: -i for i, key in enumerate(ranking)}
                        scores = np.array([lookup.get(key, -size - 1) for key in test])
                        grades = np.array([qrels[q["qid"], key] for key in test])
                        per.append(
                            {
                                "qid": q["qid"],
                                **relevance.retrieval_metrics(scores, grades),
                                "effective_variant": result["effective_variant"],
                                "core_scores":scores.tolist(),
                                "top10": ranking[:10],
                                "full_top10_core_judged": sum(
                                    key in test and qrels[q["qid"], key] >= 0
                                    for key in ranking[:10]
                                ),
                                "parsed": result["parsed"],
                            }
                        )
                    records.append(
                        {
                            "model": model,
                            "gallery": size,
                            "method": variant,
                            "language": lang,
                            "probe_weight": 0.15,
                            "per_query": per,
                            "ndcg@10": float(np.mean([r["ndcg@10"] for r in per])),
                            "p@5_known": float(np.mean([r["p@5_known"] for r in per])),
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    )
                    (out / "results.json").write_text(
                        json.dumps(
                            {
                                "source": str(SOURCE),
                                "records": records,
                                "scope": "core metrics use 140 judged test images; no claims of 10200-gallery recall; unjudged additions unknown",
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(model, size, variant, lang, records[-1]["ndcg@10"], flush=True)
        del encoder, service
        import gc

        gc.collect()


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
