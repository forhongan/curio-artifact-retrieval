"""Evaluate the existing production query rewrite, with its frozen image tower."""

import json

import numpy as np

from curio.query.parser import parse_query
from curio.represent.encoders import Encoder
from run_cv_research import ROOT, BASE, dump, evaluate_method


def main():
    out = ROOT / "artifacts/cv_research_20261104/fresh"
    labels = json.loads((out / "test_annotations.json").read_text())
    queries = json.loads((BASE / "queries.json").read_text(encoding="utf8"))
    index = ROOT / "artifacts/siglip_adapted"
    ids = json.loads((index / "item_ids.json").read_text())
    lookup = {key: i for i, key in enumerate(ids)}
    vectors = np.load(index / "image_embeddings.npy")[[lookup[r["item_id"]] for r in labels]]
    encoder = Encoder(
        "siglip", adapter_path=ROOT / "artifacts/training/siglip_lora_random/best_adapter"
    )
    results = json.loads((out / "test_results.json").read_text())
    rewrites = {}
    for lang in ("zh", "en"):
        text = [parse_query(q[lang]).rewritten for q in queries]
        rewrites[lang] = dict(zip((q["qid"] for q in queries), text))
        matrix = encoder.encode_texts(text) @ vectors.T
        scores = {q["qid"]: matrix[i] for i, q in enumerate(queries)}
        results[f"production_rewritten_{lang}"] = evaluate_method(
            scores, [r["item_id"] for r in labels], labels, queries
        )
        print(lang, results[f"production_rewritten_{lang}"]["mean_all_intents"], flush=True)
    dump(out / "test_results.json", results)
    dump(
        out / "production_rewritten_queries.json",
        {
            "warning": "Existing frozen production rewrite+adapted dense image similarity, with no probe or metadata ranking. Added as transparent baseline-control check after main evaluation; not used for any selection.",
            "queries": rewrites,
        },
    )


if __name__ == "__main__":
    main()
