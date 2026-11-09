"""Search the image-only research bank without a text-image encoder."""

import argparse
import json
from pathlib import Path

import numpy as np

from curio.retrieve.attribute_cv import attribute_scores, parse_attributes, stable_order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--method",
        choices=["raw_hog", "raw_color", "raw_texture", "raw_all", "gc_all", "dino", "siglip"],
        default="gc_all",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/cv_research_20261104/fresh")
    )
    args = parser.parse_args()
    selection = json.loads((args.model_dir / "selection.json").read_text(encoding="utf8"))
    ids = json.loads(Path("artifacts/classical_cv/item_ids.json").read_text())
    probabilities = np.load(args.model_dir / f"{args.method}_predictions.npy")
    atoms = parse_attributes(args.query)
    score, unsupported = attribute_scores(probabilities, selection["classes"], atoms)
    result = {
        "method": args.method,
        "query": args.query,
        "recognized_atoms": atoms,
        "unsupported_atoms": unsupported,
        "scope": "Closed-vocabulary image attribute research; extra unrecognized words are not validated. Siglip method uses SigLIP image features; raw/gc/dino methods use no text-image model. Scores are not correctness probabilities.",
        "gallery_count": len(ids),
        "results": [],
    }
    if score is not None:
        order = stable_order(score, ids)[: args.top_k]
        result["results"] = [
            {"item_id": ids[i], "mean_log_attribute_score": float(score[i])} for i in order
        ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
