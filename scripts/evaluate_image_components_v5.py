"""Correct-reference controlled image-image auxiliary ablations on cached pixels.

Reference images are up to four dev positives for each declared intent. They
are object-group disjoint from test images. This isolates the matching stage;
it is not an external reference acquisition success-rate experiment.
"""

import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.evaluate.relevance import retrieval_metrics  # noqa: E402
from curio.retrieve.dynamic_reference_runtime import make_sheet  # noqa: E402


def readl(path):
    return list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def z(a):
    return (a - a.mean()) / max(float(a.std()), 1e-6)


def main():
    out = ROOT / "artifacts/relevance_v5/image_components"
    out.mkdir(parents=True, exist_ok=True)
    core = readl(ROOT / "eval/relevance_v5/items.jsonl")
    queries = readl(ROOT / "eval/relevance_v5/queries.jsonl")
    qrels = {
        (r["item_id"], r["qid"]): r["grade"] for r in readl(ROOT / "eval/relevance_v5/qrels.jsonl")
    }
    ids = json.loads((ROOT / "artifacts/siglip/item_ids.json").read_text(encoding="utf-8"))
    lookup = {v: i for i, v in enumerate(ids)}
    test = [r["item_id"] for r in core if r["split"] == "test"]
    ti = [lookup[v] for v in test]
    features = {}
    for key, directory in [
        ("siglip2", "siglip"),
        ("production", "siglip_adapted"),
        ("dino", "dinov2"),
    ]:
        folder = ROOT / "artifacts" / directory
        mids = json.loads((folder / "item_ids.json").read_text(encoding="utf-8"))
        m = {v: i for i, v in enumerate(mids)}
        features[key] = unit(np.load(folder / "image_embeddings.npy")[[m[v] for v in ids]])
    fids = json.loads((ROOT / "artifacts/classical_cv/item_ids.json").read_text(encoding="utf-8"))
    fi = {v: i for i, v in enumerate(fids)}
    fo = [fi[v] for v in ids]
    raw = np.load(ROOT / "artifacts/classical_cv/features.npz")
    for key in ["raw_color", "gc_color", "raw_hog", "gc_shape", "raw_texture", "gc_texture"]:
        features[key] = unit(raw[key][fo])
    valid = raw["gc_valid"][fo].astype(bool)
    results = []
    saved = {}
    for q in queries:
        refs = sorted(
            r["item_id"] for r in core if r["split"] == "dev" and qrels[r["item_id"], q["qid"]] == 3
        )[:4]
        if not refs:
            results.append({"qid": q["qid"], "status": "unavailable_no_dev_reference"})
            continue
        ri = [lookup[v] for v in refs]
        scores = {name: matrix @ matrix[ri].mean(axis=0) for name, matrix in features.items()}
        # Equal normalized-score fusion; 0.25 is a declared sensitivity setting.
        methods = {name: value for name, value in scores.items()}
        methods["production_dino"] = 0.75 * z(scores["production"]) + 0.25 * z(scores["dino"])
        for key in ["raw_color", "gc_color", "raw_hog", "gc_shape", "raw_texture", "gc_texture"]:
            if key.startswith("gc_"):
                vr = [i for i in ri if valid[i]]
                auxiliary = (
                    features[key] @ features[key][vr].mean(axis=0) if vr else np.zeros(len(ids))
                )
                auxiliary = z(auxiliary)
                auxiliary[~valid] = 0
            else:
                auxiliary = z(scores[key])
            methods["production_" + key] = z(scores["production"]) + 0.25 * auxiliary
        grades = [qrels[v, q["qid"]] for v in test]
        results.append(
            {
                "qid": q["qid"],
                "status": "evaluated",
                "reference_ids": refs,
                "metrics": {
                    name: retrieval_metrics(value[ti], grades) for name, value in methods.items()
                },
            }
        )
        for name, value in methods.items():
            saved[q["qid"] + "/" + name] = value.astype("float32")
        if q["qid"] in {"v5_green_vase", "v5_dragon", "v5_blue_white"}:
            by_id = {r["item_id"]: r for r in core}
            for name in ["production", "production_raw_color", "production_dino"]:
                order = np.argsort(-methods[name][ti])[:5]
                make_sheet(
                    out / f"{q['qid']}_{name}.jpg",
                    f"{q['qid']} {name}; held-out test candidates",
                    [(test[i], ROOT / by_id[test[i]]["image_path"]) for i in order],
                    columns=5,
                )
    evaluated = [r for r in results if r["status"] == "evaluated"]
    summary = {
        name: {
            metric: float(np.mean([r["metrics"][name][metric] for r in evaluated]))
            for metric in ["ndcg@10", "p@5_known"]
        }
        for name in evaluated[0]["metrics"]
    }
    (out / "results.json").write_text(
        json.dumps(
            {
                "protocol": "up to 4 dev-positive references -> 140 held-out test images; whole-gallery scores cached; 0.25 exploratory fusion, no tuning on test",
                "evaluated_intents": len(evaluated),
                "summary": summary,
                "per_query": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(out / "scores.npz", **saved)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
