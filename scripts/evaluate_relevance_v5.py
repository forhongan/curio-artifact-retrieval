"""Compare frozen backbones and auxiliary channels on the reviewed v5 core.

All gallery vectors are reused. New adapter is loaded to verify a small cache
sample; only a genuine model change warrants a new image index.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.evaluate.relevance import retrieval_metrics  # noqa: E402
from curio.represent.encoders import Encoder  # noqa: E402
from curio.retrieve.attribute_cv import parse_attributes, attribute_scores  # noqa: E402
from curio.retrieve.visual_sets import visual_boundary  # noqa: E402

BASE = ROOT / "eval/relevance_v5"
OUT = ROOT / "artifacts/relevance_v5"
TEAM = ROOT / "research/teammate_review/resources/artifacts"
MODELS = {
    "clip": ROOT / "artifacts/clip",
    "siglip2": ROOT / "artifacts/siglip",
    "production_lora": ROOT / "artifacts/siglip_adapted",
    "team_comp_lora": TEAM / "siglip_comp_lora",
}


def readl(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s]


def dump(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def zscore(a):
    return (a - a.mean(axis=0)) / np.maximum(a.std(axis=0), 1e-6)


def mean_metrics(entries):
    keys = sorted({k for r in entries for k in r if isinstance(r[k], (int, float))})
    result = {k: float(np.mean([r[k] for r in entries if r.get(k) is not None])) for k in keys}
    if entries and "tp" in entries[0]:
        tp, fp, fn = (sum(r[k] for r in entries) for k in ["tp", "fp", "fn"])
        result.update(
            micro_set_precision=tp / (tp + fp) if tp + fp else None,
            micro_set_recall=tp / (tp + fn) if tp + fn else None,
            micro_set_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            macro_precision_defined_queries=sum(
                r.get("set_precision") is not None for r in entries
            ),
            empty_core_predictions=sum(r.get("selected_count") == 0 for r in entries),
        )
    return result


def choose_threshold(scores, grades, indices):
    # One shared boundary across queries, never a Top-K or percentile quota.
    trials = []
    for t in np.unique(np.quantile(scores[indices], np.linspace(0, 1, 161))):
        values = [
            retrieval_metrics(scores[indices, j], grades[indices, j], scores[indices, j] >= t)[
                "set_f1"
            ]
            for j in range(grades.shape[1])
        ]
        trials.append((np.mean(values), float(t)))
    return max(trials)[1]


def auxiliary_scores(ids, texts):
    """Existing train-only image heads; query bridge has finite coverage."""
    pred_dir = ROOT / "artifacts/cv_research_20261104/fresh"
    feature_ids = json.loads(
        (ROOT / "artifacts/classical_cv/item_ids.json").read_text(encoding="utf-8")
    )
    lookup = {v: i for i, v in enumerate(feature_ids)}
    order = np.array([lookup[v] for v in ids])
    output, coverage = {}, {}
    for name in [
        "raw_color",
        "gc_color",
        "raw_hog",
        "gc_shape",
        "raw_texture",
        "gc_texture",
        "dino",
    ]:
        head = np.load(pred_dir / f"{name}_heads.npz")
        predictions = np.load(pred_dir / f"{name}_predictions.npy")[order]
        classes = list(head["classes"])
        columns = []
        covered = []
        for text in texts:
            atoms = parse_attributes(text)
            score, missing = attribute_scores(predictions, classes, atoms)
            covered.append(not missing)
            columns.append(zscore(score) if score is not None else np.zeros(len(ids)))
        output[name] = np.column_stack(columns)
        coverage[name] = covered
    return output, coverage


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    queries = readl(BASE / "queries.jsonl")
    items = readl(BASE / "items.jsonl")
    qrels = {(r["item_id"], r["qid"]): r["grade"] for r in readl(BASE / "qrels.jsonl")}
    core_ids = [r["item_id"] for r in items]
    dev = np.array([i for i, r in enumerate(items) if r["split"] == "dev"])
    test = np.array([i for i, r in enumerate(items) if r["split"] == "test"])
    grades = np.array([[qrels[(i, q["qid"])] for q in queries] for i in core_ids])
    ids = json.loads((MODELS["siglip2"] / "item_ids.json").read_text(encoding="utf-8"))
    index = {v: i for i, v in enumerate(ids)}
    core = np.array([index[i] for i in core_ids])
    texts = [q["text_" + lang] for lang in ["en", "zh"] for q in queries]
    generic = "a photograph of a museum object"
    cache = OUT / "text_vectors.npz"
    text_hash = hashlib.sha256(
        json.dumps(texts + [generic], ensure_ascii=False).encode()
    ).hexdigest()
    if (
        cache.exists()
        and json.loads((OUT / "text_manifest.json").read_text(encoding="utf-8"))["text_hash"]
        == text_hash
    ):
        with np.load(cache) as saved:
            text_features = {k: saved[k] for k in saved.files}
    else:
        text_features = {}
        for name in ["clip", "siglip"]:
            encoder = Encoder(name)
            text_features[name] = encoder.encode_texts(texts + [generic])
            del encoder
            gc.collect()
        np.savez_compressed(cache, **text_features)
        dump(
            OUT / "text_manifest.json",
            {
                "text_hash": text_hash,
                "texts": texts + [generic],
                "siglip_revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
                "text_tower": "all adapters target vision modules only",
            },
        )
    # Verify the downloaded adapter itself and its shared image index before using it.
    verification_file = OUT / "team_cache_verification.json"
    if not verification_file.exists():
        adapter = TEAM / "training/siglip_lora_comp/best_adapter"
        encoder = Encoder("siglip", adapter_path=adapter)
        catalog = {r["item_id"]: r for r in readl(ROOT / "data/snapshots/collection_v2.jsonl")}
        team_ids = json.loads(
            (MODELS["team_comp_lora"] / "item_ids.json").read_text(encoding="utf-8")
        )
        team_vectors = np.load(MODELS["team_comp_lora"] / "image_embeddings.npy")
        chosen = np.linspace(0, len(team_ids) - 1, 16, dtype=int)
        actual = encoder.encode_images([ROOT / catalog[team_ids[i]]["image_path"] for i in chosen])
        cos = (actual * unit(team_vectors[chosen])).sum(axis=1)
        text_cos = (encoder.encode_texts(texts[:4]) * text_features["siglip"][:4]).sum(axis=1)
        digest = hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
        expected = json.loads(
            (MODELS["team_comp_lora"] / "manifest.json").read_text(encoding="utf-8")
        )["adapter_hash"]
        if digest != expected or cos.min() < 0.995 or text_cos.min() < 0.999:
            raise ValueError("Downloaded adapter/index contract verification failed")
        dump(
            verification_file,
            {
                "adapter_sha256": digest,
                "image_ids": [team_ids[i] for i in chosen],
                "cache_cosines": cos.tolist(),
                "text_cosines": text_cos.tolist(),
                "status": "passed",
                "full_gallery_reencoded": False,
                "sample": 16,
            },
        )
        del encoder
        gc.collect()
    aux, coverage = auxiliary_scores(ids, texts)
    dino_ids = json.loads((ROOT / "artifacts/dinov2/item_ids.json").read_text(encoding="utf-8"))
    lookup = {v: i for i, v in enumerate(dino_ids)}
    dino = unit(np.load(ROOT / "artifacts/dinov2/image_embeddings.npy")[[lookup[v] for v in ids]])
    # Leave the image itself out; use DINO only as a soft visual-neighborhood cue.
    with threadpool_limits(limits=1):
        neighbours = (
            NearestNeighbors(n_neighbors=9, metric="cosine", n_jobs=1)
            .fit(dino)
            .kneighbors(dino, return_distance=False)[:, 1:]
        )
    records = []
    all_scores = {}
    thresholds = {}
    selection = {}
    for name, path in MODELS.items():
        mids = json.loads((path / "item_ids.json").read_text(encoding="utf-8"))
        lookup = {v: i for i, v in enumerate(mids)}
        vectors = unit(np.load(path / "image_embeddings.npy")[[lookup[v] for v in ids]])
        text = text_features["clip" if name == "clip" else "siglip"]
        start = time.perf_counter()
        with threadpool_limits(limits=1):
            dense = vectors @ text[:-1].T
            generic_score = vectors @ text[-1]
        methods = {"dense": dense}
        for channel, aux_score in aux.items():
            choices = []
            for weight in [0.0, 0.1, 0.25, 0.5]:
                fused = zscore(dense) + weight * aux_score
                vals = [
                    retrieval_metrics(fused[core[dev], j], grades[dev, j % len(queries)])["ndcg@10"]
                    for j in range(len(texts))
                ]
                choices.append((np.mean(vals), -weight, fused))
            _, negative_weight, fused = max(choices, key=lambda r: r[:2])
            selection[name + "/" + channel] = {
                "weight": -negative_weight,
                "dev_ndcg": max(c[0] for c in choices),
            }
            methods["aux_" + channel] = fused
        for weight in [0.1, 0.25]:
            methods[f"dino_graph_{weight}"] = (1 - weight) * zscore(dense) + weight * zscore(
                dense[neighbours].mean(axis=1)
            )
        for method, scores in methods.items():
            for li, lang in enumerate(["en", "zh"]):
                cols = slice(li * len(queries), (li + 1) * len(queries))
                score = scores[:, cols]
                cscore = score[core]
                threshold = choose_threshold(cscore, grades, dev)
                key = f"{name}/{method}/{lang}"
                thresholds[key] = threshold
                all_scores[key] = score.astype("float32")
                per = []
                for j, q in enumerate(queries):
                    mask = score[:, j] >= threshold
                    m = retrieval_metrics(cscore[test, j], grades[test, j], mask[core[test]])
                    per.append({"qid": q["qid"], **m, "full_gallery_selected": int(mask.sum())})
                records.append(
                    {
                        "model": name,
                        "method": method,
                        "language": lang,
                        "metrics": mean_metrics(per),
                        "per_query": per,
                    }
                )
        # Old adaptive GMM versus simple dev-supervised calibration on image scores.
        for li, lang in enumerate(["en", "zh"]):
            cols = slice(li * len(queries), (li + 1) * len(queries))
            ds = dense[:, cols]
            margin = ds - generic_score[:, None]
            z = zscore(ds)
            feature = np.stack([ds, margin, z], axis=-1)
            devx = feature[core[dev]].reshape(-1, 3)
            devg = grades[dev].reshape(-1)
            model = LogisticRegression(C=1, max_iter=1000).fit(
                devx[devg >= 0], devg[devg >= 0] == 3
            )
            p = expit(feature @ model.coef_[0] + model.intercept_[0])
            t = choose_threshold(p[core], grades, dev)
            thresholds[f"{name}/calibrated/{lang}"] = t
            dump(
                OUT / f"boundary_{name}_{lang}.json",
                {
                    "coefficients": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "threshold": t,
                    "feature_order": ["cosine", "generic_margin", "gallery_zscore"],
                    "fit_split": "60 reviewed dev images; same 20 semantic intents; unknowns excluded",
                    "model": name,
                    "index_manifest_sha256": hashlib.sha256(
                        (path / "manifest.json").read_bytes()
                    ).hexdigest(),
                },
            )
            for method in ["adaptive_gmm", "calibrated"]:
                per = []
                for j, q in enumerate(queries):
                    yes = (
                        visual_boundary(ds[:, j], generic_score)[0]
                        if method == "adaptive_gmm"
                        else p[:, j] >= t
                    )
                    per.append(
                        {
                            "qid": q["qid"],
                            **retrieval_metrics(
                                ds[core[test], j], grades[test, j], yes[core[test]]
                            ),
                            "full_gallery_selected": int(yes.sum()),
                        }
                    )
                records.append(
                    {
                        "model": name,
                        "method": method,
                        "language": lang,
                        "metrics": mean_metrics(per),
                        "per_query": per,
                    }
                )
            all_scores[f"{name}/calibrated/{lang}"] = p.astype("float32")
        print(name, "scored", round(time.perf_counter() - start, 2), "seconds", flush=True)
    np.savez_compressed(OUT / "scores.npz", **all_scores)
    dump(OUT / "thresholds.json", thresholds)
    dump(
        OUT / "results.json",
        {
            "protocol_freeze": json.loads((BASE / "freeze.json").read_text(encoding="utf-8")),
            "gallery": len(ids),
            "core_test": len(test),
            "queries": len(queries),
            "records": records,
            "auxiliary_selection": selection,
            "auxiliary_query_coverage": coverage,
            "limitations": [
                "Reused historical test images, AI image/source relevance rather than human expert gold.",
                "Unknowns are excluded with coverage and pessimistic/optimistic precision bounds.",
                "Set recall applies to 140-image exhaustively judged core, not to the other full-gallery images.",
                "Supervised boundary uses same intents with disjoint development images; not unseen-query generalization.",
            ],
        },
    )
    lines = [
        "# Revised core comparison",
        "",
        "140 test images; 20 intents. All gallery scores computed from cached image vectors.",
        "Unknown judgments excluded; check coverage and set precision bounds in JSON.",
        "",
        "| Model | Method | Lang | nDCG@10 | P@5 known | set P | set R | set F1 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        m = r["metrics"]
        lines.append(
            f"| {r['model']} | {r['method']} | {r['language']} | {m['ndcg@10']:.4f} | {m['p@5_known']:.4f} | {m['set_precision']:.4f} | {m['set_recall']:.4f} | {m['set_f1']:.4f} |"
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    with threadpool_limits(limits=1):
        run()
