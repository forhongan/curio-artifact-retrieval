"""Frozen-benchmark backbone controls, encoding cost and prefilter recall audit."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
import platform
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageOps
import torch
from threadpoolctl import threadpool_limits
import transformers

from curio.query.parser import parse_query
from curio.represent.encoders import Encoder
from curio.represent.image_encoder import DinoEncoder
from curio.retrieve.attribute_cv import (
    attribute_scores,
    parse_attributes,
    predict_heads,
    rrf_scores,
    stable_order,
)
from run_cv_research import (
    BASE,
    ROOT,
    dump,
    evaluate_method,
    fit_model,
    make_scores,
    merge_fresh,
    metrics,
    relevance,
    rows,
    sha,
)

OLD = ROOT / "artifacts/cv_research_20261104/fresh"
OUT = ROOT / "artifacts/cv_backbones_20261104"
MODELS = {"clip": "clip", "siglip2": "siglip", "lora": "siglip_adapted", "dino": "dinov2"}
SIGLIP1_ID = "google/siglip-base-patch16-224"
SIGLIP1_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def vectors(name, target_ids):
    directory = OUT / "siglip1_index" if name == "siglip1" else ROOT / "artifacts" / MODELS[name]
    ids = read(directory / "item_ids.json")
    lookup = {key: i for i, key in enumerate(ids)}
    return np.load(directory / "image_embeddings.npy")[[lookup[key] for key in target_ids]]


def load_encoder(name):
    if name == "siglip1":
        return Siglip1Encoder()
    if name == "dino":
        return DinoEncoder()
    return Encoder(
        "clip" if name == "clip" else "siglip",
        adapter_path=ROOT / "artifacts/training/siglip_lora_random/best_adapter"
        if name == "lora"
        else None,
    )


class Siglip1Encoder(Encoder):
    """Experiment-only model; keeps the production SigLIP2 alias untouched."""

    def __init__(self):
        self.name = "siglip1"
        self.model_id = SIGLIP1_ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(4)
        self.processor = transformers.AutoProcessor.from_pretrained(
            self.model_id, revision=SIGLIP1_REVISION, use_fast=False
        )
        self.model = (
            transformers.AutoModel.from_pretrained(self.model_id, revision=SIGLIP1_REVISION)
            .to(self.device)
            .eval()
        )
        self.revision = SIGLIP1_REVISION
        self.adapter_path = None
        self.adapter_hash = None


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protocol.json"
    if target.exists():
        return
    dump(
        target,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "parent_protocol_sha256": sha(BASE / "protocol.json"),
            "parent_selection_sha256": sha(OLD / "selection.json"),
            "intent_sha256": sha(BASE / "queries.json"),
            "accuracy": "Same 21 intents, dev48/test96 AI pixel labels. Existing test results were previously exposed; this is a fixed supplementary control, not a new blind test.",
            "attribute_controls": "CLIP and production LoRA image features; identical 21 classes, train218, C=.1/1/10 selected on dev only, same LR and scoring as parent.",
            "dense_controls": "raw zh/en and production rule-expanded zh/en, all models; no metadata or attribute-head reranking",
            "prefilter": {
                "methods": ["raw_all", "gc_all", "dino", "fusion"],
                "fractions": [0.1, 0.25, 0.5, 1.0],
                "reranker": "production LoRA, existing rule-expanded English",
                "unsupported": "pass through all candidates; never hard-reject an unsupported query",
                "selection": "smallest fraction with dev macro known-positive retention>=.95 and min-query retention>=.8; ties smaller fraction; no test tuning",
                "quality_scope": "Exhaustively labeled dev48/test96 galleries only, not full4692 relevance",
            },
            "timing": "RTX3070; batch16; staged256 hash-selected images; full4692 actual native encode_images call; 32 single-image calls after warmup. CUDA synchronize each boundary. CPU file cache is warm/uncontrolled, not cold storage throughput. No network/model download in timing.",
        },
    )


def accuracy(names=("clip", "siglip2", "lora"), head_names=("clip", "lora")):
    queries = read(BASE / "queries.json")
    classes = read(OLD / "selection.json")["classes"]
    ids = read(ROOT / "artifacts/classical_cv/item_ids.json")
    index = {key: i for i, key in enumerate(ids)}
    train = [
        r for r in rows(ROOT / "eval/visual_labels/annotations.jsonl") if r["split"] == "train"
    ]
    dev = [r for r in merge_fresh() if r["split"] == "dev"]
    selection = read(OUT / "head_selection.json") if (OUT / "head_selection.json").exists() else {}
    # All new head selection precedes loading held-out labels.
    with threadpool_limits(limits=1):
        for name in head_names:
            if name in selection:
                raise FileExistsError(f"Frozen head selection already exists: {name}")
            x = vectors(name, ids)
            path = OUT / f"{name}_heads.npz"
            selection[name] = fit_model(x, train, dev, classes, index, path)
            head = np.load(path)
            np.save(
                OUT / f"{name}_predictions.npy",
                predict_heads(x, head["coefficients"], head["intercepts"]).astype(np.float32),
            )
    dump(OUT / "head_selection.json", selection)
    test = read(OLD / "test_annotations.json")
    results = read(OUT / "accuracy.json") if (OUT / "accuracy.json").exists() else {}
    for name in names:
        encoder = load_encoder(name)
        for lang in ("zh", "en"):
            for rewrite in (False, True):
                texts = [parse_query(q[lang]).rewritten if rewrite else q[lang] for q in queries]
                embeddings = encoder.encode_texts(texts)
                key = f"{name}_{'rewritten' if rewrite else 'raw'}_{lang}"
                np.save(OUT / f"{key}_text.npy", embeddings)
                results[key] = {}
                for split, labels in (("dev", dev), ("test", test)):
                    matrix = embeddings @ vectors(name, [r["item_id"] for r in labels]).T
                    scores = {q["qid"]: matrix[j] for j, q in enumerate(queries)}
                    results[key][split] = evaluate_method(
                        scores, [r["item_id"] for r in labels], labels, queries
                    )
                print(key, results[key]["test"]["mean_all_intents"], flush=True)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    for name in head_names:
        p = np.load(OUT / f"{name}_predictions.npy")
        results[f"{name}_attributes"] = {}
        for split, labels in (("dev", dev), ("test", test)):
            scores = make_scores(p[[index[r["item_id"]] for r in labels]], classes, queries)[0]
            results[f"{name}_attributes"][split] = evaluate_method(
                scores, [r["item_id"] for r in labels], labels, queries
            )
    dump(OUT / "accuracy.json", results)


def prefilters():
    queries = read(BASE / "queries.json")
    classes = read(OLD / "selection.json")["classes"]
    all_ids = read(ROOT / "artifacts/classical_cv/item_ids.json")
    lookup = {key: i for i, key in enumerate(all_ids)}
    probabilities = {
        name: np.load(OLD / f"{name}_predictions.npy") for name in ("raw_all", "gc_all", "dino")
    }
    text = np.load(OUT / "lora_rewritten_en_text.npy")
    full_dense = text @ vectors("lora", all_ids).T
    report = {}
    selected = {}
    for split in ("dev", "test"):
        if split == "test":
            dump(OUT / "prefilter_selection.json", selected)
        labels = [r for r in merge_fresh() if r["split"] == split]
        ids = [r["item_id"] for r in labels]
        take = [lookup[key] for key in ids]
        coarse = {
            name: make_scores(p[take], classes, queries)[0] for name, p in probabilities.items()
        }
        coarse["fusion"] = {
            q["qid"]: rrf_scores(coarse["dino"][q["qid"]], coarse["raw_all"][q["qid"]], ids, 0.25)
            if coarse["dino"][q["qid"]] is not None
            else None
            for q in queries
        }
        records = {}
        for name, scores in coarse.items():
            for fraction in (0.1, 0.25, 0.5, 1.0):
                per_query = []
                for j, q in enumerate(queries):
                    s = scores[q["qid"]]
                    k = max(10, math.ceil(len(ids) * fraction))
                    candidates = stable_order(s, ids)[:k] if s is not None else np.arange(len(ids))
                    fine = full_dense[j, take]
                    order = candidates[stable_order(fine[candidates], np.array(ids)[candidates])]
                    rel = relevance(labels, q["atoms"])
                    positives = (rel == 1).sum()
                    per_query.append(
                        {
                            "qid": q["qid"],
                            "supported": s is not None,
                            "candidates": len(candidates),
                            "retention": float((rel[candidates] == 1).sum() / positives)
                            if positives
                            else None,
                            "lost_known_positives": int(
                                (rel == 1).sum() - (rel[candidates] == 1).sum()
                            ),
                            "positive_count": int(positives),
                            "discarded_fraction": 1 - len(candidates) / len(ids),
                            **metrics(order, rel),
                        }
                    )
                known = [r["retention"] for r in per_query if r["retention"] is not None]
                record = {
                    "fraction": fraction,
                    "K": min(len(ids), max(10, math.ceil(len(ids) * fraction))),
                    "macro_retention": float(np.mean(known)),
                    "minimum_retention": min(known),
                    "total_lost_positives": sum(r["lost_known_positives"] for r in per_query),
                    "mean_discarded_fraction": float(
                        np.mean([r["discarded_fraction"] for r in per_query])
                    ),
                    "P@5": float(np.mean([r["P@5"] for r in per_query])),
                    "nDCG@10_known": float(np.mean([r["nDCG@10_known"] for r in per_query])),
                    "per_query": per_query,
                }
                records[f"{name}_{fraction}"] = record
                if (
                    split == "dev"
                    and name not in selected
                    and record["macro_retention"] >= 0.95
                    and record["minimum_retention"] >= 0.8
                ):
                    selected[name] = {
                        "fraction": fraction,
                        "dev_macro_retention": record["macro_retention"],
                        "dev_minimum_retention": record["minimum_retention"],
                    }
        report[split] = records
    # Cached full-gallery latency: same query text is already encoded for both paths.
    timings = {}
    gallery = vectors("lora", all_ids)
    with threadpool_limits(limits=1):
        for name in ("dense", "raw_all", "gc_all", "dino", "fusion"):
            samples = []
            for repeat in range(7):
                for j, q in enumerate(queries):
                    start = time.perf_counter()
                    candidates = np.arange(len(all_ids))
                    if name != "dense":
                        pnames = ("dino", "raw_all") if name == "fusion" else (name,)
                        scores = [
                            make_scores(probabilities[pn], classes, [q])[0][q["qid"]]
                            for pn in pnames
                        ]
                        score = (
                            rrf_scores(*scores, all_ids, 0.25)
                            if name == "fusion" and scores[0] is not None
                            else scores[0]
                        )
                        if score is not None:
                            candidates = stable_order(score, all_ids)[
                                : math.ceil(len(all_ids) * selected[name]["fraction"])
                            ]
                    fine = gallery[candidates] @ text[j]
                    stable_order(fine, np.array(all_ids)[candidates])[:10]
                    if repeat >= 2:
                        samples.append((time.perf_counter() - start) * 1000)
            timings[name] = {
                "p50_ms": float(np.median(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
                "raw_ms": samples,
            }
    dump(
        OUT / "prefilters.json",
        {"quality": report, "cached_latency_excluding_shared_text_encoder": timings},
    )
    print("prefilter_selection", selected, flush=True)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def prefilter_latency():
    """Interleave methods to reduce clock/cache/order bias; time actual parsing."""
    queries = read(BASE / "queries.json")
    classes = read(OLD / "selection.json")["classes"]
    ids = read(ROOT / "artifacts/classical_cv/item_ids.json")
    ids_array = np.array(ids)
    bank = vectors("lora", ids)
    text = np.load(OUT / "lora_rewritten_en_text.npy")
    p = {name: np.load(OLD / f"{name}_predictions.npy") for name in ("raw_all", "gc_all", "dino")}
    methods = [("dense", 1.0)] + [(n, f) for n in (*p, "fusion") for f in (0.1, 0.25, 0.5, 1.0)]
    samples = {f"{n}_{f}": [] for n, f in methods}
    rng = np.random.default_rng(240)
    with threadpool_limits(limits=1):
        for repeat in range(9):
            for j, q in enumerate(queries):
                for index in rng.permutation(len(methods)):
                    name, fraction = methods[index]
                    tick = time.perf_counter()
                    if name == "dense":
                        result = bank @ text[j]
                        stable_order(result, ids)[:10]
                    else:
                        atoms = parse_attributes(q["en"])
                        names = ("dino", "raw_all") if name == "fusion" else (name,)
                        scores = [attribute_scores(p[n], classes, atoms)[0] for n in names]
                        score = (
                            rrf_scores(*scores, ids, 0.25)
                            if name == "fusion" and scores[0] is not None
                            else scores[0]
                        )
                        if score is None or fraction == 1.0:
                            result = bank @ text[j]
                            stable_order(result, ids)[:10]
                        else:
                            keep = stable_order(score, ids)[: math.ceil(len(ids) * fraction)]
                            result = bank[keep] @ text[j]
                            stable_order(result, ids_array[keep])[:10]
                    if repeat >= 2:
                        samples[f"{name}_{fraction}"].append((time.perf_counter() - tick) * 1000)
    dump(
        OUT / "prefilter_latency_interleaved.json",
        {
            "scope": "4692 cached image vectors; text vector shared and excluded; actual English attribute parser, coarse scoring/sort and candidate-copy plus dense cosine/sort; random interleaving, CPU single BLAS thread; no HTTP",
            "results": {key: stats(value) for key, value in samples.items()},
        },
    )


def stats(values):
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "raw_ms": values,
    }


def encoding(names=None):
    pixels = rows(BASE / "pixels.jsonl")
    sample = sorted(
        pixels, key=lambda r: hashlib.sha256(("encoding-cost" + r["item_id"]).encode()).hexdigest()
    )[:256]
    dump(OUT / "encoding_sample.json", sample)
    report = {
        "hardware": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "batch_size": 16,
        "cpu_threads": 4,
        "models": {},
    }
    if (OUT / "encoding.json").exists():
        report = read(OUT / "encoding.json")
    for name in names or MODELS:
        start = time.perf_counter()
        encoder = load_encoder(name)
        sync()
        load_seconds = time.perf_counter() - start
        paths = [ROOT / r["image_path"] for r in sample]
        encoder.encode_images(paths[:16])
        stage = {
            key: []
            for key in ("decode", "processor", "host_to_device", "forward", "normalize_to_cpu")
        }
        with torch.inference_mode():
            for begin in range(0, len(paths), 16):
                tick = time.perf_counter()
                images = []
                for path in paths[begin : begin + 16]:
                    with Image.open(path) as im:
                        images.append(ImageOps.exif_transpose(im).convert("RGB"))
                stage["decode"].append((time.perf_counter() - tick) * 1000)
                tick = time.perf_counter()
                inputs = encoder.processor(images=images, return_tensors="pt")
                stage["processor"].append((time.perf_counter() - tick) * 1000)
                tick = time.perf_counter()
                inputs = inputs.to(encoder.device)
                sync()
                stage["host_to_device"].append((time.perf_counter() - tick) * 1000)
                tick = time.perf_counter()
                context = encoder.autocast() if name != "dino" else nullcontext()
                with context:
                    features = (
                        encoder.model(**inputs).last_hidden_state[:, 0]
                        if name == "dino"
                        else encoder.model.get_image_features(**inputs)
                    )
                sync()
                stage["forward"].append((time.perf_counter() - tick) * 1000)
                tick = time.perf_counter()
                torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()
                sync()
                stage["normalize_to_cpu"].append((time.perf_counter() - tick) * 1000)
        single = []
        for path in paths[:32]:
            sync()
            tick = time.perf_counter()
            encoder.encode_images([path], batch_size=1)
            sync()
            single.append((time.perf_counter() - tick) * 1000)
        # Actual full-gallery encode, no extrapolation; includes decode, preprocessing,
        # tower, normalization, transfer and concatenation; excludes loading/save.
        torch.cuda.reset_peak_memory_stats()
        sync()
        tick = time.perf_counter()
        encoded = encoder.encode_images([ROOT / r["image_path"] for r in pixels], batch_size=16)
        sync()
        elapsed = time.perf_counter() - tick
        tick = time.perf_counter()
        target = OUT / f"{name}_timed_embeddings.npy"
        np.save(target, encoded)
        save_seconds = time.perf_counter() - tick
        if name == "siglip1":
            index_dir = OUT / "siglip1_index"
            index_dir.mkdir(exist_ok=True)
            np.save(index_dir / "image_embeddings.npy", encoded)
            dump(index_dir / "item_ids.json", [r["item_id"] for r in pixels])
            dump(index_dir / "manifest.json", encoder.metadata)
            old = None
            similarity = None
        else:
            old = vectors(name, [r["item_id"] for r in pixels])
            similarity = np.sum(encoded * old, axis=1)
        result = {
            "metadata": encoder.metadata,
            "precision": "FP32"
            if name == "dino"
            else "FP16 autocast; FP32 weights and normalized outputs",
            "input_size": list(inputs["pixel_values"].shape[1:]),
            "model_load_seconds_local_cache": load_seconds,
            "stages_256": {
                key: {"total_seconds": sum(value) / 1000, **stats(value)}
                for key, value in stage.items()
            },
            "single_image_end_to_end": stats(single),
            "full_gallery": {
                "count": len(pixels),
                "seconds": elapsed,
                "images_per_second": len(pixels) / elapsed,
                "amortized_ms_per_image": elapsed * 1000 / len(pixels),
                "save_seconds": save_seconds,
                "bytes": target.stat().st_size,
                "dimension": encoded.shape[1],
                "peak_allocated_gpu_MiB_including_model": torch.cuda.max_memory_allocated() / 2**20,
            },
            "cached_index_cosine_agreement": {
                "min": float(similarity.min()),
                "mean": float(similarity.mean()),
            }
            if similarity is not None
            else None,
        }
        if name == "dino":
            # Same-backbone precision control, not used to change frozen rankings.
            times = []
            with torch.autocast("cuda", dtype=torch.float16):
                encoder.encode_images(paths[:16])
                for _ in range(3):
                    sync()
                    tick = time.perf_counter()
                    half = encoder.encode_images(paths)
                    sync()
                    times.append((time.perf_counter() - tick) * 1000)
            original = vectors(name, [r["item_id"] for r in sample])
            result["FP16_sample256_control"] = {
                **stats(times),
                "mean_cosine_to_FP32": float(np.sum(half * original, axis=1).mean()),
            }
        report["models"][name] = result
        dump(OUT / "encoding.json", report)
        print("encoding", name, result["full_gallery"], flush=True)
        del encoder, encoded, old, inputs, features
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["accuracy", "prefilters", "latency", "encoding", "siglip1", "all"]
    )
    args = parser.parse_args()
    protocol()
    if args.stage == "siglip1":
        dump(
            OUT / "siglip1_addendum.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "Actual first-generation SigLIP B/16-224 versus existing SigLIP2 B/16-224. Added fixed supplementary control after observing primary controls; no test-driven hyperparameter tuning.",
                "model_id": SIGLIP1_ID,
                "revision": SIGLIP1_REVISION,
                "attribute_training": "Same train218/dev48/classes and C grid; dev-only selection",
            },
        )
        encoding(("siglip1",))
        accuracy(("siglip1",), ("siglip1",))
    if args.stage == "latency":
        prefilter_latency()
    for key, action in (("accuracy", accuracy), ("prefilters", prefilters), ("encoding", encoding)):
        if args.stage in (key, "all"):
            action()
