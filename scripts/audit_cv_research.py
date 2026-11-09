"""Holdout stress tests, latency accounting and descriptive paired comparisons."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
from threadpoolctl import threadpool_limits

from curio.represent.classical_cv import extract_classical
from curio.represent.image_encoder import DinoEncoder
from curio.represent.encoders import Encoder
from curio.retrieve.attribute_cv import (
    attribute_scores,
    parse_attributes,
    predict_heads,
    stable_order,
)
from run_cv_research import BASE, ROOT, dump, evaluate_method, make_scores, rows, sha


def variants_for(task):
    row, source, out = task
    cv2.setNumThreads(1)
    with Image.open(ROOT / source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    original = extract_classical(image)
    variants = {"brightness065": ImageEnhance.Brightness(image).enhance(0.65)}
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=25)
    variants["jpeg25"] = Image.open(BytesIO(buffer.getvalue())).convert("RGB")
    if original["diagnostics"]["grabcut_accepted"]:
        mask = Image.fromarray(original["mask"] * 255).resize(image.size, Image.Resampling.NEAREST)
        variants["white_background"] = Image.composite(
            image, Image.new("RGB", image.size, "white"), mask
        )
    features = {}
    diagnostics = {}
    paths = {}
    for name, im in variants.items():
        target = Path(out) / name / f"{row['item_id']}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        im.save(target)
        result = extract_classical(im)
        features[name] = result["features"]
        diagnostics[name] = {**result["diagnostics"], "file_sha256": sha(target)}
        paths[name] = str(target)
    return {
        "item_id": row["item_id"],
        "features": features,
        "diagnostics": diagnostics,
        "paths": paths,
    }


def evaluate_stress(out):
    selection = json.loads((out / "selection.json").read_text())
    test = json.loads((out / "test_annotations.json").read_text())
    queries = json.loads((BASE / "queries.json").read_text(encoding="utf8"))
    pixel_map = {r["item_id"]: r for r in rows(BASE / "pixels.jsonl")}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                variants_for,
                ((r, pixel_map[r["item_id"]]["image_path"], out / "stress_images") for r in test),
            )
        )
    report = {
        "classical_extraction_seconds": time.perf_counter() - start,
        "conditions": {},
        "limits": "Original labels retained under assumed invariance. Brightness can change named color; background compositing uses unvalidated GrabCut masks. Conditional background subset is scored against its own original-image baseline, never directly against96-image scores.",
    }
    dino = DinoEncoder()
    for condition in ("brightness065", "jpeg25", "white_background"):
        available = [r for r in results if condition in r["features"]]
        ids = [r["item_id"] for r in available]
        labels = [r for r in test if r["item_id"] in set(ids)]
        assert [r["item_id"] for r in labels] == ids
        scores = {}
        features = {}
        for name in ("raw_all", "gc_all"):
            x = np.stack([r["features"][condition][name] for r in available])
            head = np.load(out / f"{name}_heads.npz")
            features[name] = x
            p = predict_heads(x, head["coefficients"], head["intercepts"])
            scores[name] = make_scores(p, selection["classes"], queries)[0]
        x = dino.encode_images([r["paths"][condition] for r in available])
        head = np.load(out / "dino_heads.npz")
        scores["dino"] = make_scores(
            predict_heads(x, head["coefficients"], head["intercepts"]),
            selection["classes"],
            queries,
        )[0]
        np.savez_compressed(out / f"stress_{condition}_features.npz", dino=x, **features)
        report["conditions"][condition] = {
            "gallery_count": len(ids),
            "ids": ids,
            "gc_accepted": sum(r["diagnostics"][condition]["grabcut_accepted"] for r in available),
            "results": {n: evaluate_method(s, ids, labels, queries) for n, s in scores.items()},
        }
    del dino
    siglip = Encoder("siglip")
    for condition, record in report["conditions"].items():
        ids = record["ids"]
        paths = [r["paths"][condition] for r in results if condition in r["paths"]]
        vectors = siglip.encode_images(paths)
        labels = [r for r in test if r["item_id"] in set(ids)]
        head = np.load(out / "siglip_heads.npz")
        scores = make_scores(
            predict_heads(vectors, head["coefficients"], head["intercepts"]),
            selection["classes"],
            queries,
        )[0]
        record["results"]["siglip"] = evaluate_method(scores, ids, labels, queries)
        for lang in ("zh", "en"):
            matrix = np.load(out / f"siglip_dot_{lang}_text.npy") @ vectors.T
            record["results"][f"siglip_dot_{lang}"] = evaluate_method(
                {q["qid"]: matrix[j] for j, q in enumerate(queries)}, ids, labels, queries
            )
        # Identical original-image candidate set for conditional-background comparisons.
        all_ids = json.loads((ROOT / "artifacts/classical_cv/item_ids.json").read_text())
        lookup = {key: i for i, key in enumerate(all_ids)}
        record["original_same_gallery"] = {}
        for name in ("raw_all", "gc_all", "dino", "siglip"):
            prob = np.load(out / f"{name}_predictions.npy")[[lookup[key] for key in ids]]
            record["original_same_gallery"][name] = evaluate_method(
                make_scores(prob, selection["classes"], queries)[0], ids, labels, queries
            )
    dump(out / "stress_results.json", report)
    # Deterministic pixel QA of masks; neither captions nor predictions are visible.
    chosen = sorted(
        test, key=lambda r: hashlib.sha256(("maskqa" + r["item_id"]).encode()).hexdigest()
    )[:12]
    sheet = Image.new("RGB", (1200, 1200), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(chosen):
        im = Image.open(ROOT / pixel_map[row["item_id"]]["image_path"]).convert("RGB")
        result = extract_classical(im)
        mask = Image.fromarray(result["mask"] * 255).convert("RGB")
        x, y = (i % 3) * 400, (i // 3) * 300
        for offset, tile in ((0, im), (200, mask)):
            thumb = ImageOps.contain(tile, (195, 260))
            sheet.paste(thumb, (x + offset + (195 - thumb.width) // 2, y + 30))
        draw.text((x + 8, y + 8), row["alias"], fill="black", font_size=16)
    sheet.save(out / "mask_qa.jpg", quality=94)
    return report


def benchmark(out):
    selection = json.loads((out / "selection.json").read_text())
    queries = json.loads((BASE / "queries.json").read_text(encoding="utf8"))
    all_ids = json.loads((ROOT / "artifacts/classical_cv/item_ids.json").read_text())
    # Exact parser-vs-frozen semantic intent check, no oracle attribute input.
    mismatches = []
    for q in queries:
        for lang in ("en", "zh"):
            if set(parse_attributes(q[lang])) != set(map(tuple, q["atoms"])):
                mismatches.append({"qid": q["qid"], "language": lang})
    assert not mismatches
    names = (selection["chosen_classical"], "gc_all", "dino", "siglip")
    p = {name: np.load(out / f"{name}_predictions.npy") for name in set(names)}
    timing = {}
    with threadpool_limits(limits=1):
        for name in p:
            values = []
            for iteration in range(12):
                for q in queries:
                    start = time.perf_counter()
                    atoms = parse_attributes(q["zh"])
                    score, _ = attribute_scores(p[name], selection["classes"], atoms)
                    if score is not None:
                        stable_order(score, all_ids)[:10]
                    if iteration >= 2:
                        values.append((time.perf_counter() - start) * 1000)
            timing[name] = {
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "observations": len(values),
                "scope": "fixed-vocabulary parse + cached image attribute prediction scoring + full4692 stable sort; CPU single BLAS thread",
            }
        chosen = selection["chosen_classical"]
        for k in (10, 25, 50):
            values = []
            for iteration in range(12):
                for q in queries:
                    start = time.perf_counter()
                    atoms = parse_attributes(q["zh"])
                    coarse, _ = attribute_scores(p[chosen], selection["classes"], atoms)
                    if coarse is not None:
                        selected = stable_order(coarse, all_ids)[:k]
                        fine, _ = attribute_scores(p["dino"][selected], selection["classes"], atoms)
                        stable_order(fine, np.array(all_ids)[selected])
                    if iteration >= 2:
                        values.append((time.perf_counter() - start) * 1000)
            timing[f"cascade_{k}"] = {
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "scope": "same cached prediction bank; K does not save offline DINO encoding; selected candidates only for fine scoring",
            }
    start = time.perf_counter()
    encoder = Encoder("siglip")
    load_ms = (time.perf_counter() - start) * 1000
    source = ROOT / "artifacts/siglip"
    vectors = np.load(source / "image_embeddings.npy")
    source_ids = json.loads((source / "item_ids.json").read_text())
    values = []
    with threadpool_limits(limits=1):
        for iteration in range(4):
            for q in queries:
                start = time.perf_counter()
                text = encoder.encode_texts([q["en"]])
                stable_order(vectors @ text[0], source_ids)[:10]
                if iteration:
                    values.append((time.perf_counter() - start) * 1000)
    timing["siglip_dot_en"] = {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "observations": len(values),
        "load_ms": load_ms,
        "scope": "GPU text encoder + cosine + full4692 stable sort; single CPU BLAS thread, image vectors cached; excludes process start/HTTP",
    }
    report = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "gallery_count": len(all_ids),
        "timings": timing,
        "parser_exact_intents": 42,
        "parser_mismatches": mismatches,
        "classical_offline_manifest": json.loads(
            (ROOT / "artifacts/classical_cv/manifest.json").read_text()
        ),
        "warning": "Cached prediction banks vs image embeddings each use their natural offline preparation. Low online latency is not low total compute. Timings are sequential single-user in-process, not HTTP or concurrency benchmarks.",
    }
    dump(out / "latency.json", report)
    return report


def compare(out):
    results = json.loads((out / "test_results.json").read_text())
    rng = np.random.default_rng(20261104)
    comparisons = []
    for a, b in (
        ("raw_color", "gc_color"),
        ("raw_all", "gc_all"),
        ("dino", "fusion_0.25"),
        ("dino", "cascade_50"),
        ("siglip_dot_en", "gc_all"),
    ):
        diff = np.array([r["nDCG@10_known"] for r in results[b]["per_query"]]) - np.array(
            [r["nDCG@10_known"] for r in results[a]["per_query"]]
        )
        samples = diff[rng.integers(0, len(diff), size=(5000, len(diff)))].mean(1)
        comparisons.append(
            {
                "a": a,
                "b": b,
                "delta": float(diff.mean()),
                "descriptive_query_bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "wins": int((diff > 1e-10).sum()),
                "ties": int((np.abs(diff) <= 1e-10).sum()),
                "losses": int((diff < -1e-10).sum()),
            }
        )
    dump(
        out / "paired_comparisons.json",
        {
            "comparisons": comparisons,
            "limits": "21 overlapping attribute intents, one96-image gallery, AI labels; query bootstrap descriptive, not population generalization or proof of significance. Multiple comparisons not corrected.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="fresh")
    parser.add_argument("--task", choices=["stress", "timing", "compare", "all"], default="all")
    args = parser.parse_args()
    out = ROOT / f"artifacts/cv_research_20261104/{args.labels}"
    if args.task in ("stress", "all"):
        evaluate_stress(out)
    if args.task in ("timing", "all"):
        benchmark(out)
    if args.task in ("compare", "all"):
        compare(out)
    print("CV audit saved", flush=True)
