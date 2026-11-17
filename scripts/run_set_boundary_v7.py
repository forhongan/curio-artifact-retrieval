"""Run frozen-score diagnosis and condition-boundary ablations, without changing serving.

Requires v6 research/runtime plus v2 production image index and SigLIP2 text model.
Use --diagnostics-only to skip text inference and the new classifiers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.evaluate.boundary_diagnostics import crossfit_boundary, threshold_curve  # noqa: E402
from curio.evaluate.relevance import retrieval_metrics  # noqa: E402
from curio.retrieve.condition_boundary import (  # noqa: E402
    FEATURE_NAMES,
    condition_features,
    plan_texts,
    prompt_plan,
    support_scores,
)


def readl(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(root):
    base = root / "eval/relevance_v5"
    freeze = json.loads((base / "freeze.json").read_text())
    for name, digest in freeze["hashes"].items():
        if sha(base / name) != digest:
            raise ValueError(f"Frozen label protocol changed: {name}")
    queries, items, qrels = (
        readl(base / name) for name in ["queries.jsonl", "items.jsonl", "qrels.jsonl"]
    )
    # v5's score matrix was written in the original SigLIP index order.
    ids = json.loads((root / "artifacts/siglip/item_ids.json").read_text())
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gallery IDs")
    index = {item: i for i, item in enumerate(ids)}
    core = np.array([index[r["item_id"]] for r in items])
    dev = np.array([i for i, r in enumerate(items) if r["split"] == "dev"])
    test = np.array([i for i, r in enumerate(items) if r["split"] == "test"])
    if set(items[i]["object_group"] for i in dev) & set(items[i]["object_group"] for i in test):
        raise ValueError("Object-group leakage between dev and test")
    lookup = {(r["item_id"], r["qid"]): r["grade"] for r in qrels}
    grades = np.array([[lookup[(r["item_id"], q["qid"])] for q in queries] for r in items])
    scores = np.load(root / "artifacts/relevance_v5/scores.npz", allow_pickle=False)
    for lang in ["en", "zh"]:
        if scores[f"production_lora/dense/{lang}"].shape != (len(ids), len(queries)):
            raise ValueError("Score matrix does not align with frozen gallery/queries")
    return queries, items, ids, core, dev, test, grades, scores


def summarize(rows):
    tp, fp, fn = (sum(row[key] for row in rows) for key in ["tp", "fp", "fn"])
    return {
        "macro_f1": float(np.mean([r["set_f1"] for r in rows])),
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "empty_core": sum(r["selected_count"] == 0 for r in rows),
        "empty_gallery": sum(r["full_gallery_selected"] == 0 for r in rows),
        "unknown_selected": sum(r["unknown_selected"] for r in rows),
        "mean_gallery_selected": float(np.mean([r["full_gallery_selected"] for r in rows])),
        "tier_macro_f1": {
            tier: float(np.mean([r["set_f1"] for r in rows if r["tier"] == tier]))
            for tier in sorted({r["tier"] for r in rows})
        },
    }


def evaluate(scores, yes, grades, queries, core_test):
    rows = []
    for j, query in enumerate(queries):
        rows.append(
            {
                "qid": query["qid"],
                "tier": query["tier"],
                **retrieval_metrics(scores[core_test, j], grades[:, j], yes[core_test, j]),
                "full_gallery_selected": int(yes[:, j].sum()),
            }
        )
    return {"metrics": summarize(rows), "per_query": rows}


def plot_diagnostics(output, diagnostics, raw, grades, queries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for lang in ["en", "zh"]:
        fig, axes = plt.subplots(4, 5, figsize=(18, 12), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            d = diagnostics[lang][j]
            points = d["points"][1:]
            if d["positive"]:
                ax.step(
                    [p["recall"] for p in points], [p["precision"] for p in points], where="pre"
                )
            ax.set(
                xlim=(0, 1.02),
                ylim=(0, 1.02),
                title=queries[j]["qid"].removeprefix("v5_"),
                xlabel="Recall (known core)",
                ylabel="Precision (known core)",
            )
            ax.text(
                0.03,
                0.04,
                f"oracle F1={d['oracle']['f1']:.2f}; positives={d['positive']}",
                transform=ax.transAxes,
                fontsize=8,
            )
        fig.suptitle(
            f"{lang.upper()} raw SigLIP2 scores — retrospective TEST diagnosis, not deployable thresholds"
        )
        fig.savefig(output / f"pr_curves_{lang}.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(4, 5, figsize=(18, 12), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            for mask, label in [
                (grades[:, j] == 3, "relevant"),
                ((grades[:, j] >= 0) & (grades[:, j] != 3), "nonrelevant"),
            ]:
                ax.hist(raw[lang][mask, j], bins=15, alpha=0.5, label=label)
            ax.set(title=queries[j]["qid"].removeprefix("v5_"), xlabel="Cosine", ylabel="Count")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(f"{lang.upper()} test-core score overlap — unknown labels excluded")
        fig.savefig(output / f"score_overlap_{lang}.png", dpi=140)
        plt.close(fig)


def text_evidence(root, output, queries, ids, frozen):
    plans = {lang: [prompt_plan(q["conditions"], lang) for q in queries] for lang in ["en", "zh"]}
    texts = sorted({t for per in plans.values() for plan in per for t in plan_texts(plan)})
    manifest = {
        "texts": texts,
        "revision": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        "image_embeddings_sha256": sha(root / "artifacts/siglip_adapted/image_embeddings.npy"),
        "image_ids_sha256": sha(root / "artifacts/siglip_adapted/item_ids.json"),
        "plans": plans,
    }
    fingerprint = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    cache = output / f"condition_text_{fingerprint[:16]}.npz"
    old_text = np.load(root / "artifacts/relevance_v5/text_vectors.npz")["siglip"]
    if cache.exists():
        cached = np.load(cache)
        text, agreement = cached["vectors"], cached["agreement"]
    else:
        os.environ.setdefault("HF_HUB_CACHE", str(root / "models/huggingface/hub"))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from curio.represent.encoders import Encoder

        encoder = Encoder("siglip")
        # All adapters are vision-only: check the recovered text tower before use.
        check = encoder.encode_texts([q["text_en"] for q in queries[:4]])
        agreement = (check * old_text[:4]).sum(1)
        if agreement.min() < 0.999:
            raise ValueError("Text tower differs from the frozen v5 cache")
        text = encoder.encode_texts(texts)
        np.savez_compressed(cache, vectors=text, agreement=agreement)
    source_ids = json.loads((root / "artifacts/siglip_adapted/item_ids.json").read_text())
    lookup = {item: i for i, item in enumerate(source_ids)}
    vectors = np.load(root / "artifacts/siglip_adapted/image_embeddings.npy")[
        [lookup[item] for item in ids]
    ]
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    # Verifies row order, model, text order, and source embeddings against saved scores.
    errors = {}
    for k, lang in enumerate(["en", "zh"]):
        error = float(
            np.abs(
                vectors @ old_text[k * len(queries) : (k + 1) * len(queries)].T
                - frozen[f"production_lora/dense/{lang}"]
            ).max()
        )
        if error > 1e-5:
            raise ValueError(f"Frozen score/index alignment failed: {lang} max error={error}")
        errors[lang] = error
    prompt_matrix = vectors @ text.T
    prompt_scores = {t: prompt_matrix[:, i] for i, t in enumerate(texts)}
    generic = vectors @ old_text[-1]
    features = {
        lang: np.stack(
            [
                condition_features(
                    frozen[f"production_lora/dense/{lang}"][:, j], generic, plan, prompt_scores
                )
                for j, plan in enumerate(per)
            ],
            axis=1,
        )
        for lang, per in plans.items()
    }
    dump(
        output / "feature_manifest.json",
        {
            **manifest,
            "fingerprint": fingerprint,
            "text_tower_cosines": agreement.tolist(),
            "frozen_score_max_error": errors,
            "feature_names": FEATURE_NAMES,
        },
    )
    return features, plans


def run(args):
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    queries, items, ids, core, dev, test, grades, frozen = load_data(root)
    thresholds = json.loads((root / "artifacts/relevance_v5/thresholds.json").read_text())
    groups = [items[i]["object_group"] for i in dev]
    records, diagnostics, models, selected_sets = {}, {}, {}, {}
    predictions = {}
    for lang in ["en", "zh"]:
        for name in ["dense", "calibrated", "aux_raw_color"]:
            key = f"production_lora/{name}/{lang}"
            score = frozen[key]
            yes = score >= thresholds[key]
            records[f"frozen_{name}/{lang}"] = evaluate(
                score, yes, grades[test], queries, core[test]
            )
        raw = frozen[f"production_lora/dense/{lang}"]
        diagnostics[lang] = [
            {"qid": q["qid"], **threshold_curve(raw[core[test], j], grades[test, j])}
            for j, q in enumerate(queries)
        ]
    dump(
        output / "diagnostics.json",
        {"warning": "Test oracle is retrospective, not a fitted method", "languages": diagnostics},
    )
    plot_diagnostics(
        output,
        diagnostics,
        {lang: frozen[f"production_lora/dense/{lang}"][core[test]] for lang in ["en", "zh"]},
        grades[test],
        queries,
    )
    if not args.diagnostics_only:
        features, plans = text_evidence(root, output, queries, ids, frozen)
        # Fixed ablations, specified before evaluation; do not tune on test results.
        for lang in ["en", "zh"]:
            for name, width, weight in [
                ("refit_base", 3, 1.0),
                ("conditions", 9, 1.0),
                ("conditions_hard", 9, 2.0),
            ]:
                x = features[lang][:, :, :width]
                model = crossfit_boundary(x[core[dev]], grades[dev], groups, partial_weight=weight)
                score = support_scores(x.reshape(-1, width), model).reshape(x.shape[:2])
                yes = score >= model["threshold"]
                key = f"{name}/{lang}"
                models[key] = model
                predictions[key] = score.astype("float32")
                records[key] = evaluate(score, yes, grades[test], queries, core[test])
                selected_sets[key] = {
                    q["qid"]: [ids[i] for i in np.flatnonzero(yes[:, j])]
                    for j, q in enumerate(queries)
                }
                print(key, records[key]["metrics"], flush=True)
                loio_score, loio_yes = np.zeros_like(score), np.zeros_like(yes)
                loio_models = {}
                for j, q in enumerate(queries):
                    keep = np.arange(len(queries)) != j
                    fitted = crossfit_boundary(
                        x[core[dev]][:, keep], grades[dev][:, keep], groups, partial_weight=weight
                    )
                    loio_score[:, j] = support_scores(x[:, j], fitted)
                    loio_yes[:, j] = loio_score[:, j] >= fitted["threshold"]
                    fitted["training_qids"] = [r["qid"] for k, r in enumerate(queries) if k != j]
                    loio_models[q["qid"]] = fitted
                lkey = f"loio_{name}/{lang}"
                models[lkey] = loio_models
                predictions[lkey] = loio_score.astype("float32")
                records[lkey] = evaluate(loio_score, loio_yes, grades[test], queries, core[test])
                selected_sets[lkey] = {
                    q["qid"]: [ids[i] for i in np.flatnonzero(loio_yes[:, j])]
                    for j, q in enumerate(queries)
                }
                print(lkey, records[lkey]["metrics"], flush=True)
        # Review queue uses UNJUDGED gallery images only; blank labels are not negatives.
        core_ids = {items[i]["item_id"] for i in range(len(items))}
        with (output / "annotation_queue.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "qid",
                    "language",
                    "query",
                    "item_id",
                    "support",
                    "threshold",
                    "missing_conditions",
                    "human_grade",
                    "notes",
                ],
            )
            writer.writeheader()
            for lang in ["en", "zh"]:
                t = models[f"conditions/{lang}"]["threshold"]
                for j, query in enumerate(queries):
                    s = predictions[f"conditions/{lang}"][:, j]
                    order = sorted(
                        (i for i, item in enumerate(ids) if item not in core_ids),
                        key=lambda i: (abs(float(s[i]) - t), ids[i]),
                    )[:10]
                    for i in order:
                        writer.writerow(
                            {
                                "qid": query["qid"],
                                "language": lang,
                                "query": query[f"text_{lang}"],
                                "item_id": ids[i],
                                "support": float(s[i]),
                                "threshold": t,
                                "missing_conditions": ";".join(plans[lang][j]["missing"]),
                            }
                        )
        np.savez_compressed(output / "scores.npz", **predictions)
        dump(
            output / "models.json",
            {"feature_manifest_sha256": sha(output / "feature_manifest.json"), "models": models},
        )
        dump(output / "selected_ids.json", selected_sets)
    comparisons = {}
    if not args.diagnostics_only:
        rng = np.random.default_rng(7)
        samples = rng.integers(0, len(queries), size=(2000, len(queries)))
        for prefix in ["", "loio_"]:
            for name in ["conditions", "conditions_hard"]:
                difference = np.mean(
                    [
                        np.array(
                            [r["set_f1"] for r in records[f"{prefix}{name}/{lang}"]["per_query"]]
                        )
                        - np.array(
                            [
                                r["set_f1"]
                                for r in records[f"{prefix}refit_base/{lang}"]["per_query"]
                            ]
                        )
                        for lang in ["en", "zh"]
                    ],
                    axis=0,
                )
                comparisons[prefix + name] = {
                    "delta_macro_f1": float(difference.mean()),
                    "paired_intent_bootstrap_95ci": np.quantile(
                        difference[samples].mean(1), [0.025, 0.975]
                    ).tolist(),
                }
    result = {
        "scope": "140 judged test images, not full-gallery recall",
        "python": platform.python_version(),
        "gallery": len(ids),
        "dev": len(dev),
        "test": len(test),
        "intents": len(queries),
        "input_hashes": {
            str(p.relative_to(root)): sha(p)
            for p in [
                root / "artifacts/relevance_v5/scores.npz",
                root / "eval/relevance_v5/freeze.json",
                root / "docs/research/set_boundary_v7/PROTOCOL.md",
            ]
        },
        "records": records,
        "comparisons": comparisons,
        "oracle_macro_f1": {
            lang: float(np.mean([d["oracle"]["f1"] for d in diagnostics[lang]]))
            for lang in ["en", "zh"]
        },
    }
    dump(output / "results.json", result)
    lines = [
        "# Set boundary v7 results",
        "",
        "140 test images; 20 paired intents; unknowns excluded with bounds in JSON.",
        "Test oracle is a retrospective score-separability diagnostic, not a deployable method.",
        "",
        "| Method / language | Macro F1 | Micro P | Micro R | Empty core | Mean gallery size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, record in records.items():
        m = record["metrics"]
        p = f"{m['micro_precision']:.4f}" if m["micro_precision"] is not None else "undefined"
        r = f"{m['micro_recall']:.4f}" if m["micro_recall"] is not None else "undefined"
        lines.append(
            f"| {key} | {m['macro_f1']:.4f} | {p} | {r} | {m['empty_core']} | {m['mean_gallery_selected']:.1f} |"
        )
    lines += [
        "",
        "Raw score test-oracle macro F1: " + json.dumps(result["oracle_macro_f1"]),
        "",
        "Paired intent bootstrap vs same-protocol refit baseline (exploratory):",
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
        "",
        "No production promotion. Structured queries, historically reused AI-assisted labels, only 60 dev images.",
        "OOF threshold selection is not risk certification; LOIO only holds out boundary training intents.",
        "Annotation queue is uncertainty-sampled and must not be used as an unbiased evaluation set.",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", output / "RESULTS.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/set_boundary_v7")
    parser.add_argument("--diagnostics-only", action="store_true")
    with threadpool_limits(limits=1):
        run(parser.parse_args())
