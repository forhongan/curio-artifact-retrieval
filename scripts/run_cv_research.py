"""Train and compare classical and image-only attribute retrieval workflows."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from threadpoolctl import threadpool_limits

from curio.retrieve.attribute_cv import (
    attribute_scores,
    predict_heads,
    stable_order,
    rrf_scores,
    cascade_order,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/cv_research_20261104"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]


def dump(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8"
    )


def merge_fresh():
    mapping = {r["alias"]: r for r in json.loads((BASE / "blind_map.json").read_text())}
    labels = []
    for person in ("a", "b"):
        expected = set(json.loads((BASE / f"annotator_{person}/aliases.json").read_text()))
        actual = rows(BASE / f"annotator_{person}/labels.jsonl")
        if len(actual) != len(expected) or {r["alias"] for r in actual} != expected:
            raise ValueError("Incomplete independent labels")
        for r in actual:
            mapped = mapping[r["alias"]]
            for task, keys in (
                ("colors", ["blue", "white", "green", "red", "black", "yellow", "brown", "pink"]),
                ("shape", ["narrow_mouth", "wide_mouth", "handles", "footed", "round_body"]),
            ):
                if set(r[task]) != set(keys) or any(
                    v not in (0, 1, None) for v in r[task].values()
                ):
                    raise ValueError("Malformed pixel labels")
            labels.append(
                {
                    **r,
                    **{k: mapped[k] for k in ("item_id", "split", "image_sha256")},
                    "label_source": "ai_image_only",
                }
            )
    return labels


def label_value(row, key):
    task, label = key.split("/")
    if task == "object_type":
        return int(row[task] == label) if row[task] != "other" else None
    return row.get(task, {}).get(label)


def relevance(labels, atoms):
    values = []
    for row in labels:
        observed = [label_value(row, f"{t}/{v}") for t, v, _ in atoms]
        if any(x is not None and x != a[2] for x, a in zip(observed, atoms)):
            values.append(0)
        elif any(x is None for x in observed):
            values.append(-1)
        else:
            values.append(1)
    return np.array(values)


def metrics(order, rel):
    # Unknowns do not silently become negatives: pessimistic and optimistic bounds.
    npos = int((rel == 1).sum())
    hits = (rel[order] == 1).astype(float)
    upper = (rel[order] != 0).astype(float)
    discounts = 1 / np.log2(np.arange(2, min(10, len(order)) + 2))
    ideal = float(discounts[: min(10, npos)].sum())
    ndcg = float(np.dot(hits[:10], discounts) / ideal) if ideal else 0.0
    return {
        "P@5": float(hits[:5].sum() / 5),
        "P@5_upper_unknown": float(upper[:5].sum() / 5),
        "nDCG@10_known": ndcg,
        "AP@10": float(np.sum(np.cumsum(hits[:10]) / np.arange(1, 11) * hits[:10]) / npos)
        if npos
        else 0.0,
        "Recall@10": float(hits[:10].sum() / npos) if npos else 0.0,
        "positive_count": npos,
        "unknown_count": int((rel == -1).sum()),
        "zero_positive": npos == 0,
    }


def fit_model(x, train, dev, classes, index, out):
    train_idx = np.array([index[r["item_id"]] for r in train])
    dev_idx = np.array([index[r["item_id"]] for r in dev])
    options = []
    winner = None
    start = time.perf_counter()
    for c in (0.1, 1.0, 10.0):
        coefficients = []
        intercepts = []
        aps = []
        per_class = []
        for key in classes:
            ytrain = np.array(
                [label_value(r, key) if label_value(r, key) is not None else -1 for r in train]
            )
            ydev = np.array(
                [label_value(r, key) if label_value(r, key) is not None else -1 for r in dev]
            )
            known = ytrain >= 0
            lr = LogisticRegression(
                C=c, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=240
            )
            lr.fit(x[train_idx[known]], ytrain[known])
            co, inter = lr.coef_[0], float(lr.intercept_[0])
            coefficients.append(co)
            intercepts.append(inter)
            use = ydev >= 0
            if len(set(ydev[use])) == 2:
                probs = predict_heads(x[dev_idx[use]], np.array([co]), np.array([inter]))[:, 0]
                ap = float(average_precision_score(ydev[use], probs))
                aps.append(ap)
                per_class.append(
                    {
                        "class": key,
                        "AP": ap,
                        "F1": float(f1_score(ydev[use], probs >= 0.5, zero_division=0)),
                    }
                )
        selection = float(np.mean(aps)) if aps else 0.0
        record = {
            "C": c,
            "dev_macro_AP": selection,
            "dev_eligible_classes": len(aps),
            "classes": per_class,
        }
        options.append(record)
        if winner is None or selection > winner[0] + 1e-12:
            winner = (
                selection,
                c,
                np.array(coefficients, dtype=np.float32),
                np.array(intercepts, dtype=np.float32),
            )
    np.savez_compressed(
        out, coefficients=winner[2], intercepts=winner[3], classes=np.array(classes)
    )
    return {
        "selected_C": winner[1],
        "dev_macro_AP": winner[0],
        "candidates": options,
        "fit_seconds": time.perf_counter() - start,
        "artifact_sha256": sha(out),
    }


def load_features():
    cdir = ROOT / "artifacts/classical_cv"
    ids = json.loads((cdir / "item_ids.json").read_text())
    x = {
        k: v.astype(np.float32)
        for k, v in dict(np.load(cdir / "features.npz")).items()
        if k != "gc_valid"
    }
    provenance = {"classical": json.loads((cdir / "manifest.json").read_text())}
    for name, directory in (("dino", "dinov2"), ("siglip", "siglip")):
        directory = ROOT / "artifacts" / directory
        source_ids = json.loads((directory / "item_ids.json").read_text())
        lookup = {s: i for i, s in enumerate(source_ids)}
        vectors = np.load(directory / "image_embeddings.npy")
        x[name] = np.asarray(vectors[[lookup[i] for i in ids]], dtype=np.float32)
        provenance[name] = {
            "manifest": json.loads((directory / "manifest.json").read_text()),
            "image_vectors_sha256": sha(directory / "image_embeddings.npy"),
        }
    return ids, x, provenance


def make_scores(prob, classes, queries):
    scores = {}
    coverage = {}
    for q in queries:
        scores[q["qid"]], missing = attribute_scores(prob, classes, q["atoms"])
        coverage[q["qid"]] = {"supported": not missing, "unsupported": missing}
    return scores, coverage


def evaluate_method(scores, ids, labels, queries):
    by_id = {r["item_id"]: r for r in labels}
    assert list(by_id) == ids
    output = []
    for q in queries:
        rel = relevance(labels, q["atoms"])
        score = scores[q["qid"]]
        order = stable_order(score, ids) if score is not None else np.array([], dtype=int)
        if len(order):
            result = metrics(order, rel)
        else:
            result = {
                **metrics(np.arange(len(ids)), rel),
                **{
                    k: 0.0
                    for k in ("P@5", "P@5_upper_unknown", "nDCG@10_known", "AP@10", "Recall@10")
                },
            }
        output.append(
            {
                "qid": q["qid"],
                "supported": score is not None,
                **result,
                "top10": [ids[i] for i in order[:10]],
            }
        )
    keys = ("P@5", "P@5_upper_unknown", "nDCG@10_known", "AP@10", "Recall@10")
    return {
        "mean_all_intents": {k: float(np.mean([r[k] for r in output])) for k in keys},
        "nonzero_positive_intents": sum(not r["zero_positive"] for r in output),
        "supported_intents": sum(r["supported"] for r in output),
        "query_count": len(output),
        "per_query": output,
    }


def main(label_set):
    out = ROOT / f"artifacts/cv_research_20261104/{label_set}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        raise FileExistsError(
            "Selection exists; use existing results or a separately versioned experiment"
        )
    old = rows(ROOT / "eval/visual_labels/annotations.jsonl")
    train = [r for r in old if r["split"] == "train"]
    label_rows = (
        merge_fresh() if label_set == "fresh" else [r for r in old if r["split"] != "train"]
    )
    dev = [r for r in label_rows if r["split"] == "dev"]
    # Do not read test labels or score them during fitting/selection.
    ids, x, provenance = load_features()
    index = {i: n for n, i in enumerate(ids)}
    queries = json.loads((BASE / "queries.json").read_text(encoding="utf8"))
    classes = []
    support = {}
    keys = [
        f"object_type/{v}"
        for v in sorted({r["object_type"] for r in train if r["object_type"] != "other"})
    ]
    keys += [f"{task}/{key}" for task in ("colors", "shape") for key in sorted(train[0][task])]
    for key in keys:
        count = Counter(label_value(r, key) for r in train)
        support[key] = {str(k): v for k, v in count.items()}
        if count[0] >= 5 and count[1] >= 5:
            classes.append(key)
    dump(out / "support.json", support)
    dump(out / "provenance.json", provenance)
    dump(
        out / "merged_annotations.json",
        {
            "train": train,
            "dev": dev,
            "heldout_count": sum(r["split"] == "test" for r in label_rows),
        },
    )
    model_reports = {}
    dev_scores = {}
    coverage = {}
    probabilities = {}
    with threadpool_limits(limits=1):
        for name, features in x.items():
            model_reports[name] = fit_model(
                features, train, dev, classes, index, out / f"{name}_heads.npz"
            )
            model = np.load(out / f"{name}_heads.npz")
            probabilities[name] = predict_heads(
                features, model["coefficients"], model["intercepts"]
            ).astype(np.float32)
            np.save(out / f"{name}_predictions.npy", probabilities[name])
            dev_prob = probabilities[name][[index[r["item_id"]] for r in dev]]
            dev_scores[name], coverage[name] = make_scores(dev_prob, classes, queries)
            print(
                name,
                model_reports[name]["selected_C"],
                round(model_reports[name]["dev_macro_AP"], 4),
                flush=True,
            )
    dev_ids = [r["item_id"] for r in dev]
    dev_results = {
        name: evaluate_method(scores, dev_ids, dev, queries) for name, scores in dev_scores.items()
    }
    # Fix classical branch and workflow choices entirely on development data.
    classical_names = [n for n in x if n.startswith(("raw_", "gc_"))]
    chosen = max(classical_names, key=lambda n: dev_results[n]["mean_all_intents"]["nDCG@10_known"])
    blend = []
    for weight in (0, 0.25, 0.5, 0.75, 1):
        scores = {
            q["qid"]: rrf_scores(
                dev_scores["dino"][q["qid"]], dev_scores[chosen][q["qid"]], dev_ids, weight
            )
            if dev_scores["dino"][q["qid"]] is not None
            else None
            for q in queries
        }
        result = evaluate_method(scores, dev_ids, dev, queries)
        blend.append({"weight": weight, "dev_nDCG@10": result["mean_all_intents"]["nDCG@10_known"]})
    fusion = max(blend, key=lambda r: r["dev_nDCG@10"])
    cascades = []
    for k in (10, 25, 50):
        scores = {}
        for q in queries:
            coarse, fine = dev_scores[chosen][q["qid"]], dev_scores["dino"][q["qid"]]
            if coarse is None:
                scores[q["qid"]] = None
                continue
            order = cascade_order(coarse, fine, dev_ids, k)
            rank = np.empty(len(dev_ids))
            rank[order] = -np.arange(len(dev_ids))
            scores[q["qid"]] = rank
        result = evaluate_method(scores, dev_ids, dev, queries)
        cascades.append({"K": k, "dev_nDCG@10": result["mean_all_intents"]["nDCG@10_known"]})
    cascade = max(cascades, key=lambda r: r["dev_nDCG@10"])
    selection = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label_set": label_set,
        "classes": classes,
        "models": model_reports,
        "chosen_classical": chosen,
        "fusion": fusion,
        "fusion_candidates": blend,
        "cascade": cascade,
        "cascade_candidates": cascades,
        "query_protocol_sha256": sha(BASE / "protocol.json"),
        "fitting_inputs": "train and dev labels only; fixed feature extraction; test never used to select C/branch/weight/K",
        "limitations": "AI labels, closed attribute vocabulary, balanced LR scores not calibrated probabilities; cup unsupported from insufficient train examples",
    }
    dump(out / "selection.json", selection)
    dump(out / "dev_results.json", dev_results)
    # Final heldout evaluation happens only after writing immutable selection.
    test = [r for r in label_rows if r["split"] == "test"]
    test_ids = [r["item_id"] for r in test]
    test_index = [index[i] for i in test_ids]
    test_scores = {
        name: make_scores(p[test_index], classes, queries)[0] for name, p in probabilities.items()
    }
    for split, labels in (("dev", dev), ("test", test)):
        dump(
            out / f"{split}_qrels.json",
            {
                q["qid"]: {
                    r["item_id"]: int(v) for r, v in zip(labels, relevance(labels, q["atoms"]))
                }
                for q in queries
            },
        )
    for weight in (0.5, fusion["weight"]):
        test_scores[f"fusion_{weight}"] = {
            q["qid"]: rrf_scores(
                test_scores["dino"][q["qid"]], test_scores[chosen][q["qid"]], test_ids, weight
            )
            if test_scores["dino"][q["qid"]] is not None
            else None
            for q in queries
        }
    for k in (10, 25, 50):
        scores = {}
        for q in queries:
            coarse, fine = test_scores[chosen][q["qid"]], test_scores["dino"][q["qid"]]
            if coarse is None:
                scores[q["qid"]] = None
                continue
            order = cascade_order(coarse, fine, test_ids, k)
            rank = np.empty(len(test_ids))
            rank[order] = -np.arange(len(test_ids))
            scores[q["qid"]] = rank
        test_scores[f"cascade_{k}"] = scores
    results = {
        name: evaluate_method(scores, test_ids, test, queries)
        for name, scores in test_scores.items()
    }
    dump(out / "test_results.json", results)
    dump(out / "test_annotations.json", test)
    # Contrastive models are baselines, without contributing to workflow selection.
    from curio.represent.encoders import Encoder

    for name, adapter in (
        ("siglip_dot", None),
        ("production_dot", ROOT / "artifacts/training/siglip_lora_random/best_adapter"),
    ):
        encoder = Encoder("siglip", adapter_path=adapter)
        source = ROOT / "artifacts" / ("siglip" if adapter is None else "siglip_adapted")
        source_ids = json.loads((source / "item_ids.json").read_text())
        source_index = {i: j for j, i in enumerate(source_ids)}
        vectors = np.load(source / "image_embeddings.npy")[[source_index[i] for i in test_ids]]
        for lang in ("zh", "en"):
            text = encoder.encode_texts([q[lang] for q in queries])
            matrix = text @ vectors.T
            score = {q["qid"]: matrix[j] for j, q in enumerate(queries)}
            results[f"{name}_{lang}"] = evaluate_method(score, test_ids, test, queries)
            np.save(out / f"{name}_{lang}_text.npy", text)
        del encoder
    dump(out / "test_results.json", results)
    dump(
        out / "run_manifest.json",
        {
            "label_set": label_set,
            "gallery_count": len(test),
            "train_count": len(train),
            "dev_count": len(dev),
            "test_count": len(test),
            "selection_sha256": sha(out / "selection.json"),
            "script_sha256": sha(__file__),
            "classes": classes,
            "labels_sha256": sha(out / "test_annotations.json"),
            "queries_sha256": sha(BASE / "queries.json"),
            "scope": "restricted exhaustively attribute-labeled gallery; not full4692 relevance or human gold",
        },
    )
    for name, result in results.items():
        print(name, json.dumps(result["mean_all_intents"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", choices=["fresh", "existing"], default="fresh")
    main(parser.parse_args().labels)
