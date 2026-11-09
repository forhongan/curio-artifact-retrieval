"""Make research tables and a scientific figure from measured artifacts."""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, cohen_kappa_score

from run_cv_research import BASE, ROOT, dump, label_value, rows, sha


OUT = ROOT / "artifacts/cv_research_20261104/fresh"


def classification_report():
    test = json.loads((OUT / "test_annotations.json").read_text())
    selection = json.loads((OUT / "selection.json").read_text())
    ids = json.loads((ROOT / "artifacts/classical_cv/item_ids.json").read_text())
    index = {key: i for i, key in enumerate(ids)}
    indices = [index[r["item_id"]] for r in test]
    result = {}
    for name in selection["models"]:
        prob = np.load(OUT / f"{name}_predictions.npy")[indices]
        records = []
        for j, key in enumerate(selection["classes"]):
            labels = np.array(
                [label_value(r, key) if label_value(r, key) is not None else -1 for r in test]
            )
            use = labels >= 0
            both = len(set(labels[use])) == 2
            records.append(
                {
                    "class": key,
                    "known": int(use.sum()),
                    "positive": int((labels == 1).sum()),
                    "AP": float(average_precision_score(labels[use], prob[use, j]))
                    if both
                    else None,
                    "F1_at_05": float(f1_score(labels[use], prob[use, j] >= 0.5, zero_division=0))
                    if both
                    else None,
                }
            )
        result[name] = {
            "macro_AP": float(np.mean([r["AP"] for r in records if r["AP"] is not None])),
            "by_task": {
                t: float(
                    np.mean(
                        [
                            r["AP"]
                            for r in records
                            if r["class"].startswith(t + "/") and r["AP"] is not None
                        ]
                    )
                )
                for t in ("object_type", "colors", "shape")
            },
            "classes": records,
        }
    dump(OUT / "classification_test.json", result)
    return result


def agreement_report():
    path = BASE / "annotator_qa/labels.jsonl"
    if not path.exists():
        return None
    original = {
        r["alias"]: r
        for person in ("a", "b")
        for r in rows(BASE / f"annotator_{person}/labels.jsonl")
    }
    repeated = rows(path)
    result = {
        "images": len(repeated),
        "annotators": "independent CLI sessions of requested gpt-6.1-terra; same model family, not human inter-rater reliability",
        "fields": {},
    }
    for task, keys in (
        ("object_type", ["object_type"]),
        ("colors", list(repeated[0]["colors"])),
        ("shape", list(repeated[0]["shape"])),
    ):
        left = []
        right = []
        unknown = 0
        total = 0
        disagreements = []
        for r in repeated:
            a = original[r["alias"]]
            for key in keys:
                av = a[task] if task == "object_type" else a[task][key]
                bv = r[task] if task == "object_type" else r[task][key]
                total += 1
                if av is None or bv is None:
                    unknown += 1
                    continue
                left.append(av)
                right.append(bv)
                if av != bv:
                    disagreements.append(
                        {"alias": r["alias"], "field": key, "original": av, "repeat": bv}
                    )
        kappa = float(cohen_kappa_score(left, right)) if len(set(left + right)) > 1 else None
        result["fields"][task] = {
            "jointly_known": len(left),
            "total": total,
            "either_unknown": unknown,
            "agreement": float(np.mean(np.array(left) == np.array(right))) if left else None,
            "kappa_descriptive": kappa,
            "disagreements": disagreements,
        }
    dump(OUT / "annotation_repeatability.json", result)
    return result


def audit_and_summary():
    selection = json.loads((OUT / "selection.json").read_text())
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    mapping = json.loads((BASE / "blind_map.json").read_text())
    catalog = {r["item_id"]: r for r in rows(ROOT / "data/snapshots/collection_v2.jsonl")}
    old = rows(ROOT / "eval/visual_labels/annotations.jsonl")
    families = {
        split: {
            catalog[r["item_id"]]["object_group"]
            for r in (old if split == "train" else mapping)
            if r["split"] == split
        }
        for split in ("train", "dev", "test")
    }
    assert not (
        families["train"] & families["dev"]
        or families["train"] & families["test"]
        or families["dev"] & families["test"]
    )
    for r in mapping:
        assert sha(ROOT / r["image_path"]) == r["image_sha256"]
    dump(
        OUT / "isolation_audit.json",
        {
            "status": "passed",
            "counts": {k: len(v) for k, v in families.items()},
            "mapping_sha256": sha(BASE / "blind_map.json"),
            "protocol_sha256": sha(BASE / "protocol.json"),
            "prior_pool_families_excluded": True,
        },
    )
    results = json.loads((OUT / "test_results.json").read_text())
    results_summary = {
        name: {
            **r["mean_all_intents"],
            "supported": r["supported_intents"],
            "intents": r["query_count"],
        }
        for name, r in results.items()
    }
    qrels = json.loads((OUT / "test_qrels.json").read_text())
    random_p = float(
        np.mean([sum(v == 1 for v in grades.values()) / len(grades) for grades in qrels.values()])
    )
    prior = {
        "expected_random_P@5": random_p,
        "scope": "Uniform random ranking over all96 images, known-positive lower relevance; expectations exact, not a fitted baseline",
    }
    dump(
        OUT / "summary.json",
        {
            "methods": results_summary,
            "random": prior,
            "gallery_count": manifest["gallery_count"],
            "selection": {
                "classical": selection["chosen_classical"],
                "fusion": selection["fusion"],
                "cascade": selection["cascade"],
            },
            "test_label_unknown_by_task": {
                t: sum(
                    v is None
                    for r in json.loads((OUT / "test_annotations.json").read_text())
                    for v in r[t].values()
                )
                for t in ("colors", "shape")
            },
        },
    )
    sections = []
    queries = json.loads((BASE / "queries.json").read_text(encoding="utf8"))
    groups = {
        "category": {q["qid"] for q in queries[:6]},
        "single_attribute": {q["qid"] for q in queries[6:15]},
        "conjunction": {q["qid"] for q in queries[15:]},
    }
    for group, keys in groups.items():
        sections.append(
            {
                "group": group,
                "query_count": len(keys),
                "methods": {
                    name: {
                        metric: float(
                            np.mean([q[metric] for q in r["per_query"] if q["qid"] in keys])
                        )
                        for metric in ("P@5", "nDCG@10_known")
                    }
                    for name, r in results.items()
                },
            }
        )
    dump(OUT / "task_breakdown.json", sections)
    return results_summary, prior


def figure(summary, prior):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [
        "raw_hog",
        "raw_color",
        "raw_texture",
        "raw_all",
        "gc_all",
        "dino",
        "fusion_0.25",
        "siglip",
        "siglip_dot_en",
        "production_dot_en",
    ]
    labels = [
        "HOG + LR",
        "HSV + LR",
        "LBP + LR",
        "HOG/HSV/LBP + LR",
        "GrabCut + classical LR",
        "DINO image + LR",
        "DINO + classical RRF",
        "SigLIP image + LR",
        "SigLIP text/image EN",
        "Adapted text/image EN",
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    colors = ["#507a73"] * 5 + ["#6686ae"] * 2 + ["#b07d55"] * 3
    vals = [summary[n]["P@5"] for n in names]
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
    ax.axvline(
        prior["expected_random_P@5"],
        color="#666666",
        ls="--",
        lw=1,
        label="uniform random expectation",
    )
    for y, val in enumerate(vals[::-1]):
        ax.text(val + 0.01, y, f"{val:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("P@5: explicitly visible positives only; unknown is not a hit")
    ax.set_title("Pixel-only gallery: 21 fixed intents, 96 fresh AI-labeled test images")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(
        0.01,
        0.01,
        "Supervised attribute heads: train218/dev48; test96. Finite vocabulary, one Terra judgment per image; not human gold or full-gallery accuracy.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    directory = ROOT / "docs/qa"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / "cv_research_comparison.png", dpi=170)
    fig.savefig(OUT / "comparison.svg")
    plt.close(fig)


if __name__ == "__main__":
    classification_report()
    agreement_report()
    summary, prior = audit_and_summary()
    figure(summary, prior)
    print("Summary and research figure saved")
