"""Compare frozen image representations on catalogue-supervised tasks.

The protocol is fixed before test scoring: dev selects only the C value of a
linear probe; test is then evaluated once for each backbone.  Catalogue text is
never supplied to a probe, so this measures image representation quality rather
than metadata lookup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.evaluate.backbone_tasks import (  # noqa: E402
    domain_metric_means,
    domain_split,
    fit_probe,
    paired_bootstrap,
)
from curio.evaluate.catalogue_tasks import TASKS  # noqa: E402
from curio.represent.validation import validate_object_splits  # noqa: E402


BACKBONES = {
    "clip": {
        "index": "artifacts/clip",
        "label": "OpenAI CLIP ViT-B/32",
        "family": "contrastive_text_image",
        "production": False,
    },
    "siglip1": {
        "index": "artifacts/cv_backbones_20261104/siglip1_index",
        "label": "SigLIP B/16-224 (first generation)",
        "family": "contrastive_text_image",
        "production": False,
    },
    "siglip2": {
        "index": "artifacts/siglip",
        "label": "SigLIP2 B/16-224",
        "family": "contrastive_text_image",
        "production": False,
    },
    "siglip2_lora": {
        "index": "artifacts/siglip_adapted",
        "label": "SigLIP2 + metadata-caption LoRA (production image tower)",
        "family": "contrastive_text_image_adapter",
        "production": True,
    },
    "siglip2_visual_lora": {
        "index": "artifacts/siglip_visual_adapted",
        "label": "SigLIP2 + visual-caption LoRA (research control)",
        "family": "contrastive_text_image_adapter",
        "production": False,
    },
    "dino": {
        "index": "artifacts/dinov2",
        "label": "DINOv2-small CLS",
        "family": "self_supervised_vision_only",
        "production": False,
    },
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_labels(path: Path, reference_ids: list[str]) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) != len(reference_ids):
        raise ValueError(f"Label count {len(rows)} differs from index count {len(reference_ids)}")
    by_id = {str(row["item_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate item_id in catalogue labels")
    missing = set(reference_ids) - set(by_id)
    if missing:
        raise ValueError(f"Index IDs missing from labels; sample: {sorted(missing)[:5]}")
    aligned = [by_id[item_id] for item_id in reference_ids]
    validate_object_splits(aligned)
    for split in ("train", "dev", "test"):
        if not any(row["split"] == split for row in aligned):
            raise ValueError(f"Missing {split} split")
    return aligned


def load_index(path: Path) -> tuple[np.ndarray, list[str], dict, dict]:
    vectors = np.load(path / "image_embeddings.npy", allow_pickle=False)
    ids = json.loads((path / "item_ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if vectors.ndim != 2 or vectors.shape[0] != len(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid index shape or duplicate IDs: {path}")
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite image vectors: {path}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"Zero image vector: {path}")
    hashes = {
        name: digest(path / name)
        for name in ("image_embeddings.npy", "item_ids.json", "manifest.json")
    }
    return vectors / norms, [str(item) for item in ids], manifest, hashes


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_report(report: dict) -> str:
    lines = [
        "# Catalogue-supervised backbone comparison (v3)",
        "",
        "This is a fixed supplementary evaluation. The catalogue supplies the "
        "labels and split; no per-item catalogue text is an inference feature. "
        "The test split has appeared in earlier studies, so it is not a new blind holdout.",
        "",
        f"Labels: `{report['labels_path']}`  ",
        f"Labels SHA-256: `{report['labels_sha256']}`  ",
        f"Backbones: {len(report['backbones'])}  ",
        "Selection: train -> dev macro-F1 selects C; test scored once after selection.",
        "",
        "## Domain results",
        "",
        "Each cell is test macro-F1 / balanced accuracy / accuracy. Only supported "
        "classes enter the metrics; unknown and known-but-unsupported catalogue "
        "values are reported separately.",
        "",
        "| Domain | n | " + " | ".join(report["backbones"]) + " |",
        "|---|---:|" + "---:|" * len(report["backbones"]),
    ]
    for domain in TASKS:
        results = report["domains"][domain]
        if results["status"] != "evaluable":
            lines.append(f"| {domain} | 0 | " + " | ".join("n/a" for _ in report["backbones"]) + " |")
            continue
        sample = next(iter(results["backbones"].values()))
        n = sample["test"]["n"]
        cells = []
        for name in report["backbones"]:
            value = results["backbones"][name]["test"]
            cells.append(f"{value['macro_f1']:.3f} / {value['balanced_accuracy']:.3f} / {value['accuracy']:.3f}")
        lines.append(f"| {domain} | {n} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Mean across evaluable domains", "",
                  "| Backbone | Mean macro-F1 | Mean balanced accuracy | Mean accuracy |",
                  "|---|---:|---:|---:|"])
    for name in report["backbones"]:
        row = report["domain_means"][name]
        lines.append(f"| {name} | {row['macro_f1']:.3f} | {row['balanced_accuracy']:.3f} | {row['accuracy']:.3f} |")
    lines.extend(["", "## Coverage and exclusions", "",
                  "| Domain | Split | Total | Catalogue known | Unknown | Evaluated | Known but unsupported |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    for domain in TASKS:
        coverage = report["coverage"][domain]
        for split in ("train", "dev", "test"):
            row = coverage[split]
            lines.append(f"| {domain} | {split} | {row['total']} | {row['catalogue_known']} | "
                         f"{row['catalogue_unknown']} | {row['supported']} | {row['known_but_unsupported']} |")
    lines.extend(["", "## Interpretation limits", "",
                  report["interpretation"], "", "## Reproduction", "", "```powershell",
                  ".\\.venv\\Scripts\\python.exe scripts/run_backbone_catalogue_v3.py",
                  ".\\.venv\\Scripts\\python.exe -m pytest tests/test_backbone_catalogue_v3.py -q",
                  ".\\.venv\\Scripts\\ruff.exe check scripts/run_backbone_catalogue_v3.py "
                  "src/curio/evaluate/backbone_tasks.py tests/test_backbone_catalogue_v3.py",
                  "```", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=(
        "artifacts/cv_v4_20261108/catalogue_v3_primary_plus_candidates/"
        "labels_catalogue_v2.jsonl"
    ))
    parser.add_argument("--output", default="artifacts/backbone_catalogue_v3_20261108")
    parser.add_argument("--repetitions", type=int, default=1000)
    args = parser.parse_args()
    labels_path = ROOT / args.labels
    output = ROOT / args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing research result: {output}")
    output.mkdir(parents=True, exist_ok=True)

    loaded = {}
    reference_ids = None
    for name, config in BACKBONES.items():
        vectors, ids, manifest, hashes = load_index(ROOT / config["index"])
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Index ordering differs for {name}")
        loaded[name] = {
            "vectors": vectors, "ids": ids, "manifest": manifest, "hashes": hashes,
            **config,
        }
    rows = load_labels(labels_path, reference_ids)
    report = {
        "version": "catalogue-backbone-comparison-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels_path": labels_path.relative_to(ROOT).as_posix(),
        "labels_sha256": digest(labels_path),
        "catalogue_sha256": "4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68",
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "dev", "test")
        },
        "object_group_counts": {
            split: len({row["object_group"] for row in rows if row["split"] == split})
            for split in ("train", "dev", "test")
        },
        "backbones": {name: {
            "label": value["label"], "family": value["family"],
            "production": value["production"], "index": value["index"],
            "dimension": int(value["vectors"].shape[1]),
            "index_sha256": value["manifest"].get("index_sha256"),
            "encoder_metadata": value["manifest"].get("encoder_metadata"),
            "artifact_sha256": value["hashes"],
        } for name, value in loaded.items()},
        "protocol": {
            "label_source": "museum catalogue only; catalogue text is never a model input",
            "split": "frozen object-group-isolated train/dev/test",
            "test_policy": "dev selects C only; test is fixed supplementary evaluation",
            "class_rule": "mutually exclusive derived ontology label; multi-value or missing is unknown",
            "class_support_floor": 20,
            "model": "class-balanced multinomial logistic regression on L2-normalized image vectors",
            "candidate_C": [0.1, 1.0, 10.0],
            "bootstrap": {"unit": "test image within each domain", "repetitions": args.repetitions},
        },
        "coverage": {}, "domains": {}, "domain_means": {},
        "interpretation": (
            "This experiment measures whether a frozen image representation can predict "
            "catalogue-derived, mutually exclusive attributes. It is not open-text retrieval, "
            "not authentication, and not evidence that a LoRA improves every visual task. "
            "DINO is a self-supervised vision-only control, not a text-aligned model. "
            "Domains have different sample sizes and coverage; the mean across domains is "
            "descriptive, not a weighted full-gallery accuracy. Color has insufficient "
            "train support under the fixed >=20 rule and is reported as not evaluable."
        ),
    }

    predictions_by_domain = {}
    for domain in TASKS:
        split = domain_split(rows, domain)
        report["coverage"][domain] = split.coverage
        if not split.classes or not all(len(split.indices[name]) for name in ("train", "dev", "test")):
            report["domains"][domain] = {
                "status": "not_evaluable",
                "reason": "no class reaches the fixed train support floor",
                "classes": list(split.classes),
                "coverage": split.coverage,
                "backbones": {},
            }
            continue
        domain_results = {}
        predictions = {}
        for name, value in loaded.items():
            fitted = fit_probe(rows, split, value["vectors"])
            if fitted.get("status") != "evaluable":
                domain_results[name] = fitted
                continue
            test_indices = split.indices["test"]
            truth = np.asarray([split.classes.index(split.labels[index])
                                for index in test_indices], dtype=np.int64)
            predictions[name] = fitted.pop("test_prediction")
            fitted.pop("dev_prediction", None)
            domain_results[name] = fitted
        if predictions:
            bootstrap = paired_bootstrap(truth, predictions, split.classes,
                                         repetitions=args.repetitions)
            for name in predictions:
                domain_results[name]["test_bootstrap"] = bootstrap[name]
            predictions_by_domain[domain] = {
                "truth": truth.tolist(), "classes": list(split.classes),
                "indices": split.indices["test"].tolist(),
                "predictions": {name: values.tolist() for name, values in predictions.items()},
            }
        report["domains"][domain] = {
            "status": "evaluable", "classes": list(split.classes),
            "coverage": split.coverage, "backbones": domain_results,
        }
        print(domain, {name: round(result["test"]["macro_f1"], 4)
                       for name, result in domain_results.items()
                       if result.get("status") == "evaluable"}, flush=True)

    for name in loaded:
        report["domain_means"][name] = domain_metric_means({
            domain: result["backbones"].get(name, {"status": "not_evaluable"})
            for domain, result in report["domains"].items()
        })

    report["pooled_diagnostic"] = {
        name: {
            "domain_prediction_count": int(sum(
                report["domains"][domain]["backbones"][name]["test"]["n"]
                for domain in predictions_by_domain
                if report["domains"][domain]["backbones"].get(name, {}).get("status") == "evaluable"
            )),
            "note": "counts repeat an image across domains; not a unique-image accuracy",
        }
        for name in loaded
    }

    write_json(output / "report.json", report)
    write_json(output / "predictions.json", predictions_by_domain)
    (output / "RESULTS.md").write_text(markdown_report(report), encoding="utf-8")
    manifest = {
        "report.json": digest(output / "report.json"),
        "predictions.json": digest(output / "predictions.json"),
        "RESULTS.md": digest(output / "RESULTS.md"),
        "source": {
            "scripts/run_backbone_catalogue_v3.py": digest(Path(__file__)),
            "src/curio/evaluate/backbone_tasks.py": digest(
                ROOT / "src/curio/evaluate/backbone_tasks.py"),
        },
    }
    write_json(output / "artifact_manifest.json", manifest)
    print(json.dumps({"status": "complete", "domains": list(report["domains"]),
                      "domain_means": report["domain_means"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
