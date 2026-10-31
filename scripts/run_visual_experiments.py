"""Select image-only retrieval and image-caption adaptation on blind visual dev data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def model_fingerprints(visual_adapter, metadata_adapter):
    paths = []
    for model in ("clip", "siglip", "siglip_adapted", "siglip_visual_adapted"):
        paths.extend(Path(f"artifacts/{model}").glob("*"))
    for adapter in (visual_adapter, metadata_adapter):
        paths.extend(Path(adapter).glob("adapter*"))
    for probes in ("siglip", "vision", "siglip_adapted", "vision_metadata_adapted", "siglip_visual_adapted", "vision_visual_adapted"):
        directory = Path(f"artifacts/probes/{probes}")
        paths.extend(directory.glob("*.npz"))
        paths.extend(directory / name for name in ("report.json", "ontology.json") if (directory / name).exists())
    return {path.as_posix(): sha(path) for path in sorted(set(paths)) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("dev", "test"), default="dev")
    parser.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    parser.add_argument("--annotations", default="eval/visual_labels/annotations.jsonl")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    destination = Path("artifacts/visual_selection.json")
    logs = Path("logs/visual_experiments")
    logs.mkdir(parents=True, exist_ok=True)
    status_path = Path("artifacts/visual_experiment_status.json")
    config = {
        "protocol_version": "visual-v2-grouped-all-image-candidates",
        "data_sha256": sha(args.data),
        "annotations_sha256": sha(args.annotations),
        "epochs": args.epochs,
        "dev_files_sha256": {
            name: sha(f"eval/visual_dev/{name}")
            for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")
        },
    }
    status = (
        read(status_path)
        if (args.resume or args.phase == "test") and status_path.exists()
        else {"config": config, "steps": {}}
    )
    if status["config"] != config:
        raise ValueError("Visual experiment resume config mismatch")

    def run(name, *command):
        if args.resume and status["steps"].get(name) == "complete":
            print(f"SKIP {name}", flush=True)
            return
        status["steps"][name] = "running"
        status_path.write_text(json.dumps(status, indent=2))
        print(f"START {name}", flush=True)
        started = time.perf_counter()
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, *command],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"},
            )
        status["steps"][name] = "complete" if result.returncode == 0 else "failed"
        status.setdefault("seconds", {})[name] = time.perf_counter() - started
        status_path.write_text(json.dumps(status, indent=2))
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {logs / (name + '.log')}")
        print(f"END {name}", flush=True)

    def paired(model, split, adapter=None):
        run(
            f"paired_{model}_{split}",
            "-m",
            "curio.evaluate.harness",
            "--paired",
            "--encoder",
            "clip" if model == "clip" else "siglip",
            "--data",
            args.data,
            "--split",
            split,
            "--visual-annotations",
            args.annotations,
            "--output",
            f"artifacts/evaluation/visual_paired_{model}_{split}.json",
            *(["--adapter", adapter] if adapter else []),
        )

    def retrieval(model, split, variant="baseline,translated", probes=None, weight=0.0, label=None):
        label = label or f"{model}_{variant.replace(',', '_')}_{split}"
        output = f"artifacts/evaluation/image_audit/{label}"
        run(
            f"audit_{label}",
            "-m",
            "curio.evaluate.harness",
            "--data",
            args.data,
            "--index",
            f"artifacts/{model}",
            "--encoder",
            "clip" if model == "clip" else "siglip",
            "--queries",
            f"eval/visual_{split}/queries.jsonl",
            "--qrels",
            f"eval/visual_{split}/qrels.jsonl",
            "--gallery-ids",
            f"eval/visual_{split}/gallery_ids.json",
            "--variants",
            variant,
            "--output",
            output,
            *(["--probes", probes, "--probe-weight", str(weight)] if probes else []),
        )
        return output

    if args.phase == "dev":
        dev_fingerprints = {
            name: sha(f"eval/visual_dev/{name}")
            for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")
        }
        run(
            "detail_probes",
            "-m",
            "curio.represent.detail_probes",
            "--data",
            args.data,
            "--index",
            "artifacts/siglip",
            "--annotations",
            args.annotations,
            "--base-probes",
            "artifacts/probes/siglip",
            "--output",
            "artifacts/probes/vision",
            "--defer-test",
        )
        candidates = []
        for sampling in ("random", "hard"):
            output = f"artifacts/training/visual_lora_{sampling}"
            run(
                f"train_visual_{sampling}",
                "-m",
                "curio.represent.train",
                "--data",
                args.data,
                "--visual-annotations",
                args.annotations,
                "--epochs",
                str(args.epochs),
                "--sampling",
                sampling,
                "--output",
                output,
            )
            candidates.append(read(f"{output}/run.json"))
        best = max(candidates, key=lambda row: (row["best_dev"]["r@5"], row["best_dev"]["mrr"]))
        visual_adapter = str(Path(best["config"]["output"]) / "best_adapter")
        run(
            "index_visual_adapted",
            "-m",
            "curio.index.build",
            "--data",
            args.data,
            "--encoder",
            "siglip",
            "--adapter",
            visual_adapter,
            "--output",
            "artifacts/siglip_visual_adapted",
        )
        run(
            "base_probes_visual_adapted", "-m", "curio.represent.probes",
            "--data", args.data, "--index", "artifacts/siglip_visual_adapted",
            "--output", "artifacts/probes/siglip_visual_adapted",
            "--defer-test",
        )
        run(
            "detail_probes_visual_adapted", "-m", "curio.represent.detail_probes",
            "--data", args.data, "--index", "artifacts/siglip_visual_adapted",
            "--annotations", args.annotations,
            "--base-probes", "artifacts/probes/siglip_visual_adapted",
            "--output", "artifacts/probes/vision_visual_adapted",
            "--defer-test",
        )
        run(
            "base_probes_metadata_adapted", "-m", "curio.represent.probes",
            "--data", args.data, "--index", "artifacts/siglip_adapted",
            "--output", "artifacts/probes/siglip_adapted", "--defer-test",
        )
        run(
            "detail_probes_metadata_adapted", "-m", "curio.represent.detail_probes",
            "--data", args.data, "--index", "artifacts/siglip_adapted",
            "--annotations", args.annotations,
            "--base-probes", "artifacts/probes/siglip_adapted",
            "--output", "artifacts/probes/vision_metadata_adapted", "--defer-test",
        )
        metadata_adapter = read("artifacts/selection.json")["adapter_path"]
        for model, adapter in (
            ("clip", None),
            ("siglip", None),
            ("siglip_adapted", metadata_adapter),
            ("siglip_visual_adapted", visual_adapter),
        ):
            paired(model, "dev", adapter)
        options = []
        for model in ("clip", "siglip", "siglip_adapted", "siglip_visual_adapted"):
            output = retrieval(model, "dev")
            report = read(f"{output}/metrics.json")
            for variant in ("baseline", "translated"):
                score = npmean(
                    [
                        row["metrics"]["ndcg@10"]["mean"]
                        for row in report["summary"]
                        if row["variant"] == variant
                    ]
                )
                options.append(
                    {
                        "model": model,
                        "variant": variant,
                        "probes": None,
                        "weight": 0.0,
                        "dev_ndcg@10": score,
                    }
                )
        for model, probe_name in (
            ("siglip", "siglip"), ("siglip", "vision"),
            ("siglip_adapted", "siglip_adapted"),
            ("siglip_adapted", "vision_metadata_adapted"),
            ("siglip_visual_adapted", "siglip_visual_adapted"),
            ("siglip_visual_adapted", "vision_visual_adapted"),
        ):
            probes = f"artifacts/probes/{probe_name}"
            for weight in (0.0, 0.05, 0.15, 0.3):
                label = f"{model}_{probe_name}_w{weight:g}_dev"
                output = retrieval(model, "dev", "image_only", probes, weight, label)
                report = read(f"{output}/metrics.json")
                if any(row["degraded_queries"] for row in report["summary"]):
                    raise RuntimeError("Image-only dev run degraded; do not select a broken model")
                options.append(
                    {
                        "model": model,
                        "variant": "image_only",
                        "probes": probes,
                        "weight": weight,
                        "dev_ndcg@10": npmean(
                            [row["metrics"]["ndcg@10"]["mean"] for row in report["summary"]]
                        ),
                    }
                )
        # Production uses the explicit image-only family. Other zero-shot and
        # adapted methods remain fully reported comparison systems.
        overall_best = max(options, key=lambda row: row["dev_ndcg@10"])
        choice = max(
            [row for row in options if row["variant"] == "image_only"],
            key=lambda row: row["dev_ndcg@10"],
        )
        # A positive weight is selected on dev for an informative head ablation,
        # even if the overall winning method has no attribute head.
        head_choice = max(
            [row for row in options if row["probes"] == "artifacts/probes/vision" and row["weight"] > 0],
            key=lambda row: row["dev_ndcg@10"],
        )
        selection = {
            **config,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "visual_dev_files_sha256": dev_fingerprints,
            "options": options,
            "selected": choice,
            "overall_best_comparison": overall_best,
            "head_ablation_weight": head_choice["weight"],
            "visual_adapter_path": visual_adapter,
            "metadata_adapter_path": metadata_adapter,
            "model_files_sha256": model_fingerprints(visual_adapter, metadata_adapter),
            "visual_training": [
                {
                    "sampling": row["config"]["sampling"],
                    "best_epoch": row["best_epoch"],
                    "best_dev": row["best_dev"],
                    "improved_over_baseline": row["improved_over_baseline"],
                }
                for row in candidates
            ],
            "selection_rule": "Production: highest bilingual visual-dev nDCG@10 within the explicit image_only family; stable ties prefer earlier candidate and lower weight. Overall best across all comparison models is reported separately. Restricted AI-labeled dev gallery, not human gold.",
        }
        destination.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"VISUAL DEV FROZEN: {choice}", flush=True)
    else:
        selection = read(destination)
        if any(selection[key] != value for key, value in config.items()):
            raise ValueError("Visual test config differs from frozen dev")
        if any(
            sha(f"eval/visual_dev/{name}") != digest
            for name, digest in selection["visual_dev_files_sha256"].items()
        ):
            raise ValueError("Visual dev labels changed after selection")
        actual_models = model_fingerprints(selection["visual_adapter_path"], selection["metadata_adapter_path"])
        if actual_models != selection["model_files_sha256"]:
            raise ValueError("Model, index or probe files changed after visual selection")
        test_fingerprints = {
            name: sha(f"eval/visual_test/{name}")
            for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json")
        }
        if status.get("test_files_sha256", test_fingerprints) != test_fingerprints:
            raise ValueError("Visual test files changed after evaluation started")
        freeze_path = Path("artifacts/visual_test_freeze.json")
        test_freeze = {"config": config, "test_files_sha256": test_fingerprints,
                       "selection_sha256": sha(destination)}
        if freeze_path.exists() and read(freeze_path) != test_freeze:
            raise ValueError("Frozen visual test inputs changed; use a new experiment version")
        if not freeze_path.exists():
            freeze_path.write_text(json.dumps(test_freeze, indent=2))
        status["test_files_sha256"] = test_fingerprints
        status_path.write_text(json.dumps(status, indent=2))
        for name, model, module in (
            ("siglip", "siglip", "probes"), ("vision", "siglip", "detail_probes"),
            ("siglip_adapted", "siglip_adapted", "probes"),
            ("vision_metadata_adapted", "siglip_adapted", "detail_probes"),
            ("siglip_visual_adapted", "siglip_visual_adapted", "probes"),
            ("vision_visual_adapted", "siglip_visual_adapted", "detail_probes"),
        ):
            run(
                f"probe_test_{name}", "-m", f"curio.represent.{module}",
                "--data", args.data, "--index", f"artifacts/{model}",
                "--output", f"artifacts/probes/{name}", "--evaluate-test",
                *(["--annotations", args.annotations] if module == "detail_probes" else []),
            )
        for model, adapter in (
            ("clip", None),
            ("siglip", None),
            ("siglip_adapted", selection["metadata_adapter_path"]),
            ("siglip_visual_adapted", selection["visual_adapter_path"]),
        ):
            paired(model, "test", adapter)
            retrieval(model, "test")
        choice = selection["selected"]
        # Report the frozen selected setting and a predeclared detail-head ablation.
        retrieval(
            choice["model"],
            "test",
            choice["variant"],
            choice["probes"],
            choice["weight"],
            "selected_test",
        )
        for probe_name in ("siglip", "vision"):
            retrieval(
                "siglip",
                "test",
                "image_only",
                f"artifacts/probes/{probe_name}",
                selection["head_ablation_weight"],
                f"siglip_{probe_name}_test",
            )
        comparisons = (
            ("clip_vs_siglip", "clip_baseline_translated_test", "siglip_baseline_translated_test", "baseline", "baseline"),
            ("metadata_lora", "siglip_baseline_translated_test", "siglip_adapted_baseline_translated_test", "baseline", "baseline"),
            ("visual_lora", "siglip_baseline_translated_test", "siglip_visual_adapted_baseline_translated_test", "baseline", "baseline"),
            ("translated_vs_selected", "siglip_baseline_translated_test", "selected_test", "translated", choice["variant"]),
            ("base_vs_detail_heads", "siglip_siglip_test", "siglip_vision_test", "image_only", "image_only"),
        )
        for name, first, second, first_variant, second_variant in comparisons:
            run(
                f"compare_{name}", "-m", "curio.evaluate.compare",
                f"artifacts/evaluation/image_audit/{first}/per_query.jsonl",
                f"artifacts/evaluation/image_audit/{second}/per_query.jsonl",
                "--baseline-variant", first_variant,
                "--candidate-variant", second_variant,
                "--metrics", "ndcg@10,precision@5,ap@10,recall@20,mrr@10",
                "--output", f"artifacts/evaluation/image_audit/comparisons/{name}",
            )
        status["completed_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2))


def npmean(values):
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
