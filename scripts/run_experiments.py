"""Real GPU experiments: select on dev, freeze choices, then evaluate test."""

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


def benchmark_fingerprints(root=ROOT):
    names = ["manifest.json"] + [f"{kind}_{split}.jsonl" for split in ("dev", "test") for kind in ("queries", "qrels")]
    return {name: hashlib.sha256((Path(root) / "eval" / name).read_bytes()).hexdigest() for name in names}


def verify_frozen_benchmark(selection, root=ROOT):
    expected = selection.get("benchmark_files_sha256")
    if not isinstance(expected, dict) or expected != benchmark_fingerprints(root):
        raise ValueError("Benchmark changed or lacks frozen provenance; freeze dev selection before any test evaluation")


def verify_benchmark_manifest(config, root=ROOT):
    manifest = json.loads((Path(root) / "eval" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != "metadata-silver-v2-informative-qrels-groups":
        raise ValueError("Stale benchmark protocol; regenerate informative qrels groups before GPU experiments")
    for key in ("data_sha256", "dev_queries", "test_queries"):
        if manifest.get(key) != config[key]:
            raise ValueError(f"Benchmark/config mismatch: {key}")
    if manifest.get("distinct_graded_qrels", 0) < config["dev_queries"] + config["test_queries"]:
        raise ValueError("Benchmark lacks enough distinct informative qrels groups")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    p.add_argument("--phase", choices=["dev", "test", "all"], default="dev")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dev-queries", type=int, default=40)
    p.add_argument("--test-queries", type=int, default=80)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.dev_queries < 1 or args.test_queries < 1:
        p.error("dev-queries and test-queries must be positive; counts are never silently reduced")
    os.chdir(ROOT)
    from curio.represent.train import read_items
    from curio.represent.validation import validate_object_splits

    validate_object_splits(read_items(args.data))

    def sha():
        return hashlib.sha256(Path(args.data).read_bytes()).hexdigest()

    config = {"data_sha256": sha(), "epochs": args.epochs, "batch_size": args.batch_size,
              "dev_queries": args.dev_queries, "test_queries": args.test_queries}
    logs = ROOT / "logs/experiments"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / "artifacts/experiment_status.json"
    selection_path = ROOT / "artifacts/selection.json"
    status_path.parent.mkdir(exist_ok=True)
    status = (
        json.loads(status_path.read_text(encoding="utf-8"))
        if (args.resume or args.phase == "test") and status_path.exists()
        else {"steps": {}}
    )
    if status.get("config") and status["config"] != config:
        raise ValueError("Resume data/config mismatch")
    status["config"] = config
    status.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }

    def run(name, *command):
        if sha() != config["data_sha256"]:
            raise RuntimeError("Catalog changed; use an immutable snapshot")
        if args.resume and status["steps"].get(name, {}).get("status") == "complete":
            if status["steps"][name].get("command") != list(command):
                raise ValueError(f"Resume command mismatch for {name}")
            print(f"SKIP {name}", flush=True)
            return
        started = time.perf_counter()
        status["steps"][name] = {
            "status": "running",
            "command": list(command),
            "log": str(logs / f"{name}.log"),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"START {name}", flush=True)
        with (logs / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, *command], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        status["steps"][name].update(
            status="complete" if result.returncode == 0 else "failed",
            seconds=time.perf_counter() - started,
            returncode=result.returncode,
        )
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(
            f"END {name} rc={result.returncode} seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        if result.returncode:
            raise RuntimeError(f"{name} failed: {logs / (name + '.log')}")

    adapter = None

    def index(name, encoder, adapted=False):
        run(
            f"index_{name}",
            "-m",
            "curio.index.build",
            "--data",
            args.data,
            "--encoder",
            encoder,
            "--output",
            f"artifacts/{name}",
            "--batch-size",
            str(args.batch_size),
            *(["--adapter", adapter] if adapted else []),
        )

    def paired(name, encoder, split, adapted=False):
        run(
            f"paired_{name}_{split}",
            "-m",
            "curio.evaluate.harness",
            "--paired",
            "--data",
            args.data,
            "--encoder",
            encoder,
            "--split",
            split,
            "--output",
            f"artifacts/evaluation/paired_{name}_{split}.json",
            *(["--adapter", adapter] if adapted else []),
        )

    def silver(name, encoder, split, output=None, weight=None):
        output = output or f"{name}_{split}"
        extra = (
            [
                "--variants",
                "visual",
                "--probes",
                "artifacts/probes/siglip",
                "--probe-weight",
                str(weight),
            ]
            if weight is not None
            else []
        )
        run(
            f"silver_{output}",
            "-m",
            "curio.evaluate.harness",
            "--data",
            args.data,
            "--encoder",
            encoder,
            "--index",
            f"artifacts/{name}",
            "--queries",
            f"eval/queries_{split}.jsonl",
            "--qrels",
            f"eval/qrels_{split}.jsonl",
            "--output",
            f"artifacts/evaluation/{output}",
            *extra,
        )

    if args.phase in {"dev", "all"}:
        # Fail before model indexing/training if informative label groups cannot
        # support the explicitly requested semantic dev/test query counts.
        run("benchmark", "-m", "curio.evaluate.benchmark", "--data", args.data,
            "--dev", str(args.dev_queries), "--test", str(args.test_queries))
        verify_benchmark_manifest(config, ROOT)
        frozen_benchmark = benchmark_fingerprints(ROOT)
        for encoder in ("clip", "siglip"):
            index(encoder, encoder)
            paired(encoder, encoder, "dev")
        for encoder in ("clip", "siglip"):
            silver(encoder, encoder, "dev")
        training_candidates = []
        for sampling in ("random", "hard"):
            train_output = f"artifacts/training/siglip_lora_{sampling}"
            run(
                f"train_siglip_{sampling}",
                "-m",
                "curio.represent.train",
                "--data",
                args.data,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--sampling",
                sampling,
                "--output",
                train_output,
            )
            training_candidates.append(
                json.loads(Path(train_output, "run.json").read_text(encoding="utf-8"))
            )
        training = max(
            training_candidates, key=lambda r: (r["best_dev"]["r@5"], r["best_dev"]["mrr"])
        )
        adapter = str(Path(training["config"]["output"]) / "best_adapter")
        index("siglip_adapted", "siglip", True)
        paired("adapted", "siglip", "dev", True)
        silver("siglip_adapted", "siglip", "dev")
        run(
            "probes_siglip",
            "-m",
            "curio.represent.probes",
            "--data",
            args.data,
            "--index",
            "artifacts/siglip",
            "--output",
            "artifacts/probes/siglip",
            "--defer-test",
        )
        candidates = []
        for weight in (0.0, 0.05, 0.15, 0.3):
            output = f"visual_w{weight:g}_dev"
            silver("siglip", "siglip", "dev", output, weight)
            report = json.loads(
                Path(f"artifacts/evaluation/{output}/metrics.json").read_text(encoding="utf-8")
            )
            if any(row["degraded_queries"] for row in report["summary"]):
                raise RuntimeError("Visual dev run degraded; fix probes before selection")
            score = sum(row["metrics"]["ndcg@10"]["mean"] for row in report["summary"]) / len(
                report["summary"]
            )
            candidates.append({"weight": weight, "dev_bilingual_ndcg@10": score})
        best_weight = max(
            candidates, key=lambda row: (row["dev_bilingual_ndcg@10"], -row["weight"])
        )["weight"]
        selection = {
            **config,
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "adapted_improves_paired_dev": training["improved_over_baseline"],
            "adapter_path": adapter,
            "training_candidates": [
                {
                    "sampling": r["config"]["sampling"],
                    "best_epoch": r["best_epoch"],
                    "best_dev": r["best_dev"],
                    "improved_over_baseline": r["improved_over_baseline"],
                }
                for r in training_candidates
            ],
            "paired_selection_rule": "dev R@5 then MRR; retain zero-shot if adapter does not improve",
            "probe_weight": best_weight,
            "probe_candidates": candidates,
            "probe_selection_rule": "mean EN/ZH metadata-silver dev nDCG@10; ties prefer lower weight",
            "benchmark_files_sha256": frozen_benchmark,
            "warning": "Silver dev tunes catalog compliance, not independent visual relevance. Test not used.",
        }
        verify_frozen_benchmark(selection, ROOT)
        verify_benchmark_manifest(config, ROOT)
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("DEV COMPLETE: artifacts/selection.json frozen", flush=True)

    if args.phase in {"test", "all"}:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if any(selection.get(k) != v for k, v in config.items()):
            raise ValueError("Selection/data/config mismatch")
        verify_frozen_benchmark(selection, ROOT)
        verify_benchmark_manifest(config, ROOT)
        adapter = selection["adapter_path"]
        run(
            "probes_siglip_test", "-m", "curio.represent.probes",
            "--data", args.data, "--index", "artifacts/siglip",
            "--output", "artifacts/probes/siglip", "--evaluate-test",
        )
        for encoder in ("clip", "siglip"):
            paired(encoder, encoder, "test")
            silver(encoder, encoder, "test")
        paired("adapted", "siglip", "test", True)
        silver("siglip_adapted", "siglip", "test")
        silver("siglip", "siglip", "test", "visual_test", selection["probe_weight"])
        for name, first, second, va, vb in [
            ("clip_vs_siglip", "clip_test", "siglip_test", "baseline", "baseline"),
            ("siglip_vs_adapted", "siglip_test", "siglip_adapted_test", "baseline", "baseline"),
            ("dense_vs_hybrid", "siglip_test", "siglip_test", "baseline", "hybrid"),
            ("hybrid_vs_visual", "siglip_test", "visual_test", "hybrid", "visual"),
        ]:
            run(
                f"compare_{name}",
                "-m",
                "curio.evaluate.compare",
                f"artifacts/evaluation/{first}/per_query.jsonl",
                f"artifacts/evaluation/{second}/per_query.jsonl",
                "--baseline-variant",
                va,
                "--candidate-variant",
                vb,
                "--metrics",
                "ndcg@10,precision@5,ap@10,recall@20,mrr@10",
                "--output",
                f"artifacts/evaluation/comparisons/{name}",
            )
        status["completed_at"] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
