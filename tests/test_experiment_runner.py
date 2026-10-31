import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from curio.evaluate import benchmark


def load_runner():
    source = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("curio_experiment_runner_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_checks_benchmark_capacity_before_any_gpu_command(tmp_path, monkeypatch):
    runner = load_runner()
    rows = [{"item_id": f"v{index}", "split": "train", "object_type": "vase", "material": "porcelain"} for index in range(3)]
    rows.extend({"item_id": f"b{index}", "split": "dev", "object_type": "bowl", "material": "bronze"} for index in range(3))
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    calls = []

    def subprocess_stub(command, **kwargs):
        calls.append(command)
        assert command[1:3] == ["-m", "curio.evaluate.benchmark"]
        assert command[command.index("--dev") + 1] == "2"
        assert command[command.index("--test") + 1] == "2"
        with pytest.raises(ValueError, match="Only 2 distinct informative graded-qrels groups"):
            benchmark.build(data, tmp_path / "eval", n_dev=2, n_test=2)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", subprocess_stub)
    monkeypatch.setattr(sys, "argv", ["run_experiments.py", "--data", str(data), "--dev-queries", "2", "--test-queries", "2"])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="benchmark failed"):
        runner.main()
    assert len(calls) == 1
    assert not (tmp_path / "eval").exists()


def test_frozen_benchmark_hashes_cover_both_splits_and_refuse_postselection_edits(tmp_path):
    runner = load_runner()
    output = tmp_path / "eval"
    output.mkdir()
    names = ["manifest.json", "queries_dev.jsonl", "queries_test.jsonl", "qrels_dev.jsonl", "qrels_test.jsonl"]
    for name in names:
        (output / name).write_text(name, encoding="utf-8")
    selection = {"benchmark_files_sha256": runner.benchmark_fingerprints(tmp_path)}
    assert len(selection["benchmark_files_sha256"]) == 5
    assert selection["benchmark_files_sha256"]["queries_test.jsonl"] == hashlib.sha256(b"queries_test.jsonl").hexdigest()
    runner.verify_frozen_benchmark(selection, tmp_path)
    (output / "qrels_test.jsonl").write_text("modified after dev selection", encoding="utf-8")
    with pytest.raises(ValueError, match="Benchmark changed"):
        runner.verify_frozen_benchmark(selection, tmp_path)
    with pytest.raises(ValueError, match="lacks frozen provenance"):
        runner.verify_frozen_benchmark({}, tmp_path)


def test_runner_rejects_old_or_insufficient_benchmark_protocol(tmp_path):
    runner = load_runner()
    directory = tmp_path / "eval"
    directory.mkdir()
    path = directory / "manifest.json"
    config = {"data_sha256": "data-hash", "dev_queries": 2, "test_queries": 4}
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Stale benchmark protocol"):
        runner.verify_benchmark_manifest(config, tmp_path)
    manifest = {**config, "protocol_version": "metadata-silver-v2-informative-qrels-groups", "distinct_graded_qrels": 5}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks enough distinct"):
        runner.verify_benchmark_manifest(config, tmp_path)
    manifest["distinct_graded_qrels"] = 6
    path.write_text(json.dumps(manifest), encoding="utf-8")
    runner.verify_benchmark_manifest(config, tmp_path)
