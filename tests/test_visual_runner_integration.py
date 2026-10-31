import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_visual_experiments.py"
    spec = importlib.util.spec_from_file_location("visual_runner_integration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runner_fixture(tmp_path, monkeypatch):
    runner = load_runner()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = []

    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    for path in ("data/snapshots/collection_v2.jsonl", "eval/visual_labels/annotations.jsonl"):
        write(path, {"synthetic": True})
    for split in ("dev", "test"):
        for name in ("queries.jsonl", "qrels.jsonl", "gallery_ids.json"):
            write(f"eval/visual_{split}/{name}", [split, name])
    for model in ("clip", "siglip", "siglip_adapted"):
        write(f"artifacts/{model}/manifest.json", {"model": model})
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        write(f"artifacts/metadata_adapter/{filename}", {"synthetic": True})
    write("artifacts/selection.json", {"adapter_path": "artifacts/metadata_adapter"})
    for filename in ("report.json", "ontology.json", "material.npz"):
        write(f"artifacts/probes/siglip/{filename}", {"synthetic": True})

    def subprocess_stub(command, **kwargs):
        assert command[1] == "-m"
        module, arguments = command[2], command[3:]
        calls.append((module, arguments))

        def option(name):
            return arguments[arguments.index(name) + 1]

        output = Path(option("--output"))
        if module == "curio.represent.train":
            score = .4 if option("--sampling") == "hard" else .3
            write(output / "run.json", {"config": {"output": str(output), "sampling": option("--sampling")},
                                        "best_dev": {"r@5": score, "mrr": score}, "best_epoch": 1,
                                        "improved_over_baseline": False})
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                write(output / "best_adapter" / filename, {"synthetic": True})
        elif module == "curio.index.build":
            write(output / "manifest.json", {"adapter_path": option("--adapter")})
        elif module in {"curio.represent.probes", "curio.represent.detail_probes"}:
            if "--evaluate-test" in arguments:
                before = (output / "report.json").read_bytes()
                write(output / "test_report.json", {"synthetic": True})
                assert (output / "report.json").read_bytes() == before
            else:
                assert "--defer-test" in arguments
                for filename in ("report.json", "ontology.json", "material.npz"):
                    write(output / filename, {"synthetic": True})
        elif module == "curio.evaluate.harness":
            if "--paired" in arguments:
                write(output, {"synthetic": True})
            else:
                assert option("--gallery-ids") == f"eval/visual_{'dev' if '_dev/' in option('--queries') else 'test'}/gallery_ids.json"
                write(output / "metrics.json", {"summary": [
                    {"variant": variant, "language": language, "degraded_queries": 0,
                     "metrics": {"ndcg@10": {"mean": .25}}}
                    for variant in option("--variants").split(",") for language in ("en", "zh")
                ]})
        elif module == "curio.evaluate.compare":
            write(output / "comparison.json", {"synthetic": True})
        else:
            raise AssertionError(module)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", subprocess_stub)

    def phase(name, resume=False):
        monkeypatch.setattr(sys, "argv", ["run_visual_experiments.py", "--phase", name,
                                         *(["--resume"] if resume else [])])
        runner.main()

    return runner, calls, phase


def test_visual_runner_commands_defer_then_evaluate_immutable_heads(runner_fixture):
    runner, calls, phase = runner_fixture
    phase("dev")
    selection = runner.read("artifacts/visual_selection.json")
    before = selection["model_files_sha256"]
    assert selection["selected"]["variant"] == "image_only"
    phase("test")
    assert runner.model_fingerprints(selection["visual_adapter_path"], selection["metadata_adapter_path"]) == before
    test_heads = [(module, args) for module, args in calls if "--evaluate-test" in args]
    assert len(test_heads) == 6
    assert all("--defer-test" not in args for _, args in test_heads)
    assert all("--annotations" in args and "--base-probes" not in args
               for module, args in test_heads if module.endswith("detail_probes"))
    assert all("test_report.json" not in name for name in before)
    assert len([1 for module, _ in calls if module == "curio.evaluate.compare"]) == 5


@pytest.mark.parametrize("changed,match", [("eval/visual_test/queries.jsonl", "Frozen visual test inputs changed"),
                                           ("artifacts/siglip/manifest.json", "Model, index or probe files changed")])
def test_visual_runner_freeze_survives_status_loss(runner_fixture, changed, match):
    _, calls, phase = runner_fixture
    phase("dev")
    phase("test")
    Path("artifacts/visual_experiment_status.json").unlink()
    Path(changed).write_text("changed synthetic input", encoding="utf-8")
    before = len(calls)
    with pytest.raises(ValueError, match=match):
        phase("test")
    assert len(calls) == before


def test_visual_runner_resume_rejects_changed_dev_inputs(runner_fixture):
    _, calls, phase = runner_fixture
    phase("dev")
    before = len(calls)
    Path("eval/visual_dev/qrels.jsonl").write_text("changed synthetic qrels", encoding="utf-8")
    with pytest.raises(ValueError, match="resume config mismatch"):
        phase("dev", resume=True)
    assert len(calls) == before
