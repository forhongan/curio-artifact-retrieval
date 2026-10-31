import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def launcher(tmp_path):
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is required for the launcher integration test")
    script = tmp_path / "scripts/run.ps1"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/run.ps1", script)

    def write(path, value):
        path = tmp_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    write(".venv/Scripts/python.exe", {})
    write("web/dist/index.html", {})
    data = write("data/snapshots/collection_v2.jsonl", {"synthetic": True})
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    index = "artifacts/siglip_visual_adapted"
    adapter = "artifacts/visual_adapter"
    probes = "artifacts/probes/vision_visual_adapted"
    manifest = {"encoder": "siglip", "data_sha256": data_sha, "index_sha256": "index",
                "adapted": True, "adapter_path": adapter}
    paths = [write(f"{index}/{name}", manifest if name == "manifest.json" else {})
             for name in ("manifest.json", "image_embeddings.npy", "item_ids.json")]
    paths += [write(f"{adapter}/{name}", {}) for name in ("adapter_config.json", "adapter_model.safetensors")]
    probe_report = {"data_sha256": data_sha, "index_sha256": "index",
                    "tasks": {"shape": {"status": "trained"}}}
    paths += [write(f"{probes}/{name}", probe_report if name == "report.json" else {})
              for name in ("report.json", "ontology.json", "shape.npz")]
    selection = {"data_sha256": data_sha,
                 "selected": {"variant": "image_only", "model": "siglip_visual_adapted",
                              "probes": probes, "weight": .15},
                 "model_files_sha256": {path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in paths}}
    # Files for comparison models are not needed to deploy the selected model.
    selection["model_files_sha256"]["artifacts/unused_comparison/manifest.json"] = "unused"
    selection_path = write("artifacts/visual_selection.json", selection)

    def run(*arguments):
        wrapper = tmp_path / "check_launcher.ps1"
        settings = "\n".join("'" + arguments[i].lstrip("-") + "'='"
                             + arguments[i + 1].replace("'", "''") + "'"
                             for i in range(0, len(arguments), 2))
        wrapper.write_text(
            "$ErrorActionPreference = 'Stop'\n$launcherOptions=@{\n" + settings
            + "\n}\n& '" + str(script).replace("'", "''") + "' -CheckOnly @launcherOptions"
            + "\n@{metrics=$env:CURIO_METRICS; probes=$env:CURIO_PROBES} | ConvertTo-Json -Compress\n",
            encoding="utf-8",
        )
        return subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                               "-File", str(wrapper)], capture_output=True, text=True, encoding="utf-8",
                              env={**os.environ, "CURIO_METRICS": "stale_inherited_evaluation.json"}, timeout=30)

    return tmp_path, run, write, selection_path, manifest


def test_launcher_accepts_exact_selected_files_without_unused_comparisons(launcher):
    _, run, _, _, _ = launcher
    result = run()
    assert result.returncode == 0, result.stderr
    assert "Using frozen visual-dev production selection" in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])["metrics"] == "artifacts/evaluation/image_audit/selected_test/metrics.json"


@pytest.mark.parametrize("artifact", [
    "siglip_visual_adapted/image_embeddings.npy", "siglip_visual_adapted/item_ids.json",
    "siglip_visual_adapted/manifest.json", "visual_adapter/adapter_config.json",
    "visual_adapter/adapter_model.safetensors", "probes/vision_visual_adapted/report.json",
    "probes/vision_visual_adapted/ontology.json", "probes/vision_visual_adapted/shape.npz",
])
def test_launcher_rejects_changed_selected_artifacts(launcher, artifact):
    root, run, _, _, _ = launcher
    path = root / "artifacts" / artifact
    path.write_bytes(path.read_bytes() + b"\n")
    result = run()
    assert result.returncode != 0
    assert "Frozen visual artifact changed" in result.stderr


def test_launcher_rejects_missing_fingerprint(launcher):
    _, run, _, selection_path, _ = launcher
    selection = json.loads(selection_path.read_text())
    del selection["model_files_sha256"]["artifacts/siglip_visual_adapted/image_embeddings.npy"]
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    result = run()
    assert result.returncode != 0 and "lacks a fingerprint" in result.stderr


def test_launcher_preserves_explicit_model_override(launcher):
    root, run, write, _, manifest = launcher
    for name in ("manifest.json", "image_embeddings.npy", "item_ids.json"):
        write(f"artifacts/custom/{name}", manifest if name == "manifest.json" else {})
    (root / "artifacts/visual_adapter/adapter_model.safetensors").write_text("changed unused selection")
    result = run("-Encoder", "siglip", "-Index", "artifacts/custom")
    assert result.returncode == 0, result.stderr
    assert "Using frozen visual-dev production selection" not in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])["metrics"] is None


def test_launcher_preserves_explicit_probe_and_weight_overrides(launcher):
    root, run, write, _, manifest = launcher
    write("artifacts/custom_probes/report.json", {"data_sha256": manifest["data_sha256"], "index_sha256": "index"})
    (root / "artifacts/probes/vision_visual_adapted/shape.npz").write_text("changed unused probes")
    result = run("-Probes", "artifacts/custom_probes", "-ProbeWeight", "0.3")
    assert result.returncode == 0, result.stderr
    assert "probes=artifacts/custom_probes probeWeight=0.3" in result.stdout
    assert "explicit probe settings" in result.stdout
    assert json.loads(result.stdout.splitlines()[-1])["metrics"] is None


def test_launcher_preserves_explicit_empty_probe_override(launcher):
    root, run, _, _, _ = launcher
    (root / "artifacts/probes/vision_visual_adapted/shape.npz").write_text("changed unused probes")
    result = run("-Probes", "")
    assert result.returncode == 0, result.stderr
    environment = json.loads(result.stdout.splitlines()[-1])
    assert environment["metrics"] is None and not environment["probes"]
