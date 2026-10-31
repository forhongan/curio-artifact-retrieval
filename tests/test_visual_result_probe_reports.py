import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest


SCRIPT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/summarize_visual_results.py"))


def fixture():
    report = {
        "data_sha256": "data", "index_sha256": "index", "index_manifest_sha256": "manifest",
        "detail_annotations_sha256": "annotations", "test_deferred": True,
        "artifacts_sha256": {"colors.npz": "weights"},
        "tasks": {"colors": {"status": "trained", "selected_C": .1, "test": None},
                  "shape": {"status": "skipped", "classes": []}},
    }
    digest = hashlib.sha256(b"frozen training report").hexdigest()
    test_report = {
        **{key: report[key] for key in ("data_sha256", "index_sha256", "index_manifest_sha256", "detail_annotations_sha256")},
        "training_report_sha256": digest, "artifacts_sha256": {"colors.npz": "weights"},
        "tasks": {"colors": {"test": {"blue": {"support": 8, "f1": .75, "roc_auc": None}}}},
    }
    return report, test_report, digest


def test_independent_test_metrics_join_without_changing_frozen_training_report():
    report, test_report, digest = fixture()
    before = copy.deepcopy(report)
    merged = SCRIPT["merge_probe_test_report"](report, test_report, digest)
    assert merged["tasks"]["colors"]["test"]["blue"]["f1"] == .75
    assert merged["tasks"]["colors"]["selected_C"] == .1
    assert merged["tasks"]["shape"]["status"] == "skipped"
    assert report == before


@pytest.mark.parametrize("field,value", [("training_report_sha256", "other"),
                                         ("data_sha256", "other"), ("index_sha256", "other"),
                                         ("detail_annotations_sha256", "other"),
                                         ("artifacts_sha256", {"colors.npz": "other"}),
                                         ("tasks", {})])
def test_independent_test_metrics_reject_stale_frozen_sources(field, value):
    report, test_report, digest = fixture()
    test_report[field] = value
    with pytest.raises(ValueError, match="Probe test"):
        SCRIPT["merge_probe_test_report"](report, test_report, digest)


def frozen_retrieval_fixture():
    prefix = "artifacts/probes/vision"
    files = {"artifacts/siglip/manifest.json": "manifest", f"{prefix}/report.json": "report",
             f"{prefix}/ontology.json": "ontology", f"{prefix}/shape.npz": "shape"}
    report = {"index_manifest_sha256": "manifest", "summary": [{"variant": "image_only"}],
              "probes": {"configured": True, "path": prefix, "weight": .05,
                         "files_sha256": {"report.json": "report", "ontology.json": "ontology", "shape.npz": "shape"}}}
    return report, {"model": "siglip", "probes": prefix, "weight": .05,
                    "variants": ["image_only"], "model_files": files}


@pytest.mark.parametrize("changed", [None, "weight", "path", "binary", "manifest", "variant"])
def test_retrieval_matches_frozen_model_head_and_weight(changed):
    report, expected = frozen_retrieval_fixture()
    if changed == "weight":
        report["probes"]["weight"] = .3
    elif changed == "path":
        report["probes"]["path"] = "artifacts/probes/siglip"
    elif changed == "binary":
        report["probes"]["files_sha256"]["shape.npz"] = "other"
    elif changed == "manifest":
        report["index_manifest_sha256"] = "other"
    elif changed == "variant":
        report["summary"] = [{"variant": "translated"}]
    if changed:
        with pytest.raises(ValueError, match="Retrieval"):
            SCRIPT["verify_frozen_retrieval"](report, **expected)
    else:
        SCRIPT["verify_frozen_retrieval"](report, **expected)


def test_summary_refuses_changed_selection_before_overwriting_outputs(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    selection = {"protocol_version": "visual-v1-deferred-test-heads", "data_sha256": "synthetic",
                 "annotations_sha256": "synthetic", "epochs": 12, "model_files_sha256": {}}
    (artifacts / "visual_selection.json").write_text(json.dumps(selection), encoding="utf-8")
    (artifacts / "visual_test_freeze.json").write_text(json.dumps({
        "config": {"epochs": 12}, "selection_sha256": "old selection", "test_files_sha256": {}
    }), encoding="utf-8")
    output = tmp_path / "result.md"
    output.write_text("preserve existing report", encoding="utf-8")
    with pytest.raises(ValueError, match="different dev selection"):
        SCRIPT["build"](root=tmp_path, output=output)
    assert output.read_text() == "preserve existing report"


def test_missing_frozen_artifact_remains_pending_evidence(tmp_path):
    evidence = SCRIPT["Evidence"](tmp_path)
    evidence.verify_hash(tmp_path / "missing.npz", "frozen hash", "probe")
    assert evidence.missing == ["missing.npz"]


def test_summary_renders_current_system_check_concurrency_schema(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "system_acceptance.json").write_text(json.dumps({
        "health_after": {"status": "ok"}, "warm_sequential": {"image_only": {"p95_ms": 10}},
        "concurrent_image_only": {"workers": 2, "requests_per_second": 9},
    }), encoding="utf-8")
    output = tmp_path / "result.md"
    SCRIPT["build"](root=tmp_path, output=output)
    text = output.read_text(encoding="utf-8")
    assert '"workers": 2' in text and '"requests_per_second": 9' in text
    assert "unverified_configuration" in text and "不能证明当前冻结生产配置" in text


def system_acceptance_fixture():
    selection = {"data_sha256": "data", "selected": {"variant": "image_only", "model": "siglip", "probes": "artifacts/probes/vision", "weight": .05},
                 "model_files_sha256": {"artifacts/siglip/manifest.json": "manifest", "artifacts/probes/vision/report.json": "report"}}
    records = [{"phase": phase, "request_id": phase, "request": {"query": "fixture", "variant": "image_only"},
                "wall_ms": 1, "degraded": False} for phase in ("warm_sequential", "concurrent_image_only")]
    report = {"schema": "curio-system-acceptance-v2", "health_after": {"status": "ok"}, "requested": {"count_per_phase": 1},
              "request_records": records, "runtime_config": {"data_sha256": "data", "default_variant": "image_only", "encoder_name": "siglip",
                 "index_manifest_sha256": "manifest", "probes": {"configured": True, "available": True, "weight": .05, "files_sha256": {"report.json": "report"}}}}
    return report, selection


@pytest.mark.parametrize("changed", [None, "manifest", "data", "weight", "probe_files", "missing_requests", "concurrent_variant"])
def test_system_acceptance_must_match_frozen_production_and_actual_requests(changed):
    report, selection = system_acceptance_fixture()
    if changed == "manifest":
        report["runtime_config"]["index_manifest_sha256"] = "other"
    elif changed == "data":
        report["runtime_config"]["data_sha256"] = "other"
    elif changed == "weight":
        report["runtime_config"]["probes"]["weight"] = .3
    elif changed == "probe_files":
        report["runtime_config"]["probes"]["files_sha256"] = {"report.json": "other"}
    elif changed == "missing_requests":
        report.pop("request_records")
    elif changed == "concurrent_variant":
        report["request_records"][-1]["request"]["variant"] = "hybrid"
    result = SCRIPT["classify_system_acceptance"](report, selection)
    assert result["scope"] == ("current_frozen_production" if changed is None else "not_current_production_evidence")
    assert bool(result["reasons"]) is (changed is not None)
