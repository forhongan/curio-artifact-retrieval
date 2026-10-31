import json
from pathlib import Path
import subprocess
import sys

import pytest

from curio.evaluate.compare import compare_runs


def write_run(tmp_path, name, rows, **overrides):
    directory = tmp_path / name
    directory.mkdir()
    metadata = {"query_sha256": "queries", "qrels_sha256": "qrels", "label_source": "metadata_silver",
                "gallery_sha256": "gallery", "gallery_count": 100,
                "label_sources": ["metadata_silver"], "model": {"encoder": name},
                "index": {"data_sha256": "dataset", "catalog_sha256": "catalog", "items": 100, "index_version": name}}
    metadata.update(overrides)
    (directory / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")
    path = directory / "per_query.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def row(qid, language="en", score=.5, variant="baseline"):
    return {"qid": qid, "language": language, "variant": variant, "ndcg@10": score, "ap@10": score, "degraded": False}


def test_paired_comparison_aligns_qid_language_and_keeps_negative_deltas(tmp_path):
    baseline = write_run(tmp_path, "a", [row("q1", "en", .2), row("q2", "en", .4), row("q1", "zh", .8), row("q2", "zh", .9)])
    candidate = write_run(tmp_path, "b", [row("q2", "zh", .7, "hybrid"), row("q2", "en", .6, "hybrid"), row("q1", "zh", .4, "hybrid"), row("q1", "en", .5, "hybrid")])
    output = tmp_path / "comparison"
    report = compare_runs(baseline, candidate, repetitions=100, output=output)
    comparisons = {r["language"]: r for r in report["comparisons"]}
    assert comparisons["en"]["delta"]["mean"] == pytest.approx(.25)
    assert comparisons["zh"]["delta"]["mean"] == pytest.approx(-.3)
    assert comparisons["en"]["qids"] == ["q1", "q2"]
    assert report["method"]["family_size"] == 2
    assert all(r["p_value_holm"] >= r["p_value"] for r in report["comparisons"])
    assert report["baseline"]["effective_variants"] == {"baseline": 4}
    assert len(report["baseline"]["report_sha256"]) == 64
    assert (output / "comparison.json").exists()
    assert "-0.3000" in (output / "comparison.md").read_text(encoding="utf-8")


def test_comparison_requires_variant_choice_for_multi_variant_file(tmp_path):
    a = write_run(tmp_path, "a", [row("q1", score=.2), row("q1", score=.7, variant="hybrid")])
    b = write_run(tmp_path, "b", [row("q1", score=.5)])
    with pytest.raises(ValueError, match="Select a variant explicitly"):
        compare_runs(a, b, repetitions=10)
    report = compare_runs(a, b, baseline_variant="hybrid", repetitions=10)
    assert report["comparisons"][0]["delta"]["mean"] == pytest.approx(-.2)


@pytest.mark.parametrize("key,value", [("query_sha256", "other-queries"), ("qrels_sha256", "other-qrels"), ("label_source", "human")])
def test_comparison_rejects_incompatible_query_and_label_provenance(tmp_path, key, value):
    a = write_run(tmp_path, "a", [row("q1")])
    b = write_run(tmp_path, "b", [row("q1")], **{key: value})
    with pytest.raises(ValueError, match=f"Incompatible {key}"):
        compare_runs(a, b, repetitions=10)


@pytest.mark.parametrize("field", ["data_sha256", "catalog_sha256", "items"])
def test_comparison_rejects_mismatched_index_gallery(tmp_path, field):
    a = write_run(tmp_path, "a", [row("q1")])
    b = write_run(tmp_path, "b", [row("q1")])
    manifest_path = b.with_name("metrics.json")
    metadata = json.loads(manifest_path.read_text())
    metadata["index"][field] = 99 if field == "items" else "different"
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=f"Incompatible index {field}"):
        compare_runs(a, b, repetitions=10)


@pytest.mark.parametrize("key,value", [("gallery_sha256", "other-gallery"), ("gallery_count", 60)])
def test_comparison_rejects_different_explicit_gallery_provenance(tmp_path, key, value):
    a = write_run(tmp_path, "a", [row("q1")])
    b = write_run(tmp_path, "b", [row("q1")], **{key: value})
    with pytest.raises(ValueError, match=f"Incompatible {key}"):
        compare_runs(a, b, repetitions=10)


def test_comparison_requires_gallery_fingerprint_even_if_both_are_absent(tmp_path):
    a = write_run(tmp_path, "a", [row("q1")], gallery_sha256=None)
    b = write_run(tmp_path, "b", [row("q1")], gallery_sha256=None)
    with pytest.raises(ValueError, match="Missing required provenance field gallery_sha256"):
        compare_runs(a, b, repetitions=10)


def test_comparison_verifies_declared_gallery_ids_against_fingerprint(tmp_path):
    a = write_run(tmp_path, "a", [row("q1")], gallery_count=1, gallery_ids=["met_1"])
    b = write_run(tmp_path, "b", [row("q1")], gallery_count=1, gallery_ids=["met_1"])
    with pytest.raises(ValueError, match="do not match declared gallery_sha256"):
        compare_runs(a, b, repetitions=10)


def test_comparison_rejects_unpaired_queries_and_duplicate_rows(tmp_path):
    a = write_run(tmp_path, "a", [row("q1"), row("q2")])
    b = write_run(tmp_path, "b", [row("q1")])
    with pytest.raises(ValueError, match="candidate missing 1"):
        compare_runs(a, b, repetitions=10)
    c = write_run(tmp_path, "c", [row("q1"), row("q1")])
    with pytest.raises(ValueError, match=r"Duplicate qid\+language"):
        compare_runs(a, c, repetitions=10)


def test_holm_family_includes_all_requested_metrics_and_languages(tmp_path):
    a = write_run(tmp_path, "a", [row("q1", "en"), row("q1", "zh")])
    b = write_run(tmp_path, "b", [row("q1", "en"), row("q1", "zh")])
    report = compare_runs(a, b, metrics=("ndcg@10", "ap@10"), repetitions=10)
    assert report["method"]["family_size"] == 4
    assert all(r["delta"]["mean"] == 0 and r["p_value_holm"] == 1 for r in report["comparisons"])


@pytest.mark.parametrize("bad_score", [float("nan"), float("inf"), None, True, 1.5])
def test_comparison_rejects_missing_nonfinite_and_invalid_metric_values(tmp_path, bad_score):
    a = write_run(tmp_path, "a", [row("q1")])
    b = write_run(tmp_path, "b", [row("q1", score=bad_score)])
    with pytest.raises(ValueError, match=r"(?:Missing/nonfinite|Out-of-range) ndcg@10"):
        compare_runs(a, b, repetitions=10)


def test_comparison_requires_provenance_instead_of_assuming_missing_hashes_match(tmp_path):
    a = write_run(tmp_path, "a", [row("q1")], qrels_sha256=None)
    b = write_run(tmp_path, "b", [row("q1")], qrels_sha256=None)
    with pytest.raises(ValueError, match="Missing required provenance field qrels_sha256"):
        compare_runs(a, b, repetitions=10)


def test_compare_cli_writes_json_and_markdown_without_loading_models(tmp_path):
    a = write_run(tmp_path, "a", [row("q1")])
    b = write_run(tmp_path, "b", [row("q1", score=.7)])
    output = tmp_path / "out"
    result = subprocess.run([sys.executable, "-m", "curio.evaluate.compare", str(a), str(b), "--output", str(output), "--repetitions", "10"], text=True, capture_output=True, check=False, cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["comparisons"][0]["delta"]["mean"] == pytest.approx(.2)
    assert (output / "comparison.md").exists()
