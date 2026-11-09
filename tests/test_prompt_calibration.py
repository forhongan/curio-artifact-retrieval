import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_prompt_calibration.py"
SPEC = importlib.util.spec_from_file_location("run_prompt_calibration", SCRIPT)
prompts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prompts
SPEC.loader.exec_module(prompts)


def test_prompt_registry_covers_requested_families_and_has_unique_names():
    names = [spec.name for spec in prompts.PROMPT_SPECS]
    assert len(names) == len(set(names))
    assert {spec.family for spec in prompts.PROMPT_SPECS} == {
        "baseline",
        "case_punctuation",
        "article",
        "prefix",
        "suffix",
        "caption_like",
    }


@pytest.mark.parametrize(
    ("name", "source", "expected"),
    [
        ("raw", "  A blue bowl.  ", "A blue bowl."),
        ("lowercase", "A Blue Bowl", "a blue bowl"),
        ("capitalize_first", "AN OLD BOWL", "An old bowl"),
        ("terminal_period", "A blue bowl...", "A blue bowl."),
        ("remove_terminal_punctuation", "A blue bowl！", "A blue bowl"),
        ("strip_leading_article", "An old bowl", "old bowl"),
        ("prepend_the", "old bowl", "the old bowl"),
        ("prefix_a_photo_of", "blue bowl", "a photo of blue bowl"),
        ("suffix_plain_background", "blue bowl.", "blue bowl, photographed against a plain background"),
        ("caption_image_showing", "blue bowl", "an image showing blue bowl"),
    ],
)
def test_prompt_rendering(name, source, expected):
    assert prompts.render_prompt(source, name) == expected


def test_article_removal_is_explicit_noop_for_chinese():
    assert prompts.render_prompt("青花瓷碗", "strip_leading_article") == "青花瓷碗"
    assert prompts.render_prompt("青花瓷碗", "raw") == "青花瓷碗"


def test_selection_uses_dev_ndcg_and_prefers_raw_on_tie():
    results = {
        spec.name: {
            "summary": {"nDCG@10": {"mean": 0.5}, "P@5": {"mean": 0.2}}
        }
        for spec in prompts.PROMPT_SPECS
    }
    assert prompts.select_variant(results)[0] == "raw"
    results["prefix_image_of"]["summary"]["nDCG@10"]["mean"] = 0.51
    results["prefix_image_of"]["summary"]["P@5"]["mean"] = 0.1
    assert prompts.select_variant(results)[0] == "prefix_image_of"


def test_ranking_metrics_returns_complete_metric_contract():
    qrels = {"a": 3, "b": 2, "c": 0, "d": 1}
    metrics = prompts.ranking_metrics(["a", "b", "d", "c"], qrels)
    assert set(metrics) == set(prompts.SUMMARY_METRICS)
    assert metrics["P@5"] == pytest.approx(0.4)
    assert metrics["nDCG@5"] == pytest.approx(1.0)
    assert metrics["nDCG@10"] == pytest.approx(1.0)
    assert metrics["MRR@10"] == pytest.approx(1.0)


def test_selection_manifest_digest_verifies_without_newline_rewrite(tmp_path):
    manifest = {
        "prompt_registry_sha256": prompts.canonical_sha256(
            [prompts.asdict(spec) for spec in prompts.PROMPT_SPECS]
        ),
        "selected": {"en": "raw", "zh": "raw"},
    }
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (tmp_path / "selection_manifest.json").write_bytes(payload)
    (tmp_path / "selection_manifest.sha256").write_bytes(
        (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii")
    )
    assert prompts.load_frozen_selection(tmp_path)["selected"] == manifest["selected"]


def test_paired_difference_is_exactly_zero_for_cosmetic_noop():
    result = prompts.paired_difference([0.4, 0.5, 0.6], [0.4, 0.5, 0.6])
    assert result["difference"] == 0
    assert (result["lower"], result["upper"]) == (0, 0)
    assert result["p_value"] == 1
    assert result["zero_differences"] == 3


def test_paired_difference_is_deterministic_and_preserves_direction():
    result = prompts.paired_difference([0.0, 0.1, 0.2, 0.3], [0.1, 0.3, 0.2, 0.4])
    assert result["difference"] == pytest.approx(0.1)
    assert result == prompts.paired_difference([0.0, 0.1, 0.2, 0.3], [0.1, 0.3, 0.2, 0.4])


def test_single_nonzero_pair_has_unit_two_sided_p_value():
    result = prompts.paired_difference([1, 1, 1, 1], [0, 1, 1, 1])
    assert result["difference"] == pytest.approx(-0.25)
    assert result["p_value"] == 1


def test_query_metrics_csv_preserves_one_row_per_template_query(tmp_path):
    result = {
        "en": {
            "split": "visual_dev",
            "templates": {
                "raw": {
                    "rows": [
                        {"qid": "q1", "top10": ["a"], **{m: 0.5 for m in prompts.SUMMARY_METRICS}}
                    ]
                }
            },
        }
    }
    path = tmp_path / "queries.csv"
    prompts._write_query_metrics_csv(path, result)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("split,language,template,qid,P@5,nDCG@5,nDCG@10")
    assert lines[1].startswith("visual_dev,en,raw,q1,")


def test_frozen_manifest_rejects_changed_selection(tmp_path):
    manifest = {
        "prompt_registry_sha256": prompts.canonical_sha256(
            [prompts.asdict(spec) for spec in prompts.PROMPT_SPECS]
        ),
        "selected": {"en": "raw", "zh": "raw"},
    }
    payload = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (tmp_path / "selection_manifest.json").write_bytes(payload)
    (tmp_path / "selection_manifest.sha256").write_bytes(
        (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii")
    )
    manifest["selected"]["en"] = "prefix_image_of"
    (tmp_path / "selection_manifest.json").write_bytes(
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError, match="SHA-256"):
        prompts.load_frozen_selection(tmp_path)
