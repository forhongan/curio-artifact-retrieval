import numpy as np

from curio.evaluate.catalogue_boundary import (
    QuerySpec,
    aggregate_query_metrics,
    build_query_specs,
    explicit_metrics,
    query_dataset,
    rank_percentile,
    truth_for_atoms,
)
from curio.evaluate.probe_sets import (
    explicit_probe_evaluation,
    supported_atoms,
    two_stage_summary,
)


def row(item_id, split, *, form=None, material=None):
    return {
        "item_id": item_id,
        "split": split,
        "derived_ontology": {
            "object_type": [form] if form else [],
            "material": [material] if material else [],
        },
    }


def test_tri_state_truth_excludes_unknown_and_rejects_known_mismatch():
    atoms = (("form", "vase"), ("fine_material", "porcelain"))
    assert truth_for_atoms(row("a", "dev", form="vase", material="porcelain"), atoms) == 1
    assert truth_for_atoms(row("b", "dev", form="bowl", material=None), atoms) == 0
    assert truth_for_atoms(row("c", "dev", form="vase", material="ceramic"), atoms) == -1


def test_query_dataset_drops_unknowns_without_turning_them_into_negatives():
    rows = [
        row("a", "dev", form="vase", material="porcelain"),
        row("b", "dev", form="bowl", material="porcelain"),
        row("c", "dev", form="vase", material="ceramic"),
    ]
    spec = QuerySpec("q", "q", (("form", "vase"), ("fine_material", "porcelain")), "pair", 1)
    data = query_dataset(rows, spec, "dev")
    assert data["ids"] == ["a", "b"]
    assert data["relevant"] == {"a"}
    assert data["unknown_excluded"] == 1


def test_query_dataset_indices_address_full_embedding_matrix():
    rows = [row("train1", "train", form="bowl"),
            row("test1", "test", form="vase"),
            row("dev1", "dev", form="bowl"),
            row("test2", "test", form="bowl")]
    spec = QuerySpec("q", "vase", (("form", "vase"),), "single_form", 1)
    data = query_dataset(rows, spec, "test")
    assert data["ids"] == ["test1", "test2"]
    assert data["indices"].tolist() == [1, 3]
    assert [rows[int(i)]["item_id"] for i in data["indices"]] == data["ids"]


def test_development_query_builder_does_not_read_test_and_creates_no_match_controls():
    rows = [
        row(f"d{i}", "dev", form="vase", material="porcelain") for i in range(3)
    ] + [
        row("d9", "dev", form="bowl", material="jade"),
        row("t1", "test", form="vase", material="porcelain"),
    ]
    specs = build_query_specs(rows, max_combinations=10, max_no_match=10)
    assert any(spec.qid == "single_form_vase" for spec in specs)
    # There must be ten wholly known dev negatives for a no-match constraint.
    assert not any(spec.family.startswith("nomatch_") for spec in specs)


def test_explicit_metrics_reports_false_positives_negatives_and_empty_score():
    metrics = explicit_metrics({"a", "x"}, {"a", "b"})
    assert metrics["tp"] == 1 and metrics["fp"] == 1 and metrics["fn"] == 1
    assert metrics["exact_set_match"] == 0
    empty = explicit_metrics(set(), set())
    assert empty["no_match_correct"] == 1


def test_query_aggregate_treats_empty_as_a_successful_query_score():
    positive = explicit_metrics({"a"}, {"a"})
    no_match = explicit_metrics(set(), set())
    report = aggregate_query_metrics([positive, no_match])
    assert report["macro_f1"] == 1.0
    assert report["no_match_accuracy"] == 1.0
    assert report["macro_set_score"] == 1.0


def test_rank_percentile_preserves_order_and_range():
    result = rank_percentile(np.array([0.1, 0.5, 0.2]))
    np.testing.assert_allclose(result, [0.0, 1.0, 0.5])


def probe_report():
    return {
        "tasks": {
            "object_type": {
                "classes": ["vase", "bowl"],
                "models": {
                    "vase": {"selected_threshold": 0.7},
                    "bowl": {"selected_threshold": 0.6},
                },
            }
        }
    }


def test_supported_atoms_are_explicit_and_never_silently_guessed():
    supported, missing = supported_atoms((("form", "vase"),), probe_report())
    assert supported and missing == []
    supported, missing = supported_atoms((("form", "vase"), ("motif", "dragon")), probe_report())
    assert not supported and missing == ["motif/dragon"]


def test_gate_and_per_class_thresholds_are_separate_decisions():
    spec = QuerySpec("q", "vase", (("form", "vase"),), "single_form", 1)
    data = {
        "ids": ["a", "b"], "relevant": {"a"},
        "unknown_excluded": 0,
        "probabilities": {"object_type": np.array([[.8], [.65]], dtype=np.float32)},
    }
    rows = explicit_probe_evaluation(
        [{"spec": spec, "data": data}], probe_report(), gate_threshold=.1
    )
    # Per-class threshold admits only a; the gate does not replace it.
    assert rows[0]["tp"] == 1 and rows[0]["fp"] == 0
    summary = two_stage_summary(rows)
    assert summary["gate_tp"] == 1 and summary["positive_nonempty_rate"] == 1.0


def test_optional_class_winner_rule_rejects_an_unrelated_binary_tail():
    spec = QuerySpec("q", "vase", (("form", "vase"),), "single_form", 1)
    data = {
        "ids": ["a"], "relevant": {"a"}, "unknown_excluded": 0,
        "probabilities": {"object_type": np.array([[.75, .95]], dtype=np.float32)},
    }
    rows = explicit_probe_evaluation(
        [{"spec": spec, "data": data}], probe_report(), gate_threshold=.1,
        require_class_winner=True,
    )
    assert rows[0]["tp"] == 0 and rows[0]["fn"] == 1
