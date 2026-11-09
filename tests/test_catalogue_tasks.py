import numpy as np

from curio.evaluate.catalogue_tasks import conjunction_truth, single_label
import importlib.util
from pathlib import Path


def load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_catalogue_tasks_v4.py"
    spec = importlib.util.spec_from_file_location("catalogue_task_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(**ontology):
    return {"derived_ontology": ontology}


def test_unknown_and_composite_do_not_become_negative_fine_materials():
    assert single_label(row(material=["ceramic"]), "fine_material") is None
    assert single_label(row(material=["ceramic", "porcelain"]), "fine_material") == "porcelain"
    assert single_label(row(material=["ivory", "wood"]), "fine_material") is None
    assert single_label(row(material=["textile", "silk"]), "broad_material") == "textile"


def test_tri_state_conjunction_can_reject_with_one_known_mismatch():
    rows = [row(object_type=["jar"], material=[]),
            row(object_type=["bowl"], material=[]),
            row(object_type=["jar"], material=["porcelain"])]
    truth = conjunction_truth(rows, [("form", "jar"), ("fine_material", "porcelain")])
    np.testing.assert_array_equal(truth, [-1, 0, 1])


def test_two_dynasties_are_unknown_but_nested_form_is_single():
    assert single_label(row(era=["song", "yuan"]), "era") is None
    assert single_label(row(object_type=["bottle vase", "vase"]), "form") == "vase"


def test_predicted_only_class_is_included_in_macro_metrics():
    report = load_runner().metrics(["a", "a"], ["a", "b"])
    assert report["truth_labels"] == ["a"]
    assert report["predicted_labels"] == ["a", "b"]
    assert report["labels_scored"] == ["a", "b"]
    assert report["support"]["b"] == 0
    # Class a has precision 1/2, recall 1 and F1 2/3; class b has F1 0.
    assert np.isclose(report["macro_f1"], 1 / 3)
