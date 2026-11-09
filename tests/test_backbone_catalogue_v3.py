import importlib.util
from pathlib import Path

import numpy as np

from curio.evaluate.backbone_tasks import (
    domain_metric_means,
    domain_split,
    metric_values,
    paired_bootstrap,
)


def row(item_id, split, *, form=None, material=None):
    return {
        "item_id": item_id,
        "split": split,
        "object_group": item_id,
        "derived_ontology": {
            "object_type": [form] if form else [],
            "material": [material] if material else [],
        },
    }


def test_unknown_and_multivalued_catalogue_facets_are_excluded_not_negative():
    rows = [
        row("a", "train", form="vase"),
        row("b", "train", form="bowl"),
        row("c", "train", form=None),
        row("d", "dev", form="vase"),
        row("e", "dev", form="bowl"),
        row("f", "test", form="vase"),
        row("g", "test", form=None),
    ]
    split = domain_split(rows, "form", min_train=1)
    assert split.classes == ("bowl", "vase")
    assert split.indices["train"].tolist() == [0, 1]
    assert split.coverage["test"]["catalogue_unknown"] == 1
    assert split.coverage["test"]["supported"] == 1


def test_class_below_support_floor_does_not_silently_enter_metrics():
    rows = [
        row("a", "train", form="vase"),
        row("b", "train", form="vase"),
        row("c", "train", form="bowl"),
        row("d", "dev", form="vase"),
        row("e", "test", form="vase"),
    ]
    split = domain_split(rows, "form", min_train=2)
    assert split.classes == ("vase",)
    assert split.coverage["train"]["known_but_unsupported"] == 1


def test_multiclass_metrics_use_observed_classes_for_macro_f1():
    truth = np.asarray([0, 0, 1])
    prediction = np.asarray([0, 1, 1])
    result = metric_values(truth, prediction, ("bowl", "vase"))
    assert result["accuracy"] == 2 / 3
    assert result["balanced_accuracy"] == (1 + 0.5) / 2
    assert result["macro_f1"] == (2 / 3 + 2 / 3) / 2
    assert result["observed_classes"] == ["bowl", "vase"]


def test_absent_test_class_is_reported_in_all_class_macro_control():
    result = metric_values(np.asarray([0, 0]), np.asarray([0, 1]), ("bowl", "vase"))
    assert result["macro_f1"] == 1 / 3
    assert result["macro_f1_observed"] == 2 / 3
    assert result["macro_f1_all_classes"] == 1 / 3
    assert result["support"]["vase"] == 0


def test_domain_means_skip_unsupported_color_task():
    means = domain_metric_means({
        "form": {"status": "evaluable", "test": {"macro_f1": .7, "balanced_accuracy": .8, "accuracy": .9, "n": 3}},
        "color": {"status": "not_evaluable"},
    })
    assert means["evaluable_domains"] == 1
    assert means["macro_f1"] == .7
    assert means["evaluated_item_domain_predictions"] == 3


def test_paired_bootstrap_reports_siglip2_reference_delta():
    truth = np.asarray([0, 0, 1, 1])
    predictions = {
        "siglip2": np.asarray([0, 0, 1, 1]),
        "clip": np.asarray([0, 1, 1, 0]),
    }
    result = paired_bootstrap(truth, predictions, ("a", "b"), repetitions=50)
    assert result["siglip2"]["vs_siglip2"]["accuracy_delta"]["mean"] == 0
    assert result["clip"]["vs_siglip2"]["accuracy_delta"]["mean"] < 0


def test_runner_refuses_to_overwrite_existing_research_directory():
    path = Path(__file__).resolve().parents[1] / "scripts/run_backbone_catalogue_v3.py"
    spec = importlib.util.spec_from_file_location("backbone_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.BACKBONES) == {
        "clip", "siglip1", "siglip2", "siglip2_lora", "siglip2_visual_lora", "dino"
    }
    assert module.BACKBONES["siglip2_visual_lora"]["production"] is False
    assert module.BACKBONES["siglip2_lora"]["production"] is True
