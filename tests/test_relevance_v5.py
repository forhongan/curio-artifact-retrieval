import hashlib
import json
from pathlib import Path

from curio.evaluate.relevance import judge, retrieval_metrics


def test_compatible_multi_forms_and_unknown():
    assert judge({"form": ["waterpot"]}, [("form_family", "bowl")])["status"] == "relevant"
    assert judge({"form": ["bottle", "vase"]}, [("form", "vase")])["grade"] == 3
    assert judge({"form": None}, [("form_family", "bowl")])["grade"] == -1
    assert judge({}, [])["grade"] == -1


def test_all_constraints_required_and_unknown_not_negative():
    facts = {
        "era": ["qing"],
        "visible_tags": ["floral", "dragon"],
        "unknown_tag_families": ["color"],
    }
    assert judge(facts, [("tag", "dragon"), ("era", "ming")])["status"] == "partial"
    assert judge(facts, [("tag", "floral"), ("tag", "green")])["status"] == "unknown"
    assert judge(facts, [("tag", "floral"), ("tag", "dragon")])["grade"] == 3
    m = retrieval_metrics([0.9, 0.8, 0.7], [3, 1, -1], [True, True, True])
    assert (m["tp"], m["fp"], m["unknown_selected"]) == (1, 1, 1)
    assert m["set_precision"] == 0.5
    assert m["precision_lower"] == 1 / 3 and m["precision_upper"] == 2 / 3


def test_coarse_fact_does_not_make_false_fine_negative():
    assert judge({"material": ["metal"]}, [("material", "bronze")])["status"] == "unknown"
    assert judge({"era": ["ming", "qing"]}, [("era", "ming")])["status"] == "unknown"
    assert (
        judge({"era": ["qing"], "qianlong_period_state": "unknown"}, [("era", "Qianlong")])[
            "status"
        ]
        == "unknown"
    )
    assert judge({"era": [], "qianlong_period_state": "yes"}, [("era", "Qianlong")])["grade"] == 3


def test_reviewed_core_frozen_complete_and_group_disjoint():
    base = Path(__file__).resolve().parents[1] / "eval/relevance_v5"
    frozen = json.loads((base / "freeze.json").read_text(encoding="utf-8"))
    for name, digest in frozen["hashes"].items():
        assert hashlib.sha256((base / name).read_bytes()).hexdigest() == digest

    def load(name):
        return [json.loads(s) for s in (base / name).read_text(encoding="utf-8").splitlines()]

    items, queries, qrels = map(load, ["items.jsonl", "queries.jsonl", "qrels.jsonl"])
    assert (len(items), len(queries), len(qrels)) == (200, 20, 4000)
    assert len({(r["item_id"], r["qid"]) for r in qrels}) == 4000
    groups = {
        split: {r["object_group"] for r in items if r["split"] == split}
        for split in ["dev", "test"]
    }
    assert not groups["dev"] & groups["test"]
    facts = {r["item_id"]: r for r in load("facts.jsonl")}
    conditions = {r["qid"]: r["conditions"] for r in queries}
    for row in qrels:
        assert judge(facts[row["item_id"]], conditions[row["qid"]])["grade"] == row["grade"]
