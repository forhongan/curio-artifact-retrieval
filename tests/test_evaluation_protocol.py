import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import numpy as np
import torch

from curio.evaluate import benchmark
from curio.evaluate import harness
from curio.evaluate.annotation import create_annotation_app, export_human
from curio.represent.train import multi_positive_loss, paired_retrieval
from curio.represent.validation import validate_object_splits


def test_silver_grade_preserves_core_match_as_grade_two_when_optional_fails():
    base = {"object_type": "vase", "material": "porcelain", "motifs": ["lotus"], "year_from": 1500, "year_to": 1550}
    query = {"object_type": "vase", "material": "porcelain", "motif": "dragon"}
    assert benchmark.grade(base, query) == 2
    assert benchmark.grade({**base, "motifs": ["dragon"]}, query) == 3


def test_silver_grade_partial_core_match_is_one_and_no_match_is_zero():
    query = {"object_type": "vase", "material": "porcelain", "motif": "dragon"}
    assert benchmark.grade({"object_type": "vase", "material": "bronze", "motifs": []}, query) == 1
    assert benchmark.grade({"object_type": "bowl", "material": "bronze", "motifs": []}, query) == 0


def test_silver_build_keeps_both_languages_of_a_predicate_in_one_split(tmp_path):
    items = []
    for index in range(6):
        items.append({"item_id": f"v{index}", "object_type": "vase", "material": "porcelain", "motifs": ["dragon" if index < 3 else "lotus"], "year_from": 1400 if index < 3 else 1700, "year_to": 1500 if index < 3 else 1800})
        items.append({"item_id": f"b{index}", "object_type": "bowl", "material": "bronze", "motifs": ["lotus" if index < 3 else "dragon"], "year_from": 1000 if index < 3 else 1400, "year_to": 1100 if index < 3 else 1500})
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    output = tmp_path / "benchmark"
    report = benchmark.build(data, output, n_dev=2, n_test=2)
    queries, qrel_signatures = {}, {}
    for split in ("dev", "test"):
        records = [json.loads(line) for line in (output / f"queries_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(record["text_en"] and record["text_zh"] for record in records)
        assert all(record["label_source"] == "metadata_silver" for record in records)
        queries[split] = {json.dumps(record["predicate"], sort_keys=True) for record in records}
        qrel_signatures[split] = {record["graded_qrels_sha256"] for record in records}
    assert queries["dev"].isdisjoint(queries["test"])
    assert qrel_signatures["dev"].isdisjoint(qrel_signatures["test"])
    assert len(qrel_signatures["dev"] | qrel_signatures["test"]) == 4
    assert report["human_annotations"] == 0
    assert "not independently judged visual relevance" in report["independence_warning"]


def test_benchmark_counts_graded_qrels_groups_instead_of_equivalent_predicates(tmp_path):
    items = []
    for index in range(3):
        items.append({"item_id": f"v{index}", "object_type": "vase", "material": "porcelain", "motifs": ["dragon"], "year_from": 1400, "year_to": 1500})
        items.append({"item_id": f"b{index}", "object_type": "bowl", "material": "bronze", "motifs": ["lotus"], "year_from": 1000, "year_to": 1100})
    pool = benchmark.candidate_pool(items)
    assert pool["raw_supported_predicates"] == 6 and pool["distinct_graded_qrels"] == 2
    assert len(pool["equivalent_predicate_groups"]) == 2
    assert all(set(row["predicate"]) == {"object_type", "material"} for row in pool["candidates"])
    reverse = benchmark.candidate_pool(list(reversed(items)))
    assert [row["graded_qrels_sha256"] for row in pool["candidates"]] == [row["graded_qrels_sha256"] for row in reverse["candidates"]]
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    output = tmp_path / "eval"
    output.mkdir()
    frozen = output / "queries_test.jsonl"
    frozen.write_text("existing frozen queries", encoding="utf-8")
    with pytest.raises(ValueError, match="Only 2 distinct informative graded-qrels groups"):
        benchmark.build(data, output, n_dev=2, n_test=2)
    assert frozen.read_text(encoding="utf-8") == "existing frozen queries"


def test_benchmark_rejects_queries_where_every_item_is_relevant():
    items = [{"item_id": f"v{index}", "object_type": "vase", "material": "porcelain",
              "motifs": ["dragon" if index < 3 else "lotus"]} for index in range(6)]
    pool = benchmark.candidate_pool(items)
    assert pool["raw_supported_predicates"] == 3
    assert pool["distinct_graded_qrels"] == 0
    assert len(pool["excluded_all_relevant_predicates"]) == 3


def test_benchmark_motif_support_counts_objects_not_repeated_tags():
    items = [{"item_id": f"v{index}", "object_type": "vase", "material": "porcelain",
              "motifs": ["dragon", "dragon", "dragon"] if index == 0 else []} for index in range(3)]
    items.append({"item_id": "b1", "object_type": "bowl", "material": "bronze"})
    pool = benchmark.candidate_pool(items)
    assert pool["raw_supported_predicates"] == 1
    assert pool["distinct_graded_qrels"] == 1
    assert all("motif" not in row["predicate"] for row in pool["candidates"])


def test_multi_positive_loss_matches_explicit_bidirectional_target_distribution():
    image = torch.tensor([[1.0, 0.2], [0.7, 0.5], [0.0, 1.0]], requires_grad=True)
    text = torch.tensor([[1.0, 0.0], [0.6, 0.4], [0.1, 0.9]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.float32)
    logits = torch.nn.functional.normalize(image, dim=-1) @ torch.nn.functional.normalize(text, dim=-1).T / 0.2
    forward_target = mask / mask.sum(dim=1, keepdim=True)
    backward_target = mask.T / mask.T.sum(dim=1, keepdim=True)
    expected = (-(forward_target * torch.log_softmax(logits, dim=1)).sum(1).mean() - (backward_target * torch.log_softmax(logits.T, dim=1)).sum(1).mean()) / 2
    loss = multi_positive_loss(image, text, mask, temperature=0.2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0


def test_duplicate_caption_retrieval_accepts_another_matching_image():
    image = np.eye(3, dtype=np.float32)
    text = image[[1, 0, 2]]
    result = paired_retrieval(image, text, ["vase", "vase", "bowl"])
    assert result["r@1"] == 1.0 and result["mrr"] == 1.0


@pytest.mark.parametrize("field", ["image_path", "sha256", "duplicate_group", "phash_group"])
def test_validation_rejects_cross_split_shared_images_or_groups(field):
    data = [
        {"item_id": "train1", "split": "train", field: "same"},
        {"item_id": "test1", "split": "test", field: "same"},
    ]
    with pytest.raises(ValueError, match="Cross-split leakage"):
        validate_object_splits(data)


@pytest.mark.parametrize("field", ["duplicate_sha256_of", "near_duplicate_of"])
def test_validation_rejects_cross_split_linked_duplicates(field):
    data = [{"item_id": "train1", "split": "train"}, {"item_id": "dev1", "split": "dev", field: "train1"}]
    with pytest.raises(ValueError, match="Cross-split near duplicate"):
        validate_object_splits(data)


def test_validation_accepts_same_split_views_but_rejects_repeated_ids():
    data = [{"item_id": "a", "split": "train", "sha256": "same"}, {"item_id": "b", "split": "train", "sha256": "same"}]
    validate_object_splits(data)
    with pytest.raises(ValueError, match="Duplicate item_id"):
        validate_object_splits([data[0], data[0]])


@pytest.mark.parametrize("architecture", ["clip", "siglip", "siglip2"])
def test_peft_vision_checkpoint_gradient_and_adapter_reload_on_tiny_cpu_model(tmp_path, architecture):
    """Exercise installed PEFT/Transformers contracts without downloads or GPU."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import CLIPConfig, CLIPModel, SiglipConfig, SiglipModel, Siglip2Config, Siglip2Model

    torch.set_num_threads(1)
    model_class, config_class = {"clip": (CLIPModel, CLIPConfig), "siglip": (SiglipModel, SiglipConfig),
                                 "siglip2": (Siglip2Model, Siglip2Config)}[architecture]
    text_config = {"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2, "num_hidden_layers": 1, "max_position_embeddings": 8}
    vision_config = {"hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 2, "num_hidden_layers": 1, "image_size": 16, "patch_size": 8}
    if architecture == "siglip2":
        vision_config["num_patches"] = 4
    config = config_class(text_config=text_config, vision_config=vision_config, projection_dim=16)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5523)
        base = model_class(config).cpu()
        state_before = {key: value.clone() for key, value in base.state_dict().items()}
        targets = [name for name, _ in base.named_modules() if "vision_model" in name and name.endswith(("q_proj", "v_proj"))]
        assert targets
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=targets, bias="none"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        model.base_model.model.text_model.eval()
        pixels = torch.randn(2, 3, 16, 16)
        image_inputs = {"pixel_values": pixels}
        if architecture == "siglip2":
            image_inputs = {"pixel_values": torch.randn(2, 4, 3 * 8 * 8),
                            "pixel_attention_mask": torch.ones(2, 4),
                            "spatial_shapes": torch.tensor([[2, 2], [2, 2]])}
        image_features = model.base_model.model.get_image_features(**image_inputs)
        loss = multi_positive_loss(image_features, torch.randn_like(image_features), torch.eye(2, dtype=torch.bool))
        loss.backward()
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        assert trainable and all("vision_model" in name and "lora_" in name for name, _ in trainable)
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for _, parameter in trainable)
        assert all(not parameter.requires_grad for parameter in model.base_model.model.text_model.parameters())
        model.eval()
        before_update = model.get_image_features(**image_inputs).detach().clone()
        torch.optim.AdamW([parameter for _, parameter in trainable], lr=.01).step()
        expected = model.get_image_features(**image_inputs).detach()
        assert not torch.allclose(expected, before_update)
        folder = tmp_path / architecture
        model.save_pretrained(folder)
        fresh = model_class(config).cpu()
        fresh.load_state_dict(state_before)
        restored = PeftModel.from_pretrained(fresh, folder).merge_and_unload().eval()
        actual = restored.get_image_features(**image_inputs).detach()
        torch.testing.assert_close(actual, expected)


def test_human_export_uses_latest_revision_and_keeps_conflicts_out_of_qrels(tmp_path):
    judgment_file = tmp_path / "judgments.jsonl"
    records = [
        {"task_id": "agreed", "qid": "q1", "item_id": "met_1", "annotator": "A", "grade": 2, "judgment_id": 1, "supersedes": None},
        {"task_id": "agreed", "qid": "q1", "item_id": "met_1", "annotator": "A", "grade": 3, "judgment_id": 2, "supersedes": 1},
        {"task_id": "agreed", "qid": "q1", "item_id": "met_1", "annotator": "B", "grade": 3, "judgment_id": 3, "supersedes": None},
        {"task_id": "conflict", "qid": "q1", "item_id": "met_2", "annotator": "A", "grade": 0, "judgment_id": 4, "supersedes": None},
        {"task_id": "conflict", "qid": "q1", "item_id": "met_2", "annotator": "B", "grade": 2, "judgment_id": 5, "supersedes": None},
    ]
    judgment_file.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    output = tmp_path / "export"
    report = export_human(judgment_file, output)
    qrels = [json.loads(line) for line in (output / "qrels_human.jsonl").read_text(encoding="utf-8").splitlines()]
    assert qrels == [{"qid": "q1", "grades": {"met_1": 3}, "label_source": "human_unanimous_or_single"}]
    assert report["unresolved_conflicts"] == 1
    # Use ordinal distances 0,1,2,3, not ranks of the observed subset 0,2,3.
    # Observed squared distance=2; chance distance=3.5; kappa=1-2/3.5.
    assert report["weighted_kappa"] == pytest.approx(3 / 7)
    assert json.loads((output / "conflicts.json").read_text(encoding="utf-8"))[0]["task_id"] == "conflict"


def test_export_reports_fixed_pair_kappas_and_degenerate_as_null(tmp_path):
    path = tmp_path / "judgments.jsonl"
    rows = []
    values = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 2), (0, 0)]
    for index, (a, b) in enumerate(values):
        for annotator, grade in (("A", a), ("B", b)):
            rows.append({"task_id": f"t{index}", "qid": "q", "item_id": f"i{index}", "annotator": annotator, "grade": grade})
    # A third rater overlaps only two tasks; its agreement is not pooled into A/B.
    for index, grade in ((0, 0), (1, 1)):
        rows.append({"task_id": f"t{index}", "qid": "q", "item_id": f"i{index}", "annotator": "C", "grade": grade})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = export_human(path, tmp_path / "out")
    kappas = {(row["annotator_a"], row["annotator_b"]): row for row in report["pairwise_kappa"]}
    assert set(kappas) == {("A", "B"), ("A", "C"), ("B", "C")}
    assert kappas[("A", "B")]["shared_tasks"] == 6
    assert kappas[("A", "C")]["shared_tasks"] == 2
    assert report["weighted_kappa"] is None
    assert report["kappa_reason"] == "multiple_annotator_pairs_reported_separately"


def test_degenerate_kappa_is_json_null_with_reason(tmp_path):
    path = tmp_path / "judgments.jsonl"
    rows = [
        {"task_id": "t1", "qid": "q", "item_id": "i1", "annotator": "A", "grade": 3},
        {"task_id": "t1", "qid": "q", "item_id": "i1", "annotator": "B", "grade": 3},
        {"task_id": "t2", "qid": "q", "item_id": "i2", "annotator": "A", "grade": 3},
        {"task_id": "t2", "qid": "q", "item_id": "i2", "annotator": "B", "grade": 3},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = export_human(path, tmp_path / "out")
    assert report["weighted_kappa"] is None
    assert report["pairwise_kappa"][0]["reason"] == "zero_expected_disagreement"
    # The artifact remains standards-compliant JSON (never NaN).
    serialized = (tmp_path / "out" / "agreement.json").read_text(encoding="utf-8")
    assert "NaN" not in serialized
    json.loads(serialized)


def test_annotation_revisions_append_rows_and_link_supersedes(tmp_path):
    item_dir = tmp_path / "data" / "processed" / "images"
    item_dir.mkdir(parents=True)
    (item_dir / "met_1.jpg").write_bytes(b"fixture")
    (tmp_path / "items.jsonl").write_text(json.dumps({"item_id": "met_1", "image_path": "data/processed/images/met_1.jpg"}) + "\n", encoding="utf-8")
    task = {"task_id": "task1", "qid": "q1", "item_id": "met_1", "text": "瓷瓶", "text_en": "porcelain vase"}
    (tmp_path / "pool.jsonl").write_text(json.dumps(task) + "\n", encoding="utf-8")
    target = tmp_path / "judgments.jsonl"
    client = TestClient(create_annotation_app(data="items.jsonl", pool="pool.jsonl", judgments="judgments.jsonl", root=tmp_path))
    endpoint = "/api/judgment"
    first = client.post(endpoint, json={"task_id": "task1", "annotator": "A", "grade": 1})
    second = client.post(endpoint, json={"task_id": "task1", "annotator": "A", "grade": 2})
    assert first.status_code == second.status_code == 201
    saved = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(saved) == 2
    assert saved[0]["supersedes"] is None
    assert saved[1]["supersedes"] == saved[0]["judgment_id"]


def test_evaluation_provenance_comes_from_qrels_not_query_metadata(tmp_path, monkeypatch):
    data = tmp_path / "items.jsonl"
    data.write_text(json.dumps({"item_id": "met_1"}) + "\n", encoding="utf-8")
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(json.dumps({"qid": "q1", "text_en": "vase", "label_source": "metadata_silver"}) + "\n", encoding="utf-8")
    qrels_file = tmp_path / "qrels.jsonl"
    qrels_file.write_text(json.dumps({"qid": "q1", "grades": {"met_1": 3}, "label_source": "human_double_blind"}) + "\n", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    (index / "manifest.json").write_text(json.dumps({"data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()}), encoding="utf-8")

    class FakeEncoder:
        metadata = {"encoder": "fake"}
        def __init__(self, *args, **kwargs): pass
        def encode_texts(self, texts): return [[1.0]]

    class FakeService:
        def __init__(self, *args, **kwargs): pass
        def health(self): return {"probes_available": False}
        def search(self, query, **kwargs):
            return {"results": [{"item_id": "met_1"}], "timings": {"total_ms": 1.0}, "degraded": False}

    monkeypatch.setattr(harness, "Encoder", FakeEncoder)
    monkeypatch.setattr(harness, "SearchService", FakeService)
    args = SimpleNamespace(data=str(data), queries=str(query_file), qrels=str(qrels_file), index=str(index), encoder="fake", device="cpu", variants="lexical", languages="en", output=str(tmp_path / "result"))
    report = harness.evaluate(args)
    assert report["label_source"] == "human_double_blind"
    assert "Silver metadata compliance" not in report["warning"]


def test_evaluation_rejects_entirely_unjudged_query_before_loading_model(tmp_path, monkeypatch):
    data = tmp_path / "items.jsonl"
    data.write_text(json.dumps({"item_id": "met_1"}) + "\n", encoding="utf-8")
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(json.dumps({"qid": "missing_query", "text_en": "vase"}) + "\n", encoding="utf-8")
    qrels_file = tmp_path / "qrels.jsonl"
    qrels_file.write_text("", encoding="utf-8")
    def never_load(*args, **kwargs):
        raise AssertionError("Model must not load with incomplete qrels")
    monkeypatch.setattr(harness, "Encoder", never_load)
    args = SimpleNamespace(data=str(data), queries=str(query_file), qrels=str(qrels_file), index=str(tmp_path / "nonexistent-index"), encoder="fake", device="cpu", variants="baseline", languages="en", output=str(tmp_path / "result"))
    with pytest.raises(ValueError, match=r"Missing qrels for 1 queries; sample: \['missing_query'\]"):
        harness.evaluate(args)


def evaluation_fixture(tmp_path, monkeypatch, *, probes=False):
    items = [{"item_id": key, "title": "Porcelain vase", "object_type": "vase", "material": "porcelain"}
             for key in ("met_1", "met_2")]
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    query_file, qrels_file = tmp_path / "queries.jsonl", tmp_path / "qrels.jsonl"
    query_file.write_text(json.dumps({"qid": "q1", "text_en": "porcelain vase"}) + "\n", encoding="utf-8")
    qrels_file.write_text(json.dumps({"qid": "q1", "grades": {"met_1": 3, "met_2": 3}, "label_source": "metadata_silver"}) + "\n", encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    vectors, ids = np.eye(2, dtype=np.float32), [item["item_id"] for item in items]
    index_hash = hashlib.sha256(vectors.tobytes() + json.dumps(ids).encode()).hexdigest()
    manifest = {"encoder": "clip", "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "index_sha256": index_hash, "dimension": 2, "items": 2, "index_version": "fixture-v1",
                "catalog_sha256": hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    np.save(index / "image_embeddings.npy", vectors)
    (index / "item_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    manifest_path = index / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    probe_dir = tmp_path / "probes"
    if probes:
        probe_dir.mkdir()
        report = {"index_sha256": index_hash, "data_sha256": manifest["data_sha256"],
                  "feature_dimension": 2, "index_version": "fixture-v1",
                  "index_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  "tasks": {"material": {"status": "trained", "classes": ["bronze", "porcelain"]}}}
        (probe_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
        np.savez(probe_dir / "material.npz", weights=np.array([[1, -1], [-1, 1]], dtype=np.float32),
                 bias=np.zeros(2), classes=np.array(["bronze", "porcelain"]))

    class FakeEncoder:
        metadata = {"encoder": "clip", "device": "cpu", "test_fixture": True}
        def __init__(self, *args, **kwargs): pass
        def encode_texts(self, texts): return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(harness, "Encoder", FakeEncoder)
    args = SimpleNamespace(data=str(data), queries=str(query_file), qrels=str(qrels_file), index=str(index),
                           encoder="clip", device="cpu", variants="hybrid,visual", languages="en",
                           output=str(tmp_path / "evaluation"), probes=str(probe_dir) if probes else None,
                           probe_weight=.2)
    return args


@pytest.mark.parametrize("missing_path", [False, True])
def test_harness_visual_without_probes_reports_exact_hybrid_fallback(tmp_path, monkeypatch, missing_path):
    args = evaluation_fixture(tmp_path, monkeypatch)
    if missing_path:
        args.probes = str(tmp_path / "absent-probes")
    report = harness.evaluate(args)
    assert report["probes"]["configured"] is missing_path
    assert not report["probes"]["available"]
    assert report["probes"]["files_sha256"] == {}
    visual = next(row for row in report["summary"] if row["variant"] == "visual")
    assert visual["degraded_queries"] == 1
    assert visual["effective_variants"] == {"hybrid": 1}
    assert visual["probe_applied_queries"] == 0
    rankings = [json.loads(line) for line in (tmp_path / "evaluation" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rankings[0]["results"] == rankings[1]["results"]
    assert rankings[1]["effective_variant"] == "hybrid" and rankings[1]["warnings"]


def test_harness_visual_persists_probe_artifact_provenance_and_actual_use(tmp_path, monkeypatch):
    args = evaluation_fixture(tmp_path, monkeypatch, probes=True)
    report = harness.evaluate(args)
    assert report["probes"]["available"]
    assert report["probes"]["tasks"] == ["material"]
    assert report["probes"]["weight"] == .2
    for name in ("report.json", "material.npz"):
        assert report["probes"]["files_sha256"][name] == hashlib.sha256((tmp_path / "probes" / name).read_bytes()).hexdigest()
    assert report["data_sha256"] == report["index"]["data_sha256"]
    assert report["index_manifest_sha256"] == hashlib.sha256((tmp_path / "index" / "manifest.json").read_bytes()).hexdigest()
    visual = next(row for row in report["summary"] if row["variant"] == "visual")
    assert visual["degraded_queries"] == 0
    assert visual["effective_variants"] == {"visual": 1}
    assert visual["probe_applied_queries"] == 1
    rows = [json.loads(line) for line in (tmp_path / "evaluation" / "per_query.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[1]["probe_applied"] and rows[1]["effective_variant"] == "visual"
    assert {"encode_ms", "retrieve_ms", "fusion_ms"} <= rows[1]["timings"].keys()


def test_harness_cli_accepts_visual_probe_arguments():
    args = harness.build_parser().parse_args(["--variants", "hybrid,visual", "--probes", "fixture-probes", "--probe-weight", ".3", "--gallery-ids", "gallery.json"])
    assert args.variants == "hybrid,visual" and args.probes == "fixture-probes" and args.probe_weight == .3
    assert args.gallery_ids == "gallery.json"


@pytest.mark.parametrize("task,label", [("colors", "blue"), ("shape", "narrow_mouth"),
                                       ("motifs", "dragon")])
def test_harness_rejects_detail_weights_changed_during_evaluation(tmp_path, monkeypatch, task, label):
    args = evaluation_fixture(tmp_path, monkeypatch, probes=True)
    probes = tmp_path / "probes"
    report_path = probes / "report.json"
    probe_report = json.loads(report_path.read_text())
    probe_report["tasks"][task] = {"status": "trained", "activation": "sigmoid", "classes": [label]}
    report_path.write_text(json.dumps(probe_report))
    binary = probes / f"{task}.npz"
    np.savez(binary, weights=np.ones((1, 2)), bias=np.zeros(1), classes=np.array([label]))
    real_search = harness.SearchService.search

    def search_and_change_weights(service, *search_args, **kwargs):
        result = real_search(service, *search_args, **kwargs)
        binary.write_bytes(b"changed detail weights")
        return result

    monkeypatch.setattr(harness.SearchService, "search", search_and_change_weights)
    with pytest.raises(ValueError, match="Probe files changed during evaluation"):
        harness.evaluate(args)
    assert not (Path(args.output) / "metrics.json").exists()


@pytest.mark.parametrize("probes,weight,effective,applied", [
    (False, .15, "dense_rewritten", 0), (True, .15, "image_only", 1), (True, 0, "image_only", 0),
])
def test_harness_image_only_records_actual_probe_use_and_dense_fallback(tmp_path, monkeypatch, probes, weight, effective, applied):
    args = evaluation_fixture(tmp_path, monkeypatch, probes=probes)
    args.variants, args.probe_weight = "image_only", weight
    report = harness.evaluate(args)
    summary = report["summary"][0]
    assert summary["effective_variants"] == {effective: 1}
    assert summary["probe_applied_queries"] == applied
    rows = [json.loads(line) for line in (tmp_path / "evaluation" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["effective_variant"] == effective and rows[0]["probe_coverage"]
    assert all(row["score_breakdown"]["bm25"] is None and row["matched_slots"] == [] for row in rows[0]["results"])
    assert all(entry["kind"] in {"similarity", "visual_probe"} for row in rows[0]["results"] for entry in row["evidence"])


def test_harness_cli_includes_image_only_in_default_variants():
    args = harness.build_parser().parse_args([])
    assert "image_only" in args.variants.split(",")


@pytest.mark.parametrize("field,value,match", [("probe_weight", 1.1, "probe_weight"),
                                                ("variants", "visual,visual", "distinct retrieval variants"),
                                                ("languages", "fr", "distinct languages")])
def test_harness_rejects_invalid_configuration_before_loading_encoder(tmp_path, monkeypatch, field, value, match):
    args = evaluation_fixture(tmp_path, monkeypatch)
    setattr(args, field, value)
    monkeypatch.setattr(harness, "Encoder", lambda *a, **k: pytest.fail("Must validate before model load"))
    with pytest.raises(ValueError, match=match):
        harness.evaluate(args)


def test_harness_explicit_gallery_keeps_original_index_and_probe_provenance(tmp_path, monkeypatch):
    args = evaluation_fixture(tmp_path, monkeypatch, probes=True)
    gallery = tmp_path / "gallery.json"
    gallery.write_text(json.dumps(["met_2"]), encoding="utf-8")
    args.gallery_ids = str(gallery)
    qrel = {"qid": "q1", "grades": {"met_2": 0}, "label_source": "ai_visual_audit"}
    (tmp_path / "qrels.jsonl").write_text(json.dumps(qrel) + "\n", encoding="utf-8")
    report = harness.evaluate(args)
    assert report["gallery_ids"] == ["met_2"] and report["gallery_count"] == 1
    assert report["gallery_restricted"] and report["qrels_scope"] == "exhaustive_explicit_gallery"
    assert report["gallery_sha256"] == hashlib.sha256(b'["met_2"]').hexdigest()
    assert report["index"]["items"] == report["service"]["indexed_images"] == 2
    assert report["probes"]["available"] and report["label_source"] == "ai_visual_audit"
    assert "AI-generated image judgments, not independent human ground truth" in report["warning"]
    rankings = [json.loads(line) for line in (tmp_path / "evaluation" / "rankings.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["ranking"] == ["met_2"] for row in rankings)
    assert all(row["metrics"]["ndcg@10"]["mean"] == 0 for row in report["summary"])
    assert "Every item in the explicitly restricted gallery is judged" in report["warning"]


@pytest.mark.parametrize("gallery_ids,grades", [(["met_1", "met_2"], {"met_1": 3}),
                                                (["met_1"], {"met_1": 3, "met_2": 0})])
def test_harness_restricted_gallery_requires_exhaustive_exact_qrels_before_model_load(tmp_path, monkeypatch, gallery_ids, grades):
    args = evaluation_fixture(tmp_path, monkeypatch)
    gallery = tmp_path / "gallery.json"
    gallery.write_text(json.dumps(gallery_ids), encoding="utf-8")
    args.gallery_ids = str(gallery)
    (tmp_path / "qrels.jsonl").write_text(json.dumps({"qid": "q1", "grades": grades}) + "\n", encoding="utf-8")
    monkeypatch.setattr(harness, "Encoder", lambda *a, **k: pytest.fail("Must validate before model load"))
    with pytest.raises(ValueError, match="Explicit gallery requires exhaustive qrels"):
        harness.evaluate(args)


def test_harness_rejects_qrels_outside_catalog_before_model_load(tmp_path, monkeypatch):
    args = evaluation_fixture(tmp_path, monkeypatch)
    (tmp_path / "qrels.jsonl").write_text(json.dumps({"qid": "q1", "grades": {"met_1": 3, "unknown": 0}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(harness, "Encoder", lambda *a, **k: pytest.fail("Must validate before model load"))
    with pytest.raises(ValueError, match="Qrels contain IDs outside the evaluation catalog"):
        harness.evaluate(args)


@pytest.mark.parametrize("ids,match", [([], "nonempty JSON list"), (["met_1", "met_1"], "Duplicate item ID"),
                                       (["unknown"], "Unknown gallery IDs"), ({"met_1": True}, "nonempty JSON list")])
def test_harness_rejects_invalid_explicit_gallery(tmp_path, monkeypatch, ids, match):
    args = evaluation_fixture(tmp_path, monkeypatch)
    gallery = tmp_path / "gallery.json"
    gallery.write_text(json.dumps(ids), encoding="utf-8")
    args.gallery_ids = str(gallery)
    with pytest.raises(ValueError, match=match):
        harness.evaluate(args)


def test_predicate_audit_reports_selectivity_and_identical_label_groups(tmp_path):
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "audit_retrieval_cpu.py"))
    items = [{"item_id": f"item_{index}", "object_type": "vase", "material": "porcelain",
              "motifs": ["dragon"], "year_from": 1400, "year_to": 1500} for index in range(3)]
    items.extend({"item_id": f"bowl_{index}", "object_type": "bowl", "material": "bronze",
                  "motifs": ["lotus"], "year_from": 1000, "year_to": 1100} for index in range(3))
    data = tmp_path / "items.jsonl"
    data.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    report = audit["predicate_audit"](data)
    assert report["real_items"] == 6 and report["existing_supported_predicates"] == 6
    assert not report["builder"]["possible"] and "cannot fabricate 120" in report["builder"]["error"]
    assert report["distinct_graded_qrels"] == 2 and report["distinct_grade3_item_sets"] == 2
    assert len(report["identical_graded_qrels_groups"]) == 2
    for record in report["predicates"]:
        assert record["grade_distribution"] == {"0": 3, "1": 0, "2": 0, "3": 3}
        assert record["exact_grade3_fraction"] == record["relevant_grade2_or_3_fraction"] == .5
    assert len(report["supported_motif_plus_era_extensions"]) == 2
    assert report["queries_with_all_items_relevant"] == []


def test_cpu_scaling_diagnostic_is_explicitly_synthetic_and_caches_probe_predictions(tmp_path, monkeypatch):
    args = evaluation_fixture(tmp_path, monkeypatch)
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "audit_retrieval_cpu.py"))
    report = audit["cpu_scaling"](args.data, args.index, size=12, repetitions=2)
    assert report["synthetic"] and report["gallery_count"] == 12
    assert report["source_real_index_rows"] == report["unique_source_rows"] == 2
    assert report["probe_predict_calls"] == 1
    assert "not relevance or real end-to-end latency" in report["purpose"]
    assert "no text model" in report["query_vectors"]
    assert {row["variant"] for row in report["results"]} == {"baseline", "hybrid", "visual"}
    assert all(row["stages"]["retrieve_ms"]["p50_ms"] >= 0 for row in report["results"])
