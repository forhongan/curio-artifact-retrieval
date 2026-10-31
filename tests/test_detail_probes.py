import hashlib
import json

import numpy as np
import pytest

from curio.represent.detail_probes import load_annotations, train
from curio.represent.probes import LinearProbes, evaluate_test, train_probes


def fixture(tmp_path):
    rng = np.random.default_rng(5523)
    items, vectors, annotations = [], [], []
    for split, count in (("train", 12), ("dev", 4), ("test", 4)):
        for label in (0, 1):
            for j in range(count):
                key = f"{split}_{label}_{j}"
                digest = hashlib.sha256(key.encode()).hexdigest()
                items.append(
                    {
                        "item_id": key,
                        "split": split,
                        "image_path": f"{key}.jpg",
                        "sha256": digest,
                        "image_sha256": digest,
                        "object_type": "bowl" if label else "vase",
                        "material": "porcelain" if label else "bronze",
                    }
                )
                vectors.append(np.array([1, label * 3, 1 - label, 0.0]) + rng.normal(0, 0.02, 4))
                annotations.append(
                    {
                        "item_id": key,
                        "split": split,
                        "image_sha256": digest,
                        "label_source": "ai_image_only",
                        "caption_en": "a synthetic test object",
                        "colors": {"blue": label, "green": None},
                        "shape": {"narrow_mouth": 1 - label},
                        "motifs": {},
                    }
                )
    data = tmp_path / "items.jsonl"
    data.write_text("\n".join(json.dumps(x) for x in items), encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(x) for x in annotations), encoding="utf-8")
    index = tmp_path / "index"
    index.mkdir()
    vectors = np.array(vectors, dtype=np.float32)
    ids = [x["item_id"] for x in items]
    np.save(index / "image_embeddings.npy", vectors)
    (index / "item_ids.json").write_text(json.dumps(ids))
    (index / "manifest.json").write_text(
        json.dumps(
            {
                "items": len(items),
                "dimension": 4,
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "index_sha256": hashlib.sha256(
                    vectors.tobytes() + json.dumps(ids).encode()
                ).hexdigest(),
                "index_version": "unit-fixture",
            }
        )
    )
    return data, labels, index, items


def test_detail_sigmoid_heads_preserve_base_probes_and_unknowns(tmp_path):
    data, labels, index, _ = fixture(tmp_path)
    base, output = tmp_path / "base", tmp_path / "detail"
    train_probes(data, index, base)
    report = train(data, index, labels, base, output)
    assert report["tasks"]["colors"]["classes"] == ["blue"]
    assert report["tasks"]["colors"]["test"]["blue"]["f1"] == 1
    assert report["tasks"]["shape"]["classes"] == ["narrow_mouth"]
    assert report["tasks"]["motifs"]["status"] == "skipped"
    predictions = LinearProbes(output).predict(np.load(index / "image_embeddings.npy"))
    assert set(predictions) == {"material", "object_type", "colors", "shape"}
    assert not np.allclose(predictions["colors"]["probabilities"], 1)
    assert np.allclose(predictions["material"]["probabilities"].sum(1), 1)


def test_annotation_split_and_image_fingerprint_are_enforced(tmp_path):
    _, labels, _, items = fixture(tmp_path)
    annotations = [json.loads(line) for line in labels.read_text().splitlines()]
    annotations[0]["split"] = "test"
    labels.write_text("\n".join(json.dumps(row) for row in annotations))
    with pytest.raises(ValueError, match="split mismatch"):
        load_annotations(labels, items)
    annotations[0]["split"] = "train"
    annotations[0]["image_sha256"] = "changed"
    labels.write_text("\n".join(json.dumps(row) for row in annotations))
    with pytest.raises(ValueError, match="image hash"):
        load_annotations(labels, items)


def test_deferred_detail_training_and_frozen_test_do_not_refit_or_change_report(tmp_path, monkeypatch):
    from sklearn.linear_model import LogisticRegression

    data, labels, index, _ = fixture(tmp_path)
    base, output = tmp_path / "base", tmp_path / "detail"
    base_report = train_probes(data, index, base, defer_test=True)
    assert all(row.get("test") is None for row in base_report["tasks"].values())
    monkeypatch.setattr(LogisticRegression, "predict_proba", lambda *a, **k: pytest.fail("Test scoring must be deferred"))
    report = train(data, index, labels, base, output, defer_test=True)
    assert report["test_deferred"]
    assert all(row.get("test") is None for row in report["tasks"].values())
    before = (output / "report.json").read_bytes()
    monkeypatch.setattr(LogisticRegression, "fit", lambda *a, **k: pytest.fail("Frozen evaluation must not refit"))
    test_report = evaluate_test(data, index, output, labels)
    assert test_report["tasks"]["colors"]["test"]["blue"]["f1"] == 1
    assert test_report["tasks"]["material"]["test"]["support"] == 8
    assert test_report["training_report_sha256"] == hashlib.sha256(before).hexdigest()
    assert set(test_report["artifacts_sha256"]) == {"material.npz", "object_type.npz", "colors.npz", "shape.npz"}
    assert (output / "report.json").read_bytes() == before
    assert json.loads((output / "test_report.json").read_text()) == test_report


def test_frozen_test_rejects_changed_annotations_and_weights(tmp_path):
    data, labels, index, _ = fixture(tmp_path)
    base, output = tmp_path / "base", tmp_path / "detail"
    train_probes(data, index, base, defer_test=True)
    train(data, index, labels, base, output, defer_test=True)
    original = labels.read_bytes()
    labels.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="annotation hash mismatch"):
        evaluate_test(data, index, output, labels)
    labels.write_bytes(original)
    (output / "colors.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        evaluate_test(data, index, output, labels)
