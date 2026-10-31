import hashlib
import json
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from curio.represent.probes import LinearProbes, train_probes


def fixture(tmp_path):
    data = tmp_path / "items.jsonl"
    index = tmp_path / "index"
    index.mkdir()
    rng = np.random.default_rng(4)
    items = []
    vectors = []
    ids = []
    centers = {
        "plate": np.array([3.0, 0.0, 0.0, 0.0]),
        "bowl": np.array([0.0, 3.0, 0.0, 0.0]),
        "vase": np.array([0.0, 0.0, 3.0, 0.0]),
    }
    for split, n in [("train", 12), ("dev", 4), ("test", 4)]:
        for label, center in centers.items():
            for j in range(n):
                item_id = f"{split}_{label}_{j}"
                ids.append(item_id)
                vectors.append(center + rng.normal(0, 0.15, 4))
                items.append(
                    {
                        "item_id": item_id,
                        "split": split,
                        "object_type": label,
                        "material": "porcelain" if label != "vase" else "bronze",
                        "image_path": f"{item_id}.jpg",
                        "sha256": item_id,
                        "duplicate_group": item_id,
                        "phash_group": item_id,
                    }
                )
    data.write_text("\n".join(json.dumps(x) for x in items), encoding="utf-8")
    arr = np.asarray(vectors, dtype=np.float32)
    np.save(index / "image_embeddings.npy", arr)
    (index / "item_ids.json").write_text(json.dumps(ids))
    index_sha = hashlib.sha256(arr.tobytes() + json.dumps(ids).encode()).hexdigest()
    (index / "manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                "index_sha256": index_sha,
                "items": len(ids),
                "dimension": 4,
                "index_version": "fixture",
                "model_version": "fixture",
            }
        )
    )
    return data, index


def test_train_and_predict(tmp_path):
    data, index = fixture(tmp_path)
    out = tmp_path / "probes"
    report = train_probes(data, index, out)
    assert report["tasks"]["object_type"]["status"] == "trained"
    assert report["tasks"]["material"]["status"] == "trained"
    probe = LinearProbes(out)
    pred = probe.predict(np.eye(4, dtype=np.float32))
    assert pred["object_type"]["probabilities"].shape[1] == 3
    assert np.allclose(pred["object_type"]["probabilities"].sum(1), 1)
    report_material = report["tasks"]["material"]
    raw_items = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    train_idx = [i for i, item in enumerate(raw_items) if item["split"] == "train"]
    expected = LogisticRegression(
        C=report_material["selected_C"], class_weight="balanced", max_iter=1000, random_state=0
    ).fit(vectors[train_idx], [raw_items[i]["material"] for i in train_idx])
    np.testing.assert_allclose(
        probe.predict(vectors)["material"]["probabilities"],
        expected.predict_proba(vectors),
        atol=1e-6,
    )
    assert report_material["test_eligible_support"] == report_material["test"]["support"]


def test_hash_mismatch_is_rejected(tmp_path):
    data, index = fixture(tmp_path)
    manifest = json.loads((index / "manifest.json").read_text())
    manifest["index_sha256"] = "bad"
    (index / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash"):
        train_probes(data, index, tmp_path / "out")


def test_probe_artifact_replacement_is_rejected(tmp_path):
    data, index = fixture(tmp_path)
    out = tmp_path / "probes"
    train_probes(data, index, out)
    with np.load(out / "material.npz", allow_pickle=False) as saved:
        np.savez_compressed(
            out / "material.npz",
            weights=saved["weights"] + 1,
            bias=saved["bias"],
            classes=saved["classes"],
        )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        LinearProbes(out)


def test_cross_split_duplicate_is_rejected(tmp_path):
    data, index = fixture(tmp_path)
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()]
    rows[-1]["sha256"] = rows[0]["sha256"]
    data.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Cross-split leakage"):
        train_probes(data, index, tmp_path / "out")
