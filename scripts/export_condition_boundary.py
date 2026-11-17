"""Export compact runtime parameters from the completed v7 experiment."""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.retrieve.condition_boundary import FEATURE_NAMES  # noqa: E402
from curio.retrieve.condition_runtime import DEFAULT_BUNDLE, vocabulary_sha256  # noqa: E402


def main():
    folder = ROOT / "artifacts/set_boundary_v7"
    fitted = json.loads((folder / "models.json").read_text())
    feature_bytes = (folder / "feature_manifest.json").read_bytes()
    if hashlib.sha256(feature_bytes).hexdigest() != fitted["feature_manifest_sha256"]:
        raise ValueError("Feature manifest changed after fitting")
    feature = json.loads(feature_bytes)
    if feature["feature_names"] != FEATURE_NAMES:
        raise ValueError("Feature schema changed after fitting")
    index_dir = ROOT / "artifacts/siglip_adapted"
    for filename, key in [
        ("image_embeddings.npy", "image_embeddings_sha256"),
        ("item_ids.json", "image_ids_sha256"),
    ]:
        if hashlib.sha256((index_dir / filename).read_bytes()).hexdigest() != feature[key]:
            raise ValueError("Image index changed after fitting")
    manifest = json.loads((index_dir / "manifest.json").read_text())
    # Label-independent encoder outputs only: never ship qrels or image scores.
    prompts = feature["texts"]
    new_text = np.load(
        folder / f"condition_text_{feature['fingerprint'][:16]}.npz", allow_pickle=False
    )["vectors"]
    vectors = dict(zip(prompts, new_text))
    frozen_manifest = json.loads((ROOT / "artifacts/relevance_v5/text_manifest.json").read_text())
    frozen_text = np.load(ROOT / "artifacts/relevance_v5/text_vectors.npz", allow_pickle=False)[
        "siglip"
    ]
    vectors.update(zip(frozen_manifest["texts"], frozen_text))
    cache_path = DEFAULT_BUNDLE.with_name("condition_text_v7.npz")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(vectors)
    np.savez_compressed(
        cache_path, texts=np.array(names), vectors=np.stack([vectors[t] for t in names])
    )
    models = {}
    for name in ["conditions", "refit_base"]:
        for lang in ["en", "zh"]:
            key = f"{name}/{lang}"
            models[key] = {k: v for k, v in fitted["models"][key].items() if k != "folds"}
    bundle = {
        "schema": "condition-boundary-v7",
        "feature_names": FEATURE_NAMES,
        "vocabulary_sha256": vocabulary_sha256(),
        "models": models,
        "index_sha256": manifest["index_sha256"],
        "gallery_count": manifest["items"],
        "text_encoder": {
            "model_id": manifest["encoder_metadata"]["model_id"],
            "revision": feature["revision"],
        },
        "source_feature_manifest_sha256": fitted["feature_manifest_sha256"],
        "source_models_sha256": hashlib.sha256((folder / "models.json").read_bytes()).hexdigest(),
    }
    DEFAULT_BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    bundle["text_cache"] = {
        "file": cache_path.name,
        "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "contents": "frozen dev-query and condition text embeddings; no candidate labels or scores",
    }
    DEFAULT_BUNDLE.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(DEFAULT_BUNDLE)


if __name__ == "__main__":
    main()
