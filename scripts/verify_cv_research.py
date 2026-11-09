"""Verify scientific provenance and reproduce saved pixel-only predictions."""

from __future__ import annotations

import json

import numpy as np
from threadpoolctl import threadpool_limits

from curio.retrieve.attribute_cv import parse_attributes, predict_heads
from run_cv_research import ROOT, BASE, dump, load_features, rows, sha


def main():
    out = ROOT / "artifacts/cv_research_20261104/fresh"
    selection = json.loads((out / "selection.json").read_text())
    protocol = json.loads((BASE / "protocol.json").read_text())
    assert sha(ROOT / "data/snapshots/collection_v2.jsonl") == protocol["catalog_sha256"]
    assert sha(ROOT / "eval/visual_labels/annotations.jsonl") == protocol["old_annotations_sha256"]
    assert sha(BASE / "queries.json") == protocol["queries_sha256"]
    assert sha(BASE / "blind_map.json") == protocol["mapping_sha256"]
    for row in json.loads((BASE / "blind_map.json").read_text()):
        assert sha(ROOT / row["image_path"]) == row["image_sha256"]
    cdir = ROOT / "artifacts/classical_cv"
    manifest = json.loads((cdir / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        assert sha(cdir / name) == expected
    expected = {r["item_id"]: r for r in rows(BASE / "pixels.jsonl")}
    actual = {r["item_id"]: r for r in rows(cdir / "pixels.jsonl")}
    assert expected == actual
    ids, features, _ = load_features()
    report = {
        "status": "passed",
        "catalog_count": len(ids),
        "head_count": len(selection["classes"]),
        "models": {},
        "parser_cases": 0,
        "selection_sha256": sha(out / "selection.json"),
        "label_files_sha256": {
            person: sha(BASE / f"annotator_{person}/labels.jsonl") for person in ("a", "b", "qa")
        },
    }
    for query in json.loads((BASE / "queries.json").read_text(encoding="utf8")):
        for lang in ("zh", "en"):
            assert set(parse_attributes(query[lang])) == set(map(tuple, query["atoms"]))
            report["parser_cases"] += 1
    with threadpool_limits(limits=1):
        for name, x in features.items():
            saved = out / f"{name}_heads.npz"
            assert sha(saved) == selection["models"][name]["artifact_sha256"]
            head = np.load(saved)
            assert head["classes"].tolist() == selection["classes"]
            recomputed = predict_heads(x, head["coefficients"], head["intercepts"])
            original = np.load(out / f"{name}_predictions.npy")
            difference = float(np.max(np.abs(recomputed - original)))
            np.testing.assert_allclose(recomputed, original, atol=3e-6, rtol=1e-5)
            report["models"][name] = {
                "maximum_prediction_difference": difference,
                "rows": len(x),
                "features": x.shape[1],
            }
    dump(out / "acceptance.json", report)
    files = [p for p in out.glob("*.json") if p.name != "artifact_manifest.json"]
    files += list(out.glob("*_heads.npz")) + list(out.glob("*_predictions.npy"))
    code = [
        ROOT / "src/curio/represent/classical_cv.py",
        ROOT / "src/curio/retrieve/attribute_cv.py",
    ]
    code += list((ROOT / "scripts").glob("*cv_research.py"))
    dump(
        out / "artifact_manifest.json",
        {
            "files": {str(p.relative_to(ROOT)): sha(p) for p in files + code},
            "note": "Current finalized artifact/code fingerprints. run_manifest retains code hash at original main experiment execution; auxiliary baseline/analysis added later without changing frozen heads or selection.",
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "models": len(report["models"]),
                "gallery": len(ids),
                "parser_cases": report["parser_cases"],
            }
        )
    )


if __name__ == "__main__":
    main()
