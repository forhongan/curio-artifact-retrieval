"""Compare saved methods on the exact same frozen eight-intent test rows."""

import hashlib
import json
from pathlib import Path
from run_joint_vlm_v9 import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "artifacts/joint_vlm_v9"
    pilot = json.loads((out / "pilot_report.json").read_text())
    rows = [json.loads(s) for s in (out / "pilot_scores.jsonl").read_text().splitlines()]
    test = [r for r in rows if r["partition"] == "test"]
    results = {
        "v7_zh": pilot["test"]["v7"]["zh"],
        "v7_en": pilot["test"]["v7"]["en"],
        "vqa_raw": pilot["test"]["vqa_raw_half"],
        "vqa_calibrated": pilot["test"]["vqa_calibrated"],
    }
    hashes = {}
    for lang in ["zh", "en"]:
        sets = {}
        for qid in pilot["protocol"]["qids"]:
            path = ROOT / f"artifacts/strict_boundary_v8/benchmark_{qid}_{lang}.json"
            sets[qid] = set(json.loads(path.read_text())["matched_item_ids"])
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        results["v8_strict_" + lang] = metrics(test, [r["item_id"] in sets[r["qid"]] for r in test])
    reference = out / "reference_report.json"
    if reference.exists():
        r = json.loads(reference.read_text())
        results.update(
            vqa_reference_raw=r["test_raw"], vqa_reference_calibrated=r["test_calibrated"]
        )
    report = {
        "scope": "Same 8 intents and 140 test objects; historical AI/source-reviewed labels, exploratory comparison, not independent human validation or full-gallery precision",
        "test_pairs": len(test),
        "methods": results,
        "strict_response_sha256": hashes,
    }
    (out / "matched_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"test_pairs": len(test), "methods": results}, indent=2))


if __name__ == "__main__":
    main()
