"""Explicit yes/no retrieval boundary on frozen visual or catalogue tasks.

The boundary is selected once on dev and frozen before the 96-image visual
test is scored. This script deliberately preserves score vectors so a later
model or fusion rule can be added without treating the test set as validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from curio.represent.encoders import Encoder
from curio.evaluate.set_boundary import calibrate_scores

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def trial_templates(text):
    text = text.strip().lower()
    return {
        "raw": text,
        "photo": f"a photo of {text}",
        "photograph": f"a photograph of {text}",
        "museum_photograph": f"a museum photograph of {text}",
        "picture": f"a picture of {text}",
        "image": f"an image of {text}",
    }


def metrics(yes, relevant, ids):
    selected = {item_id for item_id, flag in zip(ids, yes) if flag}
    true = set(relevant)
    tp = len(selected & true)
    fp = len(selected - true)
    fn = len(true - selected)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    if true:
        match = int(selected == true)
    else:
        match = int(not selected)
    result = {
        "relevant": len(true), "selected": len(selected), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "exact_set_match": match,
    }
    result["no_match"] = not true
    result["no_match_correct"] = int(not true and not selected)
    return result


def query_data(split, encoder, template):
    base = ROOT / f"eval/visual_{split}"
    ids = json.loads((base / "gallery_ids.json").read_text(encoding="utf-8"))
    queries = read_jsonl(base / "queries.jsonl")
    qrels = {row["qid"]: row["grades"] for row in read_jsonl(base / "qrels.jsonl")}
    index_ids = json.loads((ROOT / "artifacts/siglip/item_ids.json").read_text(encoding="utf-8"))
    source = {item_id: i for i, item_id in enumerate(index_ids)}
    gallery = np.load(ROOT / "artifacts/siglip/image_embeddings.npy", allow_pickle=False)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery = gallery[[source[item_id] for item_id in ids]]
    texts = [trial_templates(q["text_en"])[template] for q in queries]
    text_vectors = encoder.encode_texts(texts)
    scores = text_vectors @ gallery.T
    return queries, ids, qrels, scores


def load_unseen_queries(encoder, template):
    """Load separately authored out-of-domain no-match controls.

    These queries do not reuse visual-audit queries. Their qrels are empty and
    the exhaustive gallery is the same 96-image pool, so no candidate is
    assigned an unsupported positive ID.
    """
    queries = read_jsonl(ROOT / "eval/visual_boundary_nomatch/queries.jsonl")
    ids = json.loads((ROOT / "eval/visual_boundary_nomatch/gallery_ids.json").read_text(encoding="utf-8"))
    qrels = {query["qid"]: {} for query in queries}
    index_ids = json.loads((ROOT / "artifacts/siglip/item_ids.json").read_text(encoding="utf-8"))
    source = {item_id: i for i, item_id in enumerate(index_ids)}
    gallery = np.load(ROOT / "artifacts/siglip/image_embeddings.npy", allow_pickle=False)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery = gallery[[source[item_id] for item_id in ids]]
    vectors = encoder.encode_texts([trial_templates(query["text_en"])[template] for query in queries])
    return queries, ids, qrels, vectors @ gallery.T


def predict(scores, threshold, percentile, margin, calibration_floor=0.0, calibration_top=1.0,
            max_top_score=1.01):
    selected = []
    for original in scores:
        row = calibrate_scores(original, floor=calibration_floor, top_scale=calibration_top)
        cutoff = threshold
        if percentile > 0:
            cutoff = max(cutoff, float(np.quantile(row, percentile)))
        yes = row >= cutoff
        # A set requires either a candidate clearing the absolute gate or a
        # sufficiently separated leading cluster; otherwise return no match.
        finite = np.sort(row)[::-1]
        if len(finite) == 0 or finite[0] < max_top_score:
            yes[:] = False
        # The empty gate is separate from the selected-set threshold.
        if len(finite) < 2 or float(finite[0] - finite[1]) < margin:
            yes[:] = False
        selected.append(yes)
    return selected


def score_settings(queries, ids, qrels, scores, settings, calibration_floor=None):
    if calibration_floor is None:
        calibration_floor = calibration_floor_from_values(scores)
    return [
        {
            "config": {"threshold": t, "percentile": p, "margin": m, "empty_top": empty},
            "rows": [
                {"qid": q["qid"], **metrics(yes, [i for i,g in qrels[q["qid"]].items() if g >= 3], ids)}
                for q, yes in zip(queries, predict(scores, t, p, m,
                                                   calibration_floor=calibration_floor,
                                                   calibration_top=1.0,
                                                   max_top_score=empty))
            ],
        }
        for t, p, m, empty in settings
    ]


def calibration_floor_from_values(scores):
    """Frozen per-model lower bound; no labels or qrels enter it."""
    return float(np.quantile(np.asarray(scores, dtype=np.float64).ravel(), 0.01))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/cv_v4_20261108/boundary")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    encoder = Encoder("siglip", device="auto")
    settings = [(t, p, m, empty) for t in np.linspace(.10, .90, 17)
                for p in (0, .75, .9) for m in (0, .005, .01)
                for empty in (0.0, .15, .20, .25, .30, .35, .40)]
    # Template is a fixed secondary control, not selected from test.
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "version": "yes-no-boundary-v4",
              "score": "SigLIP2 text-image cosine", "relevance": "visual qrels grade 3",
              "selection": "dev micro-F1; no-match allowed; one threshold/percentile/margin frozen before test",
              "templates": {}}
    for template in trial_templates("x"):
        dev_q, dev_ids, dev_qrels, dev_scores = query_data("dev", encoder, template)
        test_q, test_ids, test_qrels, test_scores = query_data("test", encoder, template)
        unseen_q, unseen_ids, unseen_qrels, unseen_scores = load_unseen_queries(encoder, template)
        dev_trials = score_settings(dev_q, dev_ids, dev_qrels, dev_scores, settings)
        best = max(dev_trials, key=lambda r: (
            np.mean([x["f1"] for x in r["rows"]]),
            np.mean([x["precision"] for x in r["rows"]]),
            np.mean([x["no_match_correct"] for x in r["rows"] if x["no_match"]] or [1]),
            -np.mean([x["selected"] for x in r["rows"]]),
        ))
        test_config = (best["config"]["threshold"], best["config"]["percentile"],
                       best["config"]["margin"], best["config"].get("empty_top", 0.0))
        test_rows = score_settings(test_q, test_ids, test_qrels, test_scores, [test_config],
                                   calibration_floor=calibration_floor_from_values(dev_scores))[0]["rows"]
        unseen_rows = score_settings(unseen_q, unseen_ids, unseen_qrels, unseen_scores, [test_config],
                                     calibration_floor=calibration_floor_from_values(dev_scores))[0]["rows"]
        report["templates"][template] = {
            "selected": best["config"],
            "dev": {key: float(np.mean([x[key] for x in best["rows"]])) for key in ("precision", "recall", "f1", "exact_set_match", "selected", "no_match_correct")},
            "test": {key: float(np.mean([x[key] for x in test_rows])) for key in ("precision", "recall", "f1", "exact_set_match", "selected", "no_match_correct")},
            "test_rows": test_rows,
            "unseen_no_match_rows": unseen_rows,
            "unseen_no_match_accuracy": float(np.mean([x["no_match_correct"] for x in unseen_rows])),
        }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"dev_f1": v["dev"]["f1"], "test_f1": v["test"]["f1"], "exact": v["test"]["exact_set_match"]} for k,v in report["templates"].items()}, indent=2))


if __name__ == "__main__":
    main()
