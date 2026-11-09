"""Controlled SigLIP/CLIP prompt-template experiment on frozen visual audits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from curio.evaluate.metrics import average_precision, ndcg, precision, recall
from curio.evaluate.stats import bootstrap_ci, paired_permutation
from curio.represent.encoders import Encoder

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(ranking, qrels):
    return {
        "P@5": precision(ranking, qrels, 5),
        "nDCG@10": ndcg(ranking, qrels, 10),
        "AP@10": average_precision(ranking, qrels, 10),
        "Recall@10": recall(ranking, qrels, 10),
    }


def templates(text):
    text = text.strip()
    lower = text.lower()
    return {
        "raw": text,
        "a_photo_of": f"a photo of {lower}",
        "a_photograph_of": f"a photograph of {lower}",
        "museum_photograph": f"a museum photograph of {lower}",
        "close_up_photograph": f"a close-up photograph of {lower}",
        "a_picture_of": f"a picture of {lower}",
        "an_image_of": f"an image of {lower}",
        "capitalized": text.capitalize(),
        "period": f"{text}.",
        "image_of_no_article": f"image of {lower}",
        "plain_photo": f"photo of {lower}",
        "artifact_photo": f"a clear photo of a Chinese artifact: {lower}",
        "photographed_against_white": f"photograph of {lower} on a plain studio background",
    }


def evaluate(encoder, split_dir, gallery_dir, intent, catalog_ids, index_dir, image_vectors, trial):
    gallery_ids = json.loads((split_dir / "gallery_ids.json").read_text(encoding="utf-8"))
    qrels = {row["qid"]: row["grades"] for row in read_jsonl(split_dir / "qrels.jsonl")}
    queries = read_jsonl(split_dir / "queries.jsonl")
    rows = []
    vectors = []
    labels = []
    for query in queries:
        values = templates(query[f"text_{intent}"])
        text = values[trial]
        labels.extend([f"{query['qid']}::{name}" for name in values])
        vectors.append(text)
    embedded = encoder.encode_texts(vectors)
    lookup = dict(zip(labels, embedded))
    # The frozen vectors may be ordered differently from this audit gallery.
    source_ids = json.loads((index_dir / "item_ids.json").read_text(encoding="utf-8"))
    source_rows = {item_id: i for i, item_id in enumerate(source_ids)}
    selected = np.asarray([source_rows[item_id] for item_id in gallery_ids])
    gallery = image_vectors[selected]
    for query in queries:
        qid = query["qid"]
        text_vector = lookup[f"{qid}::{trial}"][None, :]
        scores = (text_vector @ gallery.T).ravel()
        order = np.lexsort((np.asarray(gallery_ids), -scores))
        ranking = [gallery_ids[i] for i in order]
        rows.append({"qid": qid, **metrics(ranking, qrels[qid]), "top10": ranking[:10]})
    return rows


def summarize(rows):
    result = {"queries": len(rows)}
    for metric in ("P@5", "nDCG@10", "AP@10", "Recall@10"):
        result[metric] = bootstrap_ci([row[metric] for row in rows])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("siglip", "clip"), default="siglip")
    parser.add_argument("--index", default="artifacts/siglip")
    parser.add_argument("--output", default="artifacts/cv_v4_20261108/prompt_templates")
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    index = ROOT / args.index
    image_vectors = np.load(index / "image_embeddings.npy", allow_pickle=False)
    image_vectors = image_vectors / np.linalg.norm(image_vectors, axis=1, keepdims=True)
    encoder = Encoder(args.encoder, device="auto")
    all_rows = {}
    selections = {}
    for intent in ("en", "zh"):
        dev = ROOT / "eval/visual_dev"
        test = ROOT / "eval/visual_test"
        dev_queries = read_jsonl(dev / "queries.jsonl")
        test_queries = read_jsonl(test / "queries.jsonl")
        trial_names = list(templates("x").keys())
        dev_embeddings = encoder.encode_texts([
            templates(query[f"text_{intent}"])[name]
            for query in dev_queries for name in trial_names
        ])
        test_embeddings = encoder.encode_texts([
            templates(query[f"text_{intent}"])[name]
            for query in test_queries for name in trial_names
        ])
        dev_qrels = {row["qid"]: row["grades"] for row in read_jsonl(dev / "qrels.jsonl")}
        test_qrels = {row["qid"]: row["grades"] for row in read_jsonl(test / "qrels.jsonl")}
        source_ids = json.loads((index / "item_ids.json").read_text(encoding="utf-8"))
        source_rows = {item_id: i for i, item_id in enumerate(source_ids)}
        dev_ids = json.loads((dev / "gallery_ids.json").read_text(encoding="utf-8"))
        test_ids = json.loads((test / "gallery_ids.json").read_text(encoding="utf-8"))
        dev_gallery = image_vectors[[source_rows[item_id] for item_id in dev_ids]]
        test_gallery = image_vectors[[source_rows[item_id] for item_id in test_ids]]

        def rank(queries, qrels, ids, gallery, embeddings):
            output = {}
            for j, name in enumerate(trial_names):
                rows = []
                text = embeddings[j::len(trial_names)]
                scores = text @ gallery.T
                for i, query in enumerate(queries):
                    order = np.lexsort((np.asarray(ids), -scores[i]))
                    ranking = [ids[k] for k in order]
                    rows.append({"qid": query["qid"], **metrics(ranking, qrels[query["qid"]]), "top10": ranking[:10]})
                output[name] = {"summary": summarize(rows), "rows": rows}
            return output

        dev_results = rank(dev_queries, dev_qrels, dev_ids, dev_gallery, dev_embeddings)
        selected = max(trial_names, key=lambda name: (dev_results[name]["summary"]["nDCG@10"]["mean"], name == "raw"))
        selections[intent] = selected
        all_rows[f"dev_{intent}"] = dev_results
        test_results = rank(test_queries, test_qrels, test_ids, test_gallery, test_embeddings)
        test_rows, raw_rows = test_results[selected]["rows"], test_results["raw"]["rows"]
        all_rows[f"test_{intent}"] = {
            "selected": selected,
            "selected_summary": summarize(test_rows),
            "raw_summary": summarize(raw_rows),
            "selected_rows": test_rows,
            "raw_rows": raw_rows,
            "paired_ndcg": paired_permutation(
                [r["nDCG@10"] for r in raw_rows], [r["nDCG@10"] for r in test_rows]
            ),
        }
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "encoder": encoder.metadata,
        "index_sha256": sha(index / "image_embeddings.npy"),
        "dev_queries_sha256": sha(ROOT / "eval/visual_dev/queries.jsonl"),
        "test_queries_sha256": sha(ROOT / "eval/visual_test/queries.jsonl"),
        "selection": selections,
        "selection_rule": "dev nDCG@10; raw wins ties",
        "evaluation_scope": "exhaustive AI visual audit gallery; not human gold or full-catalogue relevance",
        "results": all_rows,
    }
    (out / f"results_{args.encoder}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in selections.items()}, indent=2))


if __name__ == "__main__":
    main()
