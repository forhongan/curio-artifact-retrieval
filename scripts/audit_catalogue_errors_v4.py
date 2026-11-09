"""Export frozen-C classification errors, confidence bins, and image sheets.

This is post-evaluation inspection. It never modifies frozen catalogue labels,
selects a new C, or treats a confidence score as proof of correctness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.evaluate.catalogue_tasks import single_label  # noqa: E402

OUT = ROOT / "artifacts/catalogue_error_audit_20261108"
SOURCE = ROOT / "artifacts/cv_v4_20261108/catalogue_tasks_v3_primary_plus_candidates/report.json"
LABELS = ROOT / "artifacts/cv_v4_20261108/catalogue_v3_primary_plus_candidates/labels_catalogue_v2.jsonl"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    audited = json.loads((ROOT / "artifacts/cv_v4_20261108/catalogue_tasks_audited_20261108/report.json").read_text(encoding="utf-8"))
    ids = json.loads((ROOT / "artifacts/siglip/item_ids.json").read_text(encoding="utf-8"))
    vectors = np.load(ROOT / "artifacts/siglip/image_embeddings.npy")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    by_id = {row["item_id"]: row for row in read_rows(LABELS)}
    rows = [by_id[item_id] for item_id in ids]
    catalog = {row["item_id"]: row for row in read_rows(ROOT / "data/snapshots/collection_v2.jsonl")}
    output = []
    summaries = {}
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    for domain, task in source["tasks"].items():
        if not task["methods"]:
            continue
        method = task["methods"]["siglip_logreg"]
        classes = task["classes"]
        labels = [single_label(row, domain) for row in rows]
        train = [i for i, row in enumerate(rows) if row["split"] == "train" and labels[i] in classes]
        test = [i for i, row in enumerate(rows) if row["split"] == "test" and labels[i] in classes]
        model = LogisticRegression(C=method["selected_C"], class_weight="balanced", max_iter=1500, random_state=0)
        model.fit(vectors[train], [labels[i] for i in train])
        probs = model.predict_proba(vectors[test])
        saved = audited["tasks"][domain]["methods"]["siglip_logreg"]["test"]
        if saved["item_ids"] != [ids[i] for i in test]:
            raise ValueError(f"Export item order mismatch: {domain}")
        pred = np.asarray(saved["predicted"])
        records = []
        for local, index in enumerate(test):
            item_id = ids[index]
            order = np.argsort(-probs[local])
            record = {
                "domain": domain, "item_id": item_id, "truth": labels[index],
                "predicted": str(pred[local]), "correct": labels[index] == pred[local],
                "top1_probability": float(probs[local, order[0]]),
                "confidence_source": "same-C refit; inspection only; saved class decision is authoritative",
                "margin": float(probs[local, order[0]] - probs[local, order[1]]),
                "top3": [{"class": str(model.classes_[j]), "probability": float(probs[local, j])}
                         for j in order[:3]],
                "image_path": catalog[item_id]["image_path"],
                "source": catalog[item_id]["source"],
                "source_id": catalog[item_id]["source_id"],
                "source_url": catalog[item_id].get("source_url"),
                "official_catalog": rows[index]["official_catalog"],
                "derived_ontology": rows[index]["derived_ontology"],
            }
            records.append(record)
        # Confirm the exported errors reproduce the existing fixed evaluation.
        accuracy = sum(row["correct"] for row in records) / len(records)
        if abs(accuracy - method["test"]["accuracy"]) > 1e-10:
            raise ValueError(f"Frozen classification mismatch for {domain}: {accuracy}")
        errors = sorted([row for row in records if not row["correct"]],
                        key=lambda row: (-row["top1_probability"], row["item_id"]))
        selected = errors[:4] + errors[-4:] if len(errors) > 8 else errors
        bins = []
        for lo, hi in ((0, .5), (.5, .7), (.7, .9), (.9, 1.00001)):
            subset = [row for row in records if lo <= row["top1_probability"] < hi]
            bins.append({"range": [lo, min(1, hi)], "n": len(subset),
                         "errors": sum(not row["correct"] for row in subset),
                         "accuracy": sum(row["correct"] for row in subset) / len(subset) if subset else None})
        summaries[domain] = {"n": len(records), "errors": len(errors), "accuracy": accuracy,
                             "confidence_bins": bins, "selected_ids": [row["item_id"] for row in selected],
                             "selected_C_frozen": method["selected_C"]}
        columns, width, height = 4, 380, 355
        sheet = Image.new("RGB", (columns * width, 40 + height * max(1, (len(selected) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"{domain}: highest-confidence errors first, lowest-confidence errors last", fill="black", font=font)
        for i, row in enumerate(selected):
            x, y = (i % columns) * width, 40 + (i // columns) * height
            with Image.open(ROOT / row["image_path"]) as opened:
                im = ImageOps.exif_transpose(opened).convert("RGB")
                im.thumbnail((width - 18, 260))
                sheet.paste(im, (x + (width - im.width) // 2, y))
            texts = [row["item_id"], f"true={row['truth']} | pred={row['predicted']}",
                     f"p={row['top1_probability']:.3f} margin={row['margin']:.3f}"]
            for j, text in enumerate(texts):
                draw.text((x + 8, y + 268 + j * 24), text, fill="black", font=font)
        sheet.save(OUT / f"{domain}_errors.jpg", quality=93)
        output.extend(records)
        print(domain, summaries[domain]["n"], summaries[domain]["errors"], flush=True)
    (OUT / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({
        "source_report_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "label_sha256": hashlib.sha256(LABELS.read_bytes()).hexdigest(),
        "policy": "Frozen source C; no test-based model or threshold selection; inspection only",
        "domains": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
