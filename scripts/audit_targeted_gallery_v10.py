"""Diagnose full-gallery extrapolation of the initial targeted visual heads."""

import json
import hashlib
from pathlib import Path
import numpy as np
from prepare_boundary_followup import sheet


def main():
    root = Path(__file__).resolve().parents[1]
    base = root / "eval/boundary_targeted_v10"
    out = root / "artifacts/targeted_boundary_v10"
    heads = json.loads((out / "visual_heads.json").read_text())["heads"]
    report = json.loads((base / "report.json").read_text())
    ids = json.loads((root / "artifacts/siglip_adapted/item_ids.json").read_text())
    x = np.load(root / "artifacts/siglip_adapted/image_embeddings.npy")
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    gallery = {
        r["item_id"]: r
        for r in map(
            json.loads, (root / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
        )
    }
    previous = json.loads(
        (root / "artifacts/joint_vlm_v9/full_gallery_blue_white.json").read_text()
    )
    assert previous["total"] == 4692 and previous["complete"]
    print("decision sample", str(previous["decisions"][0])[:400])
    gray = {r["item_id"] for r in previous["decisions"] if r["reason"] == "color_not_observable"}
    assert len(gray) == 1063
    excluded = {
        r["object_group"] for r in map(json.loads, (base / "labels.jsonl").read_text().splitlines())
    }
    summary = {}
    labels = []
    for q, h in heads.items():
        p = 1 / (1 + np.exp(-(x @ np.array(h["coefficient"]) + h["intercept"])))
        if q in ("red_bowl", "blue_white"):
            p[[i for i, ident in enumerate(ids) if ident in gray]] = 0
        threshold = report["methods"]["visual_head"]["per_query"][q]["calibration"]["threshold"]
        selected = [
            {**gallery[ident], "score": float(p[i]), "qid": q}
            for i, ident in enumerate(ids)
            if p[i] >= threshold
        ]
        all_rows = [
            {"item_id": ident, "qid": q, "score": float(p[i]), "decision": bool(p[i] >= threshold)}
            for i, ident in enumerate(ids)
        ]
        (out / (q + "_gallery.jsonl")).write_text("".join(json.dumps(r) + "\n" for r in all_rows))
        candidates = [r for r in selected if r["object_group"] not in excluded]
        # Freeze a diagnostic sample before judging; this is no new training/evaluation set.
        candidates.sort(
            key=lambda r: hashlib.sha256(
                ("targeted-audit-v10/" + q + r["object_group"]).encode()
            ).hexdigest()
        )
        sample = candidates[:16]
        sheet(
            [dict(r, caption=f"score={r['score']:.3f}") for r in sample],
            out / (q + "_gallery_audit.png"),
            q + " | full gallery accepted sample",
        )
        labels += [
            {
                k: r[k]
                for k in [
                    "qid",
                    "item_id",
                    "object_group",
                    "image_path",
                    "title",
                    "medium",
                    "score",
                    "source_url",
                ]
            }
            for r in sample
        ]
        summary[q] = {
            "total": 4692,
            "yes": len(selected),
            "no": 4692 - len(selected),
            "threshold": threshold,
            "audit_sample": len(sample),
            "excluded_reviewed_groups": len(excluded),
        }
    (out / "gallery_audit_sample.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n"
    )
    (base / "full_gallery_counts.json").write_text(
        json.dumps(
            {
                "counts": summary,
                "color_gate_source": previous["job_id"],
                "color_excluded": len(gray),
                "note": "Counts are predictions, not verified matches. This exploratory head is not deployed.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
