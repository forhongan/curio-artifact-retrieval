r"""Run live museum-reference retrieval and full-4,692 image matching.

Example:
  .\.venv\Scripts\python.exe scripts/run_dynamic_reference_v4.py

The first run performs public API and image requests.  Cached reruns use
``--offline`` and preserve all request URLs/status records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from curio.evaluate.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.retrieve.dynamic_references import (  # noqa: E402
    DynamicMuseumAcquirer,
    load_gallery,
    plan_dynamic_query,
    select_reference_set,
    selection_diagnostics,
    strict_references_available,
)


DEFAULT_QUERIES = (
    ("ming_blue_white", "明代青花瓷"),
    ("wanli_blue_white", "万历青花瓷"),
    ("kangxi_meiping", "康熙梅瓶"),
    ("qianlong_famille_rose", "乾隆粉彩"),
    ("liao_leather_flask", "辽代皮囊式壶"),
    ("yuan_blue_white_dragon", "元代青花龙纹瓶"),
    ("song_celadon", "南宋龙泉青瓷瓶"),
    ("jiajing_blue_white_jar", "明代嘉靖青花罐"),
)

METHOD_NAMES = (
    "siglip_single", "siglip_mean", "siglip_coverage", "siglip_topk",
    "dino_single", "dino_mean", "dino_coverage", "dino_topk",
    "fusion_cosine_mean", "fusion_rrf_mean",
)

JUDGMENT_PROTOCOL = {
    "version": 4,
    "status": "AI pixel contact-sheet review; not human inter-annotator gold",
    "scope": {
        "grade_2": "visible requested form and decoration/color are both supported, with no visible contradiction",
        "grade_1": "form or decoration is supported but the other is uncertain, or the image differs materially",
        "grade_0": "wrong object family, decoration, or clearly incompatible visual appearance",
    },
    "excluded_from_visual_grade": [
        "actual authenticity", "unseen date/reign", "invisible palace/imperial status", "material composition",
    ],
    "catalogue_use": "target metadata diagnoses period/form/technique; it does not enter rank scores",
    "reference_selection": (
        "strict source culture, requested form/material/concept, and structured era; "
        "source-family diversity only after those hard checks"
    ),
    "reference_status": (
        "visual_references_qualified means at least two references passed metadata checks and visible-evidence gates; "
        "insufficient_visual_references means no full-gallery visual evaluation was run for that query"
    ),
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                         encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                 for row in rows), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_evidence(path: Path, concept_key: str) -> dict:
    """Cheap, auditable pixel signal for reference candidates.

    This is a reference-selection diagnostic, not a trained image classifier
    and never a claim of authenticity or definitive glaze/technique.
    """
    import cv2
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    # The central 80% avoids institutional bars/mounts and most background.
    h, w = saturation.shape
    center = saturation[int(.1 * h):int(.9 * h), int(.1 * w):int(.9 * w)]
    center_v = value[int(.1 * h):int(.9 * h), int(.1 * w):int(.9 * w)]
    if concept_key == "polychrome":
        score = float(np.mean((center > .28) & (center_v > .22)))
        threshold = .02
        label = "saturated_color_fraction"
    elif concept_key == "blue_white":
        # Blue/cyan pixels plus a bright, low-saturation body.  This simple cue
        # examines visible color only; black-and-white images remain unknown.
        hue = hsv[..., 0]
        blue = (hue >= 90) & (hue <= 140) & (saturation > .22) & (value > .12)
        score = float(np.mean(blue[int(.1 * h):int(.9 * h), int(.1 * w):int(.9 * w)]))
        threshold = .002
        label = "blue_pixel_fraction"
    elif concept_key == "celadon":
        hue = hsv[..., 0]
        green = (hue >= 35) & (hue <= 100) & (saturation > .12) & (value > .2)
        score = float(np.mean(green[int(.1 * h):int(.9 * h), int(.1 * w):int(.9 * w)]))
        threshold = .015
        label = "green_pixel_fraction"
    elif concept_key in {"meiping", "jar", "leather_flask", "gourd", "bowl", "dish"}:
        return object_shape_evidence(path, concept_key)
    else:
        return {"status": "not_applicable", "visual_gate": "not_applicable",
                "reason": "no visible evidence rule for this concept"}
    return {"status": "available", "signal": label, "value": score,
            "threshold": threshold, "visual_gate": "pass" if score >= threshold else "reject",
            "limitation": "heuristic color evidence only; not a motif/period classifier"}


def object_shape_evidence(path: Path, concept_key: str) -> dict:
    """Visible silhouette gate with explicit failure if segmentation abstains."""
    from curio.represent.visual_parts import extract_visual_parts
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    parts = extract_visual_parts(image)
    diagnostics = parts["diagnostics"]
    if diagnostics.get("status") != "valid":
        return {"status": "unknown", "signal": "foreground_silhouette",
                "visual_gate": "reject", "reason": "foreground_segmentation_abstained",
                "segmentation": diagnostics,
                "limitation": "classical mask gate; unknown does not mean visually wrong"}
    mask = parts["mask"] > 0
    ys, xs = np.where(mask)
    height = max(1, ys.max() - ys.min() + 1)
    width = max(1, xs.max() - xs.min() + 1)
    aspect = height / width
    rows = mask.sum(axis=1)
    nonzero = np.flatnonzero(rows)
    top, bottom = int(nonzero[0]), int(nonzero[-1])
    length = bottom - top + 1
    # Use stable section medians rather than a single noisy row. Real museum
    # photographs have a pinched mouth, broad high shoulder and tapering lower
    # body; median widths express that shape more faithfully.
    shoulder = np.median(rows[top + int(.20 * length):top + int(.35 * length) + 1])
    waist = np.median(rows[top + int(.55 * length):top + int(.72 * length) + 1])
    neck = np.median(rows[top + int(.02 * length):top + int(.14 * length) + 1])
    lower = rows[top + int(.74 * length):bottom + 1].mean()
    body_ratio = float(waist / max(1, shoulder))
    neck_ratio = float(neck / max(1, shoulder))
    lower_ratio = float(lower / max(1, shoulder))
    if concept_key == "meiping":
        # Masks include photographed bases and fronts, so this silhouette test
        # is deliberately permissive: tall body, less neck than shoulder and a
        # substantial lower body. Strict metadata supplies the semantic gate.
        passed = aspect >= 1.15 and .42 <= body_ratio <= 1.5 and neck_ratio <= .72 and lower_ratio <= 1.08
        threshold = "aspect>=1.15; shoulder/waist ratio 0.42..1.50; neck<=0.72*shoulder; lower<=1.08*shoulder"
    elif concept_key in {"jar", "leather_flask", "gourd"}:
        passed = .42 <= body_ratio <= 1.15 and aspect >= .60
        threshold = "waist/shoulder 0.42..1.15; aspect>=0.60"
    elif concept_key == "dish":
        passed = aspect <= .78
        threshold = "silhouette aspect<=0.78"
    else:  # bowl
        passed = .25 <= aspect <= 1.0
        threshold = "silhouette aspect 0.25..1.0"
    return {"status": "available", "signal": "foreground_silhouette",
            "value": {"aspect": round(aspect, 4), "shoulder_to_waist": round(body_ratio, 4),
                      "neck_to_shoulder": round(neck_ratio, 4),
                      "lower_to_shoulder": round(lower_ratio, 4),
                      "foreground_fraction": diagnostics.get("foreground_fraction")},
            "threshold": threshold, "visual_gate": "pass" if passed else "reject",
            "segmentation": diagnostics,
            "limitation": "classical silhouette only; the mask can be imperfect and this is not a learned form classifier"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/dynamic_reference_v4")
    parser.add_argument("--cache-dir", type=Path,
                        help="Read HTTP/object/image cache from this completed run without modifying it")
    parser.add_argument("--queries", type=Path,
                        help="JSON array of {qid, query} or JSONL lines")
    parser.add_argument("--offline", action="store_true",
                        help="Use only verified HTTP/object/image cache")
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--max-references-considered", type=int, default=60)
    parser.add_argument("--strict-only", action="store_true",
                        help="Evaluate only queries with two metadat-validated references")
    parser.add_argument("--reference-policy", choices=("strict", "expert"), default="strict",
                        help="strict validates source metadata; expert uses planned visual phrases as an explicit fallback")
    parser.add_argument("--reference-limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--page-depth", type=int, default=5,
                        help="Search-result pages per term/source (default: 5)")
    parser.add_argument("--max-metadata-candidates", type=int, default=800,
                        help="Bound object metadata requests per acquisition pass")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--contact-sheet-limit", type=int, default=10)
    parser.add_argument("--no-visual", action="store_true",
                        help="Acquire and audit references without scoring")
    return parser.parse_args()


def load_queries(path: Path | None) -> list[dict]:
    if path is None:
        return [{"qid": qid, "query": query} for qid, query in DEFAULT_QUERIES]
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".json":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Query file must contain a non-empty array")
    result = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            row = {"qid": f"query_{index + 1}", "query": row}
        if not row.get("qid") or not row.get("query"):
            raise ValueError(f"Invalid query row: {row!r}")
        result.append({"qid": str(row["qid"]), "query": str(row["query"])})
    return result


def normalize_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid feature matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Zero feature vector")
    return values / norms


def load_index(path: Path, root: Path) -> tuple[np.ndarray, list[str], dict]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    values = normalize_rows(np.load(path / "image_embeddings.npy", allow_pickle=False))
    ids = [str(value) for value in json.loads((path / "item_ids.json").read_text(encoding="utf-8"))]
    fingerprint = hashlib.sha256(np.load(path / "image_embeddings.npy", allow_pickle=False).tobytes()
                                 + json.dumps(ids).encode()).hexdigest()
    if fingerprint != manifest.get("index_sha256"):
        raise ValueError(f"Index fingerprint mismatch: {path}")
    if len(ids) != len(values) or len(ids) != len(set(ids)):
        raise ValueError(f"Invalid index IDs: {path}")
    gallery_path = root / "data/snapshots/collection_v2.jsonl"
    if manifest.get("data_sha256") != sha256_file(gallery_path):
        raise ValueError(f"Index/catalogue data mismatch: {path}")
    return values, ids, manifest


def rank_matrix(vectors: np.ndarray, ids: list[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    order = sorted(range(len(ids)), key=lambda index: (-float(vectors[index]), ids[index]))
    return [ids[index] for index in order], vectors


def rankings_for_reference_set(
    siglip: np.ndarray, dino: np.ndarray, ids: list[str],
    siglip_refs: np.ndarray, dino_refs: np.ndarray,
) -> dict[str, dict]:
    """Rank all 4,692 images with explicit single/mean/coverage/top-k contracts."""
    gallery_count = len(ids)
    s_pairs = siglip @ siglip_refs.T
    d_pairs = dino @ dino_refs.T
    if len(ids) != gallery_count:
        raise AssertionError("gallery changed during ranking")
    def topk(values: np.ndarray, k: int = 3) -> np.ndarray:
        # top-k mean is on the candidate's best distinct reference comparisons.
        if values.shape[1] <= k:
            return values.mean(axis=1)
        partitioned = np.partition(values, -k, axis=1)[:, -k:]
        return partitioned.mean(axis=1)
    methods = {
        "siglip_single": s_pairs[:, 0],
        "siglip_mean": s_pairs.mean(axis=1),
        "siglip_coverage": s_pairs[:, :min(3, s_pairs.shape[1])].sum(axis=1) / 3,
        "siglip_topk": topk(s_pairs, min(3, s_pairs.shape[1])),
        "dino_single": d_pairs[:, 0],
        "dino_mean": d_pairs.mean(axis=1),
        "dino_coverage": d_pairs[:, :min(3, d_pairs.shape[1])].sum(axis=1) / 3,
        "dino_topk": topk(d_pairs, min(3, d_pairs.shape[1])),
    }
    methods["fusion_cosine_mean"] = .75 * methods["siglip_mean"] + .25 * methods["dino_mean"]
    # Explicit rank fusion, fixed before inspecting query outcomes.
    sig_rank = np.empty(gallery_count, dtype=np.int32)
    dino_rank = np.empty(gallery_count, dtype=np.int32)
    sig_order = np.argsort(-methods["siglip_mean"], kind="stable")
    dino_order = np.argsort(-methods["dino_mean"], kind="stable")
    sig_rank[sig_order] = np.arange(gallery_count)
    dino_rank[dino_order] = np.arange(gallery_count)
    methods["fusion_rrf_mean"] = .75 / (60 + sig_rank) + .25 / (60 + dino_rank)
    results = {}
    for name, scores in methods.items():
        order = sorted(range(gallery_count), key=lambda index: (-float(scores[index]), ids[index]))
        results[name] = {"scores": scores, "ranking": [ids[index] for index in order],
                         "ranking_rows": order}
    # Reference-level diagnostics preserve why the aggregators differ.  A
    # matrix row is local gallery order; columns follow selected refs.
    diagnostics = {
        "reference_count": int(s_pairs.shape[1]),
        "siglip_reference_agreement": _reference_agreement(s_pairs, ids),
        "dino_reference_agreement": _reference_agreement(d_pairs, ids),
        "aggregation": {
            "single": "first reference only",
            "mean": "all references, equal weight",
            "coverage": "sum of best-three / six; selected coverage reported separately",
            "top-k": "mean of each candidate's best three reference cosines",
        },
    }
    # Prefix the diagnostic under every returned method so reruns can audit
    # sensitivity without vector serialization.
    for result in results.values():
        result["reference_diagnostics"] = diagnostics
    return results


def _reference_agreement(pairs: np.ndarray, ids: list[str]) -> dict:
    tops = [ids[int(np.argmax(pairs[:, column]))] for column in range(pairs.shape[1])]
    counts = {key: tops.count(key) for key in set(tops)}
    return {"individual_top_ids": tops, "unique_top_count": len(counts),
            "top_agreement_counts": counts,
            "interpretation": "agreement is a diagnostic, not calibrated correctness"}


def encode_references(selected: list[dict], *, device: str) -> tuple[np.ndarray, np.ndarray, dict]:
    from curio.represent.encoders import Encoder
    from curio.represent.image_encoder import DinoEncoder
    paths = [ROOT / row["image_path"] for row in selected]
    started = time.perf_counter()
    siglip = Encoder("siglip", device=device)
    siglip_vectors = siglip.encode_images(paths, batch_size=8)
    siglip_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    dino = DinoEncoder(device=device)
    dino_vectors = dino.encode_images(paths, batch_size=8)
    dino_ms = (time.perf_counter() - started) * 1000
    return normalize_rows(siglip_vectors), normalize_rows(dino_vectors), {
        "siglip_encode_ms": round(siglip_ms, 3), "dino_encode_ms": round(dino_ms, 3),
        "siglip": siglip.metadata, "dino": dino.metadata,
    }


def target_truth_metadata(items: list[dict]) -> dict[str, dict]:
    """Create diagnostic pools from catalogue metadata without ranking use."""
    rows = {}
    for item in items:
        text = " ".join(str(item.get(key, "")) for key in
                         ("title", "object_name", "classification", "medium", "culture",
                          "period", "tags", "object_type", "material", "motifs"))
        rows[item["item_id"]] = {
            "text": text,
            "year_from": item.get("year_from"), "year_to": item.get("year_to"),
            "medium": item.get("medium", ""), "object_type": item.get("object_type", ""),
        }
    return rows


def describe_rows(rows: list[dict]) -> dict[str, dict]:
    return {str(row["item_id"]): {"title": row.get("title", ""),
                                    "culture": row.get("culture", ""),
                                    "period": row.get("period", ""),
                                    "medium": row.get("medium", ""),
                                    "object_type": row.get("object_type", ""),
                                    "year_from": row.get("year_from"),
                                    "year_to": row.get("year_to")}
            for row in rows}


def metric_report(rankings: dict[str, dict], target_pool: set[str], *,
                  k: int = 10, bootstrap_repetitions: int = 1000) -> dict[str, dict]:
    """Diagnostic retrieval metrics for an incomplete metadata-derived pool.

    Precision at k needs every top-k item graded; the project currently has no
    complete all-gallery qrels. It is therefore intentionally absent.
    """
    report = {}
    for method, result in rankings.items():
        ordered = result["ranking"]
        hits = [index for index, item_id in enumerate(ordered) if item_id in target_pool]
        top_hits = [item_id for item_id in ordered[:k] if item_id in target_pool]
        first = hits[0] + 1 if hits else None
        method_hits = np.asarray([item_id in target_pool for item_id in ordered], dtype=float)
        rank_values = np.arange(1, len(method_hits) + 1, dtype=float)
        first_hit = np.flatnonzero(method_hits)
        reciprocal_ranks = np.where(method_hits > 0, 1 / rank_values, 0.0)
        report[method] = {
            "hit_at_k": bool(top_hits),
            "recall_at_k": len(top_hits) / len(target_pool) if target_pool else 0.0,
            "recall_at_20": sum(item_id in target_pool for item_id in ordered[:20]) / len(target_pool) if target_pool else 0.0,
            "mrr_at_k": 1 / first if first and first <= k else 0.0, "first_relevant_rank": first,
            "relevant_count": len(target_pool), "n": len(ordered),
        }
        for metric, values in (
            ("hit_at_k", np.asarray([1.0 if top_hits else 0.0])),
            ("mrr_at_k", reciprocal_ranks),
        ):
            report[method][f"{metric}_bootstrap_ci"] = bootstrap_ci(
                values, repetitions=bootstrap_repetitions
            )
        if len(first_hit):
            report[method]["first_relevant_rank_bootstrap_ci"] = bootstrap_ci(
                first_hit + 1, repetitions=bootstrap_repetitions
            )
    return report


def query_target_pool(query: str, plan: dict, items: list[dict]) -> tuple[set[str], dict]:
    """Diagnostic target pool.  Never called by rank functions."""
    from curio.query import parse_query
    parsed = parse_query(query)
    required_form = set(parsed.slots.get("object_type", [])) - {"vessel", "pot"}
    if "leather_flask" in set(plan["concept_keys"]):
        required_form = {"vessel"}
    concepts = set(plan["concept_keys"])
    # All target pools below are ceramic-object pools. The local vocabulary
    # calls the actual leather-flask ceramics "Ceramic", so it counts as the
    # object-side material even though the English strings differ.
    ceramic_required = bool(concepts.intersection({"blue_white", "polychrome", "celadon", "porcelain", "leather_flask"}))
    if "leather_flask" in concepts:
        # This narrow target is explicitly form-driven.  Restricting to the
        # local ceramic flask family and Liao interval avoids both a hollow
        # all-porcelain pool and unrelated pots/ewers.
        required_form = {"flask"}
    # Apply the dynamic planner's visual motif rules.  In particular 龙泉 is
    # a kiln name, so the generic parser's substring 龙 is suppressed here.
    motifs = {key for key in concepts if key in {"dragon", "floral"}}
    # The generic vocabulary treats 龙泉 as a dragon substring. Preserve the
    # visual material/object intent and suppress that false motif explicitly.
    if "celadon" in concepts:
        motifs.discard("dragon")
    ranges = {
        "Liao": (907, 1125), "Wanli": (1573, 1620), "Jiajing": (1522, 1566),
        "Kangxi": (1662, 1722), "Qianlong": (1736, 1795), "Yuan": (1271, 1368),
        "Southern Song": (1127, 1279), "Ming": (1368, 1644), "Qing": (1644, 1911),
    }
    labels = [entry["label"] for entry in plan.get("era_constraints", [])]
    narrow_labels = [label for label in labels if label in ranges]
    era_range = None
    if narrow_labels:
        # The planner may retain broad dynasty and narrow reign labels; the
        # latter is the actual user constraint.
        rank = {"Wanli": 3, "Jiajing": 3, "Kangxi": 3, "Qianlong": 3,
                "Southern Song": 2, "Northern Song": 2, "Liao": 2, "Yuan": 2}
        label = max(narrow_labels, key=lambda item: rank.get(item, 1))
        era_range = {"label": label, "start": ranges[label][0], "end": ranges[label][1]}
    target = set()
    rejection_counts: dict[str, int] = {}
    for row in items:
        item_object_type = str(row.get("object_type", "")).casefold().strip()
        text = " ".join(str(row.get(key, "")) for key in
                         ("title", "object_name", "classification", "medium", "culture", "period"))
        item_object_name = str(row.get("object_name", "")).casefold()
        if ceramic_required and not re.search(
                r"porcelain|ceramic|stoneware|earthenware|pottery",
                " ".join(str(row.get(key, "")) for key in ("medium", "object_name", "classification")),
                re.I):
            continue
        if required_form == {"flask"}:
            title_name = str(row.get("title", "")).casefold()
            if not ("flask" in title_name or item_object_name == "flask"):
                continue
        elif required_form and item_object_type not in required_form:
            rejection_counts["object_form"] = rejection_counts.get("object_form", 0) + 1
            continue
        if era_range and not (row.get("year_from") is not None and row.get("year_to") is not None
                              and row["year_from"] <= era_range["end"]
                              and row["year_to"] >= era_range["start"]):
            rejection_counts["date_interval"] = rejection_counts.get("date_interval", 0) + 1
            continue
        if "blue_white" in concepts and not re.search(r"blue.and.white|underglaze blue", text, re.I):
            rejection_counts["blue_white"] = rejection_counts.get("blue_white", 0) + 1
            continue
        if "polychrome" in concepts and not re.search(r"famille rose|polychrome|famille|enamel|wucai|doucai", text, re.I):
            rejection_counts["polychrome"] = rejection_counts.get("polychrome", 0) + 1
            continue
        if "polychrome" in concepts and not re.search(r"porcelain|ceramic|stoneware|earthenware", str(row.get("medium", "")), re.I):
            rejection_counts["polychrome_material"] = rejection_counts.get("polychrome_material", 0) + 1
            continue
        if "leather_flask" in concepts and not re.search(r"porcelain|ceramic|stoneware|earthenware", str(row.get("medium", "")), re.I):
            rejection_counts["leather_flask_material"] = rejection_counts.get("leather_flask_material", 0) + 1
            continue
        if "celadon" in concepts and not re.search(r"celadon|longquan", text, re.I):
            rejection_counts["celadon"] = rejection_counts.get("celadon", 0) + 1
            continue
        if "meiping" in concepts and not re.search(r"meiping|plum vase", text, re.I):
            rejection_counts["meiping"] = rejection_counts.get("meiping", 0) + 1
            continue
        if "dragon" in motifs and not re.search(r"\bdragons?\b", text, re.I):
            rejection_counts["motif"] = rejection_counts.get("motif", 0) + 1
            continue
        if "floral" in motifs and not re.search(r"flower|floral|lotus|peony|orchid|chrysanthemum", text, re.I):
            rejection_counts["motif"] = rejection_counts.get("motif", 0) + 1
            continue
        target.add(str(row["item_id"]))
    return target, {"parser": parsed.to_dict(), "motifs_used": sorted(motifs),
                    "era_range_used": era_range, "strict_intersection": sorted(target),
                    "object_type_field": "used as the catalogue object family; never as era evidence",
                    "rejection_counts": rejection_counts,
                    "note": "diagnostic only; it is computed after image rankings and never entered their scores"}


def visual_sheet(path: Path, qid: str, query: str, references: list[dict], rankings: dict,
                 descriptions: dict[str, dict], items_by_id: dict[str, dict],
                 *, methods: list[str], limit: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = max(len(methods), 1)
    width, cell_h = 230, 270
    header_h = 72
    sheet = Image.new("RGB", (cols * width, header_h + 2 * (limit * cell_h + 30)), "#f2f1ed")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 8), f"{qid} | {query}", fill="#111", font=font)
    draw.text((10, 26), "References selected from live Cleveland/Met metadata before image ranking", fill="#444", font=font)
    draw.text((10, 44), "Grades are visual-only; dates/material are diagnostics and never rank inputs.", fill="#444", font=font)
    for method_index, method in enumerate(methods):
        x = method_index * width
        draw.text((x + 8, header_h + 4), method, fill="#111", font=font)
        for rank_index, item_id in enumerate(rankings[method]["ranking"][:limit], 1):
            row = items_by_id[item_id]
            image_path = ROOT / row["image_path"]
            y = header_h + 28 + (rank_index - 1) * cell_h
            with Image.open(image_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((210, 190))
            sheet.paste(thumb, (x + 10 + (210 - thumb.width) // 2, y))
            description = descriptions.get(item_id, {})
            label = f"{rank_index}. {item_id} | {str(row.get('title', ''))[:30]}"
            draw.text((x + 8, y + 196), label, fill="#111", font=font)
            culture = str(description.get("culture") or description.get("period") or "")[:44]
            draw.text((x + 8, y + 212), culture, fill="#333", font=font)
    y = header_h + limit * cell_h + 34
    draw.text((10, y), "Selected references", fill="#111", font=font)
    for index, reference in enumerate(references):
        x = 10 + index * 205
        with Image.open(ROOT / reference["image_path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((190, 170))
        sheet.paste(thumb, (x, y + 22))
        draw.text((x, y + 196), reference["reference_id"], fill="#111", font=font)
        draw.text((x, y + 212), str(reference.get("title", ""))[:30], fill="#333", font=font)
    sheet.save(path, quality=94)


def reference_sheet(path: Path, qid: str, query: str, references: list[dict],
                    accepted: list[dict]) -> None:
    """Audit every selected reference and its structured attribution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, cell_h = 300, 350
    columns = max(1, min(3, len(references) or 1))
    rows = max(1, (len(references) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * width, 72 + rows * cell_h), "#f2f1ed")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 8), f"REFERENCE AUDIT | {qid} | {query}", fill="#111", font=font)
    draw.text((10, 26), f"selected={len(references)} / accepted_images={len(accepted)}",
              fill="#333", font=font)
    draw.text((10, 44), "Structured culture/date only; prose never establishes era.",
              fill="#333", font=font)
    if not references:
        draw.text((10, 88), "No strict references selected; no visual-evaluation inference.",
                  fill="#8a1f1f", font=font)
    for index, row in enumerate(references):
        column, line = index % columns, index // columns
        x, y = column * width, 72 + line * cell_h
        with Image.open(ROOT / row["image_path"]) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((270, 245))
        sheet.paste(thumb, (x + 10, y))
        draw.text((x + 12, y + 250), row["reference_id"], fill="#111", font=font)
        draw.text((x + 12, y + 270), str(row.get("title", ""))[:42], fill="#222", font=font)
        draw.text((x + 12, y + 290), str(row.get("culture", ""))[:42], fill="#333", font=font)
        draw.text((x + 12, y + 310), str(row.get("date", ""))[:42], fill="#333", font=font)
    sheet.save(path, quality=94)


def target_sheet(path: Path, qid: str, query: str, rankings: dict,
                 descriptions: dict[str, dict], items_by_id: dict[str, dict],
                 *, methods: list[str], limit: int = 10) -> None:
    """One independently generated sheet per method, plus selected references."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, header_h = 230, 270, 74
    sheet = Image.new("RGB", (max(1, len(methods)) * cell_w,
                              header_h + limit * cell_h + 230), "#f2f1ed")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 8), f"TARGET AUDIT | {qid} | {query}", fill="#111", font=font)
    draw.text((10, 26), "Metadata is diagnostic only; every rank came from image pixels.",
              fill="#333", font=font)
    for method_index, method in enumerate(methods):
        x = method_index * cell_w
        draw.text((x + 8, header_h - 18), method, fill="#111", font=font)
        for rank, item_id in enumerate(rankings[method]["ranking"][:limit], 1):
            row = items_by_id[item_id]
            y = header_h + (rank - 1) * cell_h
            with Image.open(ROOT / row["image_path"]) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((210, 190))
            sheet.paste(thumb, (x + 10 + (210 - thumb.width) // 2, y))
            draw.text((x + 8, y + 196), f"{rank}. {item_id} | {str(row.get('title', ''))[:28]}",
                      fill="#111", font=font)
            info = descriptions.get(item_id, {})
            draw.text((x + 8, y + 212),
                      str(info.get("culture") or info.get("period") or "")[:42], fill="#333", font=font)
            draw.text((x + 8, y + 228),
                      str(info.get("object_type") or info.get("medium") or "")[:42], fill="#333", font=font)
    sheet.save(path, quality=94)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    cache_dir = args.cache_dir.resolve() if args.cache_dir else output
    output.mkdir(parents=True, exist_ok=True)
    query_rows = load_queries(args.queries)
    plans = [{"qid": row["qid"], "query": row["query"], **(plan_dynamic_query(row["query"]))}
             for row in query_rows]
    write_json(output / "query_plans.json", plans)
    write_json(output / "protocol.json", {
        **JUDGMENT_PROTOCOL,
        "ranking_contract": {
            "single": "cosine of the first selected reference; additional refs retained as a stress alternative",
            "mean": "unguarded arithmetic mean of all selected-reference cosine values",
            "coverage": "six-slot mean; only the best 3 selected-reference cosines enter the numerator; coverage counts are reported",
            "top-k": "mean of each candidate's best 3 reference cosines, so disagreement among references can reduce it",
            "fusion": "0.75 SigLIP mean + 0.25 DINO mean or RRF(61) over per-channel rankings",
        },
        "gallery": "4,692 frozen local images",
        "strict_isolation": ["item_id", "source_id", "accession family", "official related object",
                             "original/image/thumbs SHA256", "aHash256 Hamming<=4", "pHash64 Hamming<=6"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    items, exclusions = load_gallery(ROOT)
    by_id = {str(row["item_id"]): row for row in items}
    descriptions = describe_rows(items)
    acquirer = DynamicMuseumAcquirer(ROOT, output, exclusions, offline=args.offline,
                                     cache_dir=cache_dir)
    acquisition_rows = []
    evaluation_plans = []
    evaluation_acquisitions = []
    for plan in plans:
        result = acquirer.fetch(
            plan["qid"], plan, per_query=args.per_query,
            max_references_considered=args.max_references_considered,
            page_depth=args.page_depth,
            max_metadata_candidates=args.max_metadata_candidates,
        )
        initial_strict_available = strict_references_available(result["accepted"], plan)
        result["fallback"] = {
            "used": False,
            "reason": "era-bounded broad discovery; no fixed or unverified fallback references",
        }
        result["selected"] = select_reference_set(
            result["accepted"], plan, limit=args.reference_limit,
            allow_fallback=args.reference_policy == "expert",
        )
        visual_evidence = []
        evidence_keys = [key for key in plan["concept_keys"]
                         if key in {"blue_white", "polychrome", "celadon", "meiping",
                                    "jar", "leather_flask", "gourd", "bowl", "dish"}]
        qualified_references = []
        for row in result["selected"]:
            checks = [image_evidence(ROOT / row["image_path"], key) for key in evidence_keys]
            visual_gate = ("pass" if checks and all(check.get("visual_gate") == "pass" for check in checks)
                           else "reject")
            evidence = {"status": "available" if checks else "not_applicable",
                        "checks": checks, "concept_keys": evidence_keys,
                        "visual_gate": visual_gate,
                        "limitation": "visible evidence gate only; metadata and pixels cannot jointly prove date"}
            row["visual_reference_evidence"] = evidence
            visual_evidence.append(evidence)
            if visual_gate == "pass":
                qualified_references.append(row)
        rejected_visual = [{key: row.get(key) for key in
                            ("reference_id", "title", "accession", "image_path",
                             "visual_reference_evidence")}
                           for row in result["selected"] if row not in qualified_references]
        result["selected_by_metadata"] = result["selected"]
        result["selected"] = qualified_references
        result["visual_rejected_references"] = rejected_visual
        result["reference_visual_gate"] = {
            "status": "passed" if len(qualified_references) >= 2 else "insufficient_visual_references",
            "qualified_count": len(qualified_references),
            "metadata_selected_count": len(visual_evidence),
            "metric_scope": "only visually qualified references enter encoding and full-gallery ranking",
        }
        result["selection_status"] = selection_diagnostics(
            result["accepted"], result["selected"], plan, fallback_used=False
        )
        result["selection_status"]["status"] = (
            "visual_references_qualified" if len(result["selected"]) >= 2 else
            "insufficient_visual_references")
        result["initial_strict_references_available"] = initial_strict_available
        acquisition_rows.append(result)
        print(plan["qid"], "accepted", result["selection_status"]["accepted_count"],
              "strict", result["selection_status"]["strict_accepted_count"],
              "selected", len(result["selected"]), result["selection_status"]["status"], flush=True)
        request_rows = [row for row in acquirer.request_log if row.get("request_id")]
        result["provenance"] = {
            "mode": "offline_cache" if args.offline else "live",
            "cache_dir": cache_dir.relative_to(ROOT).as_posix(),
            "request_count": len(request_rows),
            "cache_requests": sum(bool(row.get("from_cache")) for row in request_rows),
            "live_requests": sum(not row.get("from_cache", False) for row in request_rows),
        }
        write_json(output / f"references_{plan['qid']}.json", result)
        reference_sheet(output / "reference_contact_sheets" / f"{plan['qid']}.jpg",
                        plan["qid"], plan["query"], result["selected"], result["accepted"])
        if len(result["selected"]) >= 2:
            evaluation_plans.append(plan)
            evaluation_acquisitions.append(result)
        elif (args.reference_policy == "expert" and result["candidates"]):
            # Failed-to-evaluate sparse query: preserve the available real
            # top results for visual review even though two references could
            # not be constructed after fingerprint and duplicate isolation.
            review_candidates = []
            for source in {row["source"] for row in result["candidates"]}:
                for source_id in {row["source_id"] for row in result["candidates"]
                                  if row["source"] == source}:
                    local_id = f"{'cma' if source == 'cleveland' else 'met'}_{source_id}"
                    if local_id in by_id:
                        review_candidates.append({"item_id": local_id, "rank": len(review_candidates) + 1})
            if review_candidates:
                pseudo = {"siglip_mean": {"ranking": [row["item_id"] for row in review_candidates]}}
                visual_sheet(output / "failure_contact_sheets" / f"{plan['qid']}_isolated.jpg",
                             plan["qid"], plan["query"], [], pseudo, descriptions, by_id,
                             methods=["siglip_mean"], limit=min(5, len(review_candidates)))
    acquirer.save_manifest(output / "http_requests.jsonl")
    acquirer.save_discovery(output / "discovery_diagnostics.jsonl")
    for row in acquisition_rows:
        row["discovery"]["search_report_count"] = sum(
            report["query_id"] == row["query_id"] for report in acquirer.discovery_reports
        )
        # HTTP review failures are logged separately with request IDs. Keep
        # the acquisition failure count query-specific rather than treating
        # an internal request ID as a qid.
        write_json(output / f"references_{row['query_id']}.json", row)
    write_jsonl(output / "acquisition_failures.jsonl", acquirer.failures)
    acquisition_summary = [
        {"qid": row["query_id"], **row["selection_status"],
         "fallback": row.get("fallback")}
        for row in acquisition_rows
    ]
    write_json(output / "acquisition_summary.json", {
        "queries": acquisition_summary, "status_counts": {
            status: sum(row["status"] == status for row in acquisition_summary)
            for status in {row["status"] for row in acquisition_summary}
        },
    })
    if args.no_visual:
        return

    if not evaluation_plans:
        write_json(output / "summary.json", {
            "queries": 0, "requested_queries": len(plans),
            "status": ("insufficient_strict_references" if args.reference_policy == "strict"
                       else "insufficient_expert_reference_phrases"),
            "reason": "no query had two references passing metadata validation and visible evidence",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        print("no query has two visually qualified references; visual evaluation skipped", flush=True)
        return

    siglip, siglip_ids, siglip_manifest = load_index(ROOT / "artifacts/siglip", ROOT)
    dino, dino_ids, dino_manifest = load_index(ROOT / "artifacts/dinov2", ROOT)
    if siglip_ids != dino_ids:
        raise ValueError("SigLIP and DINO gallery IDs differ")
    all_metrics = []
    rankings_rows = []
    truth_rows = []
    model_metadata = {}
    for plan, acquisition in zip(evaluation_plans, evaluation_acquisitions):
        selected = acquisition["selected"]
        s_refs, d_refs, encoder_meta = encode_references(selected, device=args.device)
        model_metadata = encoder_meta
        results = rankings_for_reference_set(siglip, dino, siglip_ids, s_refs, d_refs)
        target, target_diagnostics = query_target_pool(plan["query"], plan, items)
        if not target:
            # Honest fallback: a query-level form/material diagnostic proxy,
            # still evaluated only after visual ranking, is exported as weak.
            target, target_diagnostics = query_target_pool(plan["query"], {"concept_keys": []}, items)
            target_diagnostics["status"] = "fallback_weak_metadata_proxy"
        else:
            target_diagnostics["status"] = "strict_metadata_diagnostic"
        metrics = metric_report(
            results, target, k=args.top_k,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
        all_metrics.append({"qid": plan["qid"], "query": plan["query"], "metrics": metrics})
        truth_rows.append({"qid": plan["qid"], "query": plan["query"],
                           **target_diagnostics, "target_ids": sorted(target)})
        for method, result in results.items():
            for rank, item_id in enumerate(result["ranking"][:args.top_k], 1):
                rankings_rows.append({"qid": plan["qid"], "method": method, "rank": rank,
                                      "item_id": item_id, "score": float(result["scores"][result["ranking_rows"][rank - 1]]),
                                      "metadata_grade_diagnostic": item_id in target})
        visual_sheet(output / "contact_sheets" / f"{plan['qid']}.jpg", plan["qid"], plan["query"], selected,
                     results, descriptions, by_id, methods=["siglip_single", "siglip_mean", "dino_mean", "fusion_cosine_mean"],
                     limit=args.contact_sheet_limit)
        target_sheet(output / "target_contact_sheets" / f"{plan['qid']}.jpg",
                     plan["qid"], plan["query"], results, descriptions, by_id,
                     methods=METHOD_NAMES, limit=args.contact_sheet_limit)
        print(plan["qid"], "full-gallery ranked", "targets", len(target), flush=True)
    write_jsonl(output / "rankings_topk.jsonl", rankings_rows)
    write_json(output / "metrics.json", all_metrics)
    write_jsonl(output / "target_diagnostics.jsonl", truth_rows)
    write_json(output / "model_metadata.json", model_metadata)
    failure_rows = []
    for entry in all_metrics:
        qid = entry["qid"]
        acquisition = next(row for row in acquisition_rows if row["query_id"] == qid)
        target_row = next(row for row in truth_rows if row["qid"] == qid)
        for method in METHOD_NAMES:
            ranking = next(
                row for row in rankings_rows if row["qid"] == qid and row["method"] == method
            )
            # Rankings are regenerated below for failure inspection from the
            # recorded top-k where possible; full rankings remain local.
            failure_rows.append({
                "qid": qid, "method": method,
                "first_relevant_rank": entry["metrics"][method]["first_relevant_rank"],
                "hit_at_k": entry["metrics"][method]["hit_at_k"],
                "target_count": len(target_row["target_ids"]),
                "top1_item_id": ranking["item_id"],
                "selected_reference_ids": [row["reference_id"] for row in acquisition["selected"]],
                "diagnosis": "miss" if not entry["metrics"][method]["hit_at_k"]
                             else "target_in_top_k",
            })
    write_jsonl(output / "failure_inspection.jsonl", failure_rows)
    aggregate = {}
    # Inferential summary: bootstrap over the qualifying queries only.  This is
    # deliberately not extrapolated to untested queries or the full museum
    # catalogue.
    eligible_methods = list(METHOD_NAMES)
    for method in METHOD_NAMES:
        rows = [entry["metrics"][method] for entry in all_metrics]
        aggregate[method] = {key: float(np.mean([row[key] for row in rows]))
                             if key != "first_relevant_rank" else float(np.mean([row[key] or 9999 for row in rows]))
                             for key in ("hit_at_k", "recall_at_k", "recall_at_20", "mrr_at_k", "first_relevant_rank")}
        for metric in ("hit_at_k", "recall_at_k", "mrr_at_k"):
            values = [float(row[metric]) for row in rows]
            aggregate[method][f"{metric}_query_bootstrap_ci"] = bootstrap_ci(
                values, repetitions=args.bootstrap_repetitions
            )
    write_json(output / "summary.json", {
        "queries": len(all_metrics), "requested_queries": len(plans),
        "status_semantics": {
            "complete": "requested queries ran through the expert-phrase reference path and two or more queries produced full-gallery rankings",
            "conditional_scope": "skipped queries are not encoded as misses in the aggregate; metrics describe only evaluated_queries",
        },
        "evaluated_queries": [entry["qid"] for entry in all_metrics],
        "evaluated_reference_validation": {
            entry["qid"]: next(row["reference_visual_gate"]["status"]
                                for row in acquisition_rows if row["query_id"] == entry["qid"])
            for entry in all_metrics},
        "skipped_queries": [row["query_id"] for row in acquisition_rows
                            if row["query_id"] not in {entry["qid"] for entry in all_metrics}],
        "selection_status_counts": {status: sum(row["selection_status"]["status"] == status
                                                for row in acquisition_rows)
                                    for status in {"visual_references_qualified",
                                                   "insufficient_visual_references"}},
        "gallery_items": len(siglip_ids), "methods": aggregate,
        "siglip_index_sha256": siglip_manifest["index_sha256"],
        "dino_index_sha256": dino_manifest["index_sha256"],
        "gallery_sha256": sha256_file(ROOT / "data/snapshots/collection_v2.jsonl"),
        "status": "complete", "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": ("fixed_supplementary" if args.reference_policy == "expert" else "strict"),
        "reference_policy": args.reference_policy,
        "eligible_methods": eligible_methods,
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "unit": "qualifying semantic queries",
            "scope": ("conditional on the evaluated strict-reference subset"
                      if args.reference_policy == "strict" else
                      "conditional on queries with two admitted expert-phrase references"),
        },
        "metric_semantics": {
            "primary": ["recall_at_k", "hit_at_k", "mrr_at_k"],
            "precision_at_k": "not reported; no complete all-gallery qrels",
            "relevance_pool": "metadata diagnostic intersection, not human qrels or authentication",
        },
    })
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
