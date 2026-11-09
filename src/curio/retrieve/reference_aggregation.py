"""Comparable reference aggregation over numeric image embeddings only.

``aggregate_references`` has no model loading, I/O, museum rules or catalogue
fields. Its optional query vector must encode the *complete original query*
using the same SigLIP checkpoint as the reference image vectors. Text assists
reference weighting only: gallery candidates are then scored by image-image
cosines. This hybrid method must not be described as wholly image-only.

For query/reference cosine a_j, temperature T and uniform floor e > 0:

    w_j = (1-e) * softmax(a/T)_j + e / R
    S_i = sum_j w_j * cosine(gallery_image_i, reference_image_j)

The same weights transfer to the paired DINO reference images; DINO is not
compared to text. Every reference keeps positive weight. Weights are numerical
mixture coefficients, not relevance/correctness probabilities. Temperature
and floor are disclosed sensitivity parameters, not learned calibration.

Each aggregation returns separate SigLIP/DINO scores and their tie-aware rank
fusion: b * 61/(60 + rank_siglip) + (1-b) * 61/(60 + rank_dino).
The ``diverse`` control reproduces the existing runtime's complete-link,
first-fit clusters and 0.75 cluster-weighted mean + 0.25 max score. Its clustering
depends on reference order. ``single_best`` is a transparent baseline selection
(highest query cosine, or image medoid without a query), never an oracle.

Outputs contain NumPy arrays, aligned with input gallery/reference rows. Missing
query vectors or empty reference sets are explicit unavailable states; they do
not fabricate all-zero retrieval evidence. Identical references produce equal
soft weights. Empty galleries produce empty score/order arrays. No inputs are
mutated. All arithmetic is float64; very small positive input vectors are safe.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


METHODS = ("single_best", "uniform_mean", "max", "diverse", "query_aligned")


def _unit_rows(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite 2-D matrix with a positive feature dimension")
    if not len(values):
        return values.copy()
    # Scale before the norm to avoid overflow/underflow for finite input data.
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError(f"{name} contains a zero vector")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def _order(scores: np.ndarray, gallery_ids: list[str] | None) -> np.ndarray:
    if gallery_ids is None:
        return np.argsort(-scores, kind="stable")
    return np.lexsort((np.asarray(gallery_ids), -scores))


def _rrf(scores: np.ndarray, gallery_ids: list[str] | None) -> np.ndarray:
    order = _order(scores, gallery_ids)
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return 61 / (60 + ranks)


def _diverse_weights(siglip: np.ndarray, dino: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
    siglip_pairs, dino_pairs = siglip @ siglip.T, dino @ dino.T
    clusters: list[list[int]] = []
    for index in range(len(siglip)):
        for cluster in clusters:
            if all(siglip_pairs[index, other] >= .80 and dino_pairs[index, other] >= .65
                   for other in cluster):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    weights = np.empty(len(siglip), dtype=np.float64)
    for cluster in clusters:
        weights[cluster] = 1 / (len(clusters) * len(cluster))
    return clusters, weights


def aggregate_references(
    siglip_gallery,
    dino_gallery,
    siglip_references,
    dino_references,
    query_vector=None,
    *,
    gallery_ids: Sequence[str] | None = None,
    temperature: float = .07,
    uniform_floor: float = .10,
    siglip_weight: float = .70,
) -> dict:
    """Return all comparative methods, weights, image scores and index orders.

    Gallery counts must match across channels; reference rows must identify
    the same ordered photographs across channels. Channel feature dimensions
    can differ. ``query_vector`` accepts (D,) or (1,D) in SigLIP space.

    ``methods[name]['scores']`` and ``['orders']`` have keys ``siglip``,
    ``dino``, ``fusion``. Each order is an array of input gallery row indices.
    Optional unique string IDs only break equal-score presentation ties.
    ``reference_weights`` is None for max, whose contributing reference varies
    by candidate; ``argmax_reference_indices`` exposes that selection instead.
    Diverse scores include a candidate-dependent max term in addition to their
    disclosed static cluster weights.

    ``single_best`` uses query alignment when available (therefore hybrid),
    otherwise the reference with highest average joint SigLIP/DINO cosine to
    the other references. Ties select the first input row. It does not select
    the reference using target relevance labels or gallery score maxima.
    """
    if not np.isfinite(temperature) or temperature < 1e-12:
        raise ValueError("temperature must be finite and >= 1e-12")
    if not np.isfinite(uniform_floor) or not 0 < uniform_floor <= 1:
        raise ValueError("uniform_floor must be finite and in (0, 1]")
    if not np.isfinite(siglip_weight) or not 0 <= siglip_weight <= 1:
        raise ValueError("siglip_weight must be finite and in [0, 1]")
    sg = _unit_rows(siglip_gallery, "siglip_gallery")
    dg = _unit_rows(dino_gallery, "dino_gallery")
    sr = _unit_rows(siglip_references, "siglip_references")
    dr = _unit_rows(dino_references, "dino_references")
    if len(sg) != len(dg):
        raise ValueError("Gallery channel row counts differ")
    if len(sr) != len(dr):
        raise ValueError("Reference channel row counts differ")
    if sg.shape[1] != sr.shape[1] or dg.shape[1] != dr.shape[1]:
        raise ValueError("Gallery/reference dimensions differ within a channel")
    ids = None if gallery_ids is None else list(gallery_ids)
    if ids is not None and (len(ids) != len(sg) or any(not isinstance(x, str) for x in ids)
                            or len(set(ids)) != len(ids)):
        raise ValueError("gallery_ids must be unique strings parallel to gallery rows")
    query = None
    if query_vector is not None:
        raw_query = np.asarray(query_vector, dtype=np.float64)
        if raw_query.ndim == 1:
            raw_query = raw_query[None, :]
        if raw_query.shape != (1, sr.shape[1]):
            raise ValueError("query_vector must be (D,) or (1,D) in SigLIP reference space")
        query = _unit_rows(raw_query, "query_vector")[0]
    siglip_pairs = np.clip(sg @ sr.T, -1, 1)
    dino_pairs = np.clip(dg @ dr.T, -1, 1)
    result = {
        "version": "numeric-reference-aggregation-v1",
        "status": "no_references" if not len(sr) else "empty_gallery" if not len(sg) else "ok",
        "gallery_count": len(sg), "reference_count": len(sr),
        "gallery_ids": ids,
        "parameters": {"temperature": float(temperature), "uniform_floor": float(uniform_floor),
                       "siglip_fusion_weight": float(siglip_weight), "rrf_constant": 60,
                       "diverse_siglip_cluster_threshold": .80, "diverse_dino_cluster_threshold": .65,
                       "diverse_mean_weight": .75, "diverse_max_weight": .25},
        "pairwise_scores": {"siglip": siglip_pairs, "dino": dino_pairs},
        "references": {"query_cosines": None, "soft_weights": None, "uniform_weights": np.empty(0),
                       "diverse_weights": np.empty(0), "visual_clusters": [],
                       "single_best_index": None, "single_best_selection": "unavailable"},
        "methods": {},
        "limits": [
            "Query-aligned weights are text-image-assisted mixture coefficients, not correctness probabilities.",
            "Candidate scoring after weighting uses image-image similarities; no direct query-gallery term is added.",
            "The original full-query embedding can misalign with dates, specialist wording, language or photographed views.",
            "Soft weighting cannot repair missing relevant references or establish object date/authenticity.",
            "Repeated photographs can accumulate soft weight; upstream source/duplicate isolation remains necessary.",
            "Diverse clustering is an order-dependent baseline; an isolated noisy visual mode can receive excess weight.",
        ],
    }
    if not len(sr):
        result["methods"] = {name: {"status": "unavailable_no_references", "scores": None,
                                     "orders": None, "reference_weights": None}
                             for name in METHODS}
        return result
    uniform = np.full(len(sr), 1 / len(sr))
    clusters, diverse = _diverse_weights(sr, dr)
    soft, query_cosines = None, None
    medoid_scores = (.5 * (sr @ sr.T).mean(axis=1) + .5 * (dr @ dr.T).mean(axis=1))
    best_index, best_selection = int(np.argmax(medoid_scores)), "joint_image_reference_medoid"
    if query is not None:
        query_cosines = np.clip(sr @ query, -1, 1)
        logits = (query_cosines - query_cosines.max()) / temperature
        with np.errstate(under="ignore"):
            exponential = np.exp(logits)
        soft = (1 - uniform_floor) * exponential / exponential.sum() + uniform_floor / len(sr)
        best_index, best_selection = int(np.argmax(query_cosines)), "highest_original_query_reference_siglip_cosine"
    result["references"].update(query_cosines=query_cosines, soft_weights=soft,
                                 uniform_weights=uniform, diverse_weights=diverse,
                                 visual_clusters=clusters, single_best_index=best_index,
                                 single_best_selection=best_selection, medoid_scores=medoid_scores)

    def add_method(name, siglip_scores, dino_scores, weights, formula, text_assisted=False):
        scores = {"siglip": siglip_scores, "dino": dino_scores}
        scores["fusion"] = (siglip_weight * _rrf(siglip_scores, ids)
                            + (1 - siglip_weight) * _rrf(dino_scores, ids))
        result["methods"][name] = {"status": "available", "scores": scores,
            "orders": {channel: _order(values, ids) for channel, values in scores.items()},
            "reference_weights": weights, "formula": formula,
            "text_assisted_reference_selection_or_weighting": text_assisted,
            "candidate_scoring": "image_image_only_after_reference_selection_or_weighting"}

    single_weights = np.eye(1, len(sr), best_index, dtype=np.float64)[0]
    add_method("single_best", siglip_pairs[:, best_index], dino_pairs[:, best_index],
               single_weights, "one reference; selected by disclosed query alignment or image medoid",
               text_assisted=query is not None)
    add_method("uniform_mean", siglip_pairs @ uniform, dino_pairs @ uniform, uniform,
               "arithmetic mean of all reference image cosines")
    add_method("max", siglip_pairs.max(axis=1), dino_pairs.max(axis=1), None,
               "per-candidate maximum image cosine; winning reference may differ by channel")
    result["methods"]["max"]["argmax_reference_indices"] = {
        "siglip": siglip_pairs.argmax(axis=1), "dino": dino_pairs.argmax(axis=1)}
    add_method("diverse", .75 * (siglip_pairs @ diverse) + .25 * siglip_pairs.max(axis=1),
               .75 * (dino_pairs @ diverse) + .25 * dino_pairs.max(axis=1), diverse,
               "0.75 equal-cluster weighted mean + 0.25 per-candidate maximum")
    if soft is None:
        result["methods"]["query_aligned"] = {
            "status": "unavailable_query_vector_missing", "scores": None,
            "orders": None, "reference_weights": None,
            "text_assisted_reference_selection_or_weighting": True}
    else:
        add_method("query_aligned", siglip_pairs @ soft, dino_pairs @ soft, soft,
                   "weighted image cosine; w=(1-floor)*softmax(query_reference_cosine/temperature)+floor/R",
                   text_assisted=True)
    return result
