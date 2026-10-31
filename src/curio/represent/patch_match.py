"""Frozen DINO local matching with an abstaining, untrained foreground heuristic.

Pixels alone do not establish material, chronology, or object identity. A valid
flag means heuristic checks passed, not that foreground segmentation is correct.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from .image_encoder import DINO_ID, DINO_REVISION, DinoEncoder


VERSION = "dino-patch-border-v1"
PARAMETERS = {
    "threshold_floor": .15, "threshold_ceiling": .65, "border_offset": .08,
    "border_mad_multiplier": 3, "max_border_distance_p90": .35,
    "min_foreground_patches": 6, "max_foreground_fraction": .8,
    "max_foreground_border_fraction": .15, "min_component_fraction": .6,
    "min_foreground_contrast": .15, "max_local_tokens": 96,
    "local_weight": .75, "profile_weight": .25,
}


def foreground_patch_features(tokens, grid_shape=None):
    """Build soft foreground weights from normalized tokens and border centroid.

    ImageProcessor's fixed center crop defines the coordinates. This can remove
    parts of the object; diagnostics do not establish complete object coverage.
    """
    tokens = np.asarray(tokens, dtype=np.float32)
    if tokens.ndim != 2 or not len(tokens) or not np.isfinite(tokens).all():
        raise ValueError("tokens must be a nonempty finite matrix")
    norms = np.linalg.norm(tokens, axis=1)
    if (norms < 1e-8).any():
        raise ValueError("tokens must have nonzero norms")
    tokens = tokens / norms[:, None]
    if grid_shape is None:
        side = math.isqrt(len(tokens))
        grid_shape = (side, side)
    height, width = grid_shape
    if height < 3 or width < 3 or height * width != len(tokens):
        raise ValueError("patch grid must match tokens and be at least 3 by 3")
    border = np.zeros((height, width), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    centroid = tokens[border.ravel()].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-8)
    distance = (1 - tokens @ centroid).reshape(height, width)
    edge = distance[border]
    median = float(np.median(edge))
    mad = float(np.median(np.abs(edge - median)))
    threshold = float(np.clip(median + PARAMETERS["border_mad_multiplier"] * mad
                              + PARAMETERS["border_offset"],
                              PARAMETERS["threshold_floor"], PARAMETERS["threshold_ceiling"]))
    raw = distance > threshold
    labels, count = ndimage.label(raw)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    mask = labels == int(np.argmax(sizes)) if count else np.zeros_like(raw)
    dominance = float(mask.sum() / max(1, raw.sum()))
    contrast = float(np.quantile(distance, .9) - median)
    reasons = []
    if float(np.quantile(edge, .9)) > PARAMETERS["max_border_distance_p90"]:
        reasons.append("inconsistent_border_tokens")
    if mask.sum() < PARAMETERS["min_foreground_patches"] or mask.mean() > PARAMETERS["max_foreground_fraction"]:
        reasons.append("foreground_area_out_of_range")
    if raw[border].mean() > PARAMETERS["max_foreground_border_fraction"]:
        reasons.append("foreground_at_border")
    if dominance < PARAMETERS["min_component_fraction"]:
        reasons.append("fragmented_foreground")
    if contrast < PARAMETERS["min_foreground_contrast"]:
        reasons.append("weak_foreground_contrast")
    weights = np.where(mask, np.clip((distance - threshold) / max(float(distance.max() - threshold), 1e-8), 0, 1), 0)
    selected = np.flatnonzero(weights.ravel() > 0)
    selected = selected[np.argsort(-weights.ravel()[selected], kind="stable")[:PARAMETERS["max_local_tokens"]]]
    profile = np.zeros(64, dtype=np.float32)
    log_aspect = 0.0
    if mask.any():
        yy, xx = np.where(mask)
        cropped = weights[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
        resized = np.asarray(Image.fromarray(cropped.astype(np.float32), mode="F").resize((32, 32), Image.Resampling.BILINEAR))
        profile = np.r_[resized.mean(axis=1), resized.mean(axis=0)].astype(np.float32)
        profile /= max(float(np.linalg.norm(profile)), 1e-8)
        log_aspect = math.log(cropped.shape[1] / cropped.shape[0])
    return {
        "tokens": tokens[selected], "weights": weights.ravel()[selected].astype(np.float32),
        "profile": profile, "log_aspect": float(log_aspect),
        "foreground_weights": weights.astype(np.float32),
        "diagnostics": {
            "method": VERSION, "status": "valid" if not reasons else "unknown", "reasons": reasons,
            "foreground_fraction": float(mask.mean()), "selected_patches": len(selected),
            "border_distance_p90": float(np.quantile(edge, .9)), "threshold": threshold,
            "foreground_contrast": contrast, "dominant_component_fraction": dominance,
            "foreground_border_fraction": float(raw[border].mean()),
            "learned_segmentation": False, "complete_object_coverage_verified": False,
        },
    }


class DinoPatchMatcher:
    def __init__(self, encoder=None, device="auto"):
        self.encoder = encoder if encoder is not None else DinoEncoder(device=device)
        metadata = self.encoder.metadata
        if metadata.get("model_id") != DINO_ID or metadata.get("revision") != DINO_REVISION:
            raise ValueError("Patch matcher requires the pinned DINOv2 encoder revision")

    @property
    def metadata(self):
        return {**self.encoder.metadata, "feature": "last_hidden_state_patch_tokens", "method": VERSION,
                "parameters": dict(PARAMETERS), "learned_segmentation": False}

    def encode_images(self, paths, batch_size=16):
        """Accept image paths or PIL images, using an existing encoder when given."""
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        output = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                images = []
                for source in paths[start:start + batch_size]:
                    if isinstance(source, Image.Image):
                        images.append(ImageOps.exif_transpose(source).convert("RGB"))
                    else:
                        with Image.open(Path(source)) as loaded:
                            images.append(ImageOps.exif_transpose(loaded).convert("RGB"))
                inputs = self.encoder.processor(images=images, return_tensors="pt").to(self.encoder.device)
                hidden = self.encoder.model(**inputs).last_hidden_state[:, 1:]
                tokens = hidden.float().cpu().numpy()
                patch = self.encoder.model.config.patch_size
                grid = (inputs["pixel_values"].shape[-2] // patch, inputs["pixel_values"].shape[-1] // patch)
                output.extend(foreground_patch_features(value, grid) for value in tokens)
        return output


def match_patch_features(reference, candidate):
    """Symmetric soft-weighted nearest-patch cosine plus a spatial profile.

    The Chamfer operation has no one-to-one correspondence or geometric proof.
    It can reward repeated motifs and cannot establish exact object identity.
    """
    if any(row["diagnostics"]["status"] != "valid" for row in (reference, candidate)):
        return {"status": "unknown", "local_score": None, "shape_similarity": None, "combined_score": None}
    similarities = reference["tokens"] @ candidate["tokens"].T
    local = .5 * (float(np.average(similarities.max(axis=1), weights=reference["weights"]))
                  + float(np.average(similarities.max(axis=0), weights=candidate["weights"])))
    profile = .5 * (float(reference["profile"] @ candidate["profile"])
                    + math.exp(-abs(reference["log_aspect"] - candidate["log_aspect"])))
    combined = PARAMETERS["local_weight"] * ((local + 1) / 2) + PARAMETERS["profile_weight"] * profile
    return {"status": "available", "local_score": local, "shape_similarity": profile, "combined_score": combined}


def rerank_patch_candidates(reference, candidates, base_scores, item_ids, *, mode="combined_score"):
    """Rerank accepted candidates in their existing slots; unknowns stay in place.

    Call on top100 CLS recall candidates only. Return diagnostic coverage and
    scores; a rejected reference returns the original CLS order explicitly.
    """
    if mode not in {"combined_score", "local_score", "shape_similarity"}:
        raise ValueError("Unknown patch rerank mode")
    scores = np.asarray(base_scores, dtype=float)
    if len(candidates) != len(scores) or len(item_ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Candidates, IDs and finite base scores must have equal lengths")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("Candidate IDs must be unique")
    order = sorted(range(len(scores)), key=lambda idx: (-scores[idx], str(item_ids[idx])))
    details = [match_patch_features(reference, candidate) for candidate in candidates]
    slots = [position for position, idx in enumerate(order) if details[idx]["status"] == "available"]
    accepted = [order[position] for position in slots]
    accepted.sort(key=lambda idx: (-details[idx][mode], -scores[idx], str(item_ids[idx])))
    for position, idx in zip(slots, accepted):
        order[position] = idx
    return {"order": order, "details": details, "candidate_count": len(scores),
            "accepted_candidate_count": len(accepted),
            "coverage": len(accepted) / len(scores) if len(scores) else 0,
            "status": "reranked" if accepted else "fallback_cls",
            "reference_status": reference["diagnostics"]["status"], "method": VERSION, "mode": mode}
