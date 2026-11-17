"""User-authorized catalogue identity plus independently visible conditions.

Catalogue category never supplies a color or a mouth measurement. Missing facts
fall back to visual evidence; unknown query syntax retains the full VQA query.
"""

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.special import expit

from curio.represent.object_surface import extract_surface
from .catalogue_query import (
    VERSION,
    DESCRIPTIONS,
    catalogue_category,
    catalogue_fact,
    compile_catalogue,
)
from .factor_boundary import FactorEvidence
from .joint_pixels import color_observability
from .joint_query import COLOR_KEYS, JointQuery
from .shape_counterevidence import PROBES as COUNTERS, counter_plan
from .strict_boundary import CONDITION_LABELS

LABELS = {
    **CONDITION_LABELS,
    "form/meiping": "梅瓶器类",
    "form/gourd_generic": "葫芦瓶器类",
    "form/gourd_double": "双腹葫芦瓶",
    "form/gourd_triple": "三节葫芦瓶",
}
FAMILY_IMAGE = {
    "bottle_vase": "the main object in the photograph is a stand-alone bottle or vase, rather than a human or animal figurine, painting, lid, stopper or a group of unrelated objects",
    "bowl": "the main object in the photograph is an open concave container with a rim and a visibly deep basin, including bowls with a foot or stem, rather than a flat plate, a lid or a display of unrelated objects",
    "plate_dish": "the main object is a plate, dish or saucer",
    "pouring_vessel": "the main object is a teapot, ewer, jug or pitcher",
    "jar": "the main object is a jar",
}
MATERIAL_IMAGE = {
    "material/porcelain": "the main object is visually consistent with a ceramic or porcelain object",
    "broad_material/ceramic": "the main object is visually consistent with a ceramic, pottery or porcelain object",
    "material/bronze": "the main object is visually consistent with a bronze or metal object",
    "material/glass": "the main object is visually consistent with a glass object",
}


def mouth_body_ratio(mask, valid=True):
    """Conservative outer-rim/body silhouette proxy for upright vessel photos.

    This is an image measurement, not a calibrated physical aperture diameter.
    The semantic checks separately require an actual visible uncovered opening.
    """
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    if not valid or len(xx) < 64:
        return None
    if (
        yy.min() <= 0
        or xx.min() <= 0
        or yy.max() >= mask.shape[0] - 1
        or xx.max() >= mask.shape[1] - 1
    ):
        return None
    crop = mask[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1]
    height, width = crop.shape
    if height < 24 or width < 12:
        return None
    spans, edges = [], []
    for row in crop:
        xs = np.flatnonzero(row)
        spans.append(float(xs[-1] - xs[0] + 1) if len(xs) else 0.0)
        edges.append((float(xs[0]), float(xs[-1])) if len(xs) else (0.0, -1.0))
    rim = float(np.quantile(spans[: max(3, round(height * 0.08))], 0.9))
    axis = float(
        np.median([(a + b) / 2 for a, b in edges[: max(3, round(height * 0.08))] if a <= b])
    )
    # Ignore one-sided shadow/background extensions: an upright vessel body
    # must extend on both sides of the mouth axis. This is conservative for
    # asymmetric or tilted objects, which may need a better photograph.
    body = max(
        (
            max(0.0, 2 * min(axis - a, b - axis) + 1)
            for a, b in edges[round(height * 0.15) : round(height * 0.9)]
            if a <= b
        ),
        default=0.0,
    )
    return rim / body if body >= 12 and rim >= 2 else None


class CatalogueVisualScorer:
    def __init__(self, service, visual):
        self.service, self.visual = service, visual
        self.root = Path(service.root).resolve()
        self.by_path = {r["image_path"]: r for r in service.items if r.get("image_path")}
        self.factor = FactorEvidence(visual)
        self._surfaces = {}
        self._heads = None
        self._support_cache = {}
        self._image_hashes = {}

    @property
    def identity(self):
        fields = ["item_id", "title", "object_name", "medium", "image_path"]
        metadata = [{k: r.get(k) for k in fields} for r in self.service.items]
        paths = [
            Path(__file__).with_name(name)
            for name in (
                "catalogue_visual.py",
                "catalogue_query.py",
                "factor_boundary.py",
                "joint_query.py",
                "joint_pixels.py",
                "shape_counterevidence.py",
            )
        ]
        paths.append(Path(__file__).parents[1] / "represent/object_surface.py")
        return {
            "visual": self.visual.identity,
            "version": VERSION,
            "policy_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest(),
            "catalogue_sha256": hashlib.sha256(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
            "artifacts": {
                str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    self.root / "artifacts/factor_boundary_v11/heads.json",
                    self.root / "artifacts/surface_boundary_v13/red_surface_head.json",
                    self.root / "data/snapshots/collection_v2.jsonl",
                    self.root
                    / "artifacts/targeted_boundary_v10/support_followup/meiping_gallery.jsonl",
                    self.root
                    / "artifacts/targeted_boundary_v10/support_followup/gourd_gallery.jsonl",
                ]
                if p.exists()
            },
        }

    def plan(self, query):
        known = compile_catalogue(query)
        if known:
            return known
        translated = self.visual.plan(query)
        # Translation may expose known bilingual terms; otherwise keep every
        # qualifier in the original/full translated visual question.
        return compile_catalogue(translated.canonical) or translated

    def _visual(self, path, text, *, color=False):
        return self.visual.score(path, JointQuery(text, color, (VERSION,), VERSION))

    def _surface(self, path):
        stat = (self.root / path).stat()
        stamp = (path, stat.st_size, stat.st_mtime_ns)
        policy = hashlib.sha256(
            Path(__file__).read_bytes()
            + (Path(__file__).parents[1] / "represent/object_surface.py").read_bytes()
        ).hexdigest()
        key = hashlib.sha256(json.dumps([policy, stamp]).encode()).hexdigest()
        cache = self.root / "artifacts/catalogue_visual_v15/surfaces" / (key + ".json")
        if stamp not in self._surfaces and cache.exists():
            self._surfaces[stamp] = json.loads(cache.read_text())
        if stamp not in self._surfaces:
            data = extract_surface(self.root / path)
            interior = binary_erosion(data["mask"], iterations=2)
            hsv = cv2.cvtColor(data["rgb"], cv2.COLOR_RGB2HSV).astype(float)
            h, s, v = hsv[..., 0] * 2, hsv[..., 1] / 255, hsv[..., 2] / 255
            chromatic = (s >= 0.25) & (v >= 0.15)
            masks = {
                "green": chromatic & (h >= 70) & (h <= 175),
                "blue": chromatic & (h >= 190) & (h <= 260),
                "yellow": chromatic & (h >= 40) & (h <= 70),
                "white": (s <= 0.16) & (v >= 0.65),
                "black": v <= 0.20,
                "brown": (s >= 0.18) & (h >= 15) & (h <= 50) & (v >= 0.15) & (v <= 0.70),
                "pink": (s >= 0.12) & (s <= 0.55) & (v >= 0.55) & ((h >= 315) | (h <= 15)),
            }
            fractions = {
                k: float(a[interior].mean()) if interior.any() else 0.0 for k, a in masks.items()
            }
            self._surfaces[stamp] = {
                "valid": data["valid"],
                "features": data["features"],
                "fractions": fractions,
                "mouth_body_ratio": mouth_body_ratio(data["mask"], data["valid"]),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix(".tmp")
            temp.write_text(json.dumps(self._surfaces[stamp]))
            temp.replace(cache)
        return self._surfaces[stamp]

    def _color_heads(self):
        if self._heads is None:
            bundle = json.loads(
                (self.root / "artifacts/factor_boundary_v11/heads.json").read_text()
            )
            if bundle["vlm_identity"] != self.visual.identity:
                raise RuntimeError("颜色分项模型与当前视觉模型不一致。")
            red = json.loads(
                (self.root / "artifacts/surface_boundary_v13/red_surface_head.json").read_text()
            )
            self._heads = bundle["heads"], red
        return self._heads

    def _candidate_supported(self, item, clause):
        """Frozen learned image support; explicit museum identities bypass it.

        Missing or incompatible snapshots fall back to direct visual inference.
        This stage is a recall limitation, not proof that a category is absent.
        """
        qid = "gourd" if clause.startswith("form/gourd_") else "meiping"
        if qid not in self._support_cache:
            snapshot = self.root / "data/snapshots/collection_v2.jsonl"
            predictions = (
                self.root / f"artifacts/targeted_boundary_v10/support_followup/{qid}_gallery.jsonl"
            )
            mapping = None
            if snapshot.exists() and predictions.exists():
                original = {
                    r["item_id"]: r for r in map(json.loads, snapshot.read_text().splitlines())
                }
                if len(original) == len(self.service.items) and all(
                    r["item_id"] in original
                    and all(
                        r.get(k) == original[r["item_id"]].get(k)
                        for k in ("image_path", "image_sha256")
                    )
                    for r in self.service.items
                ):
                    rows = list(map(json.loads, predictions.read_text().splitlines()))
                    if (
                        len(rows) == len(original)
                        and {r["item_id"] for r in rows} == set(original)
                        and all(type(r["decision"]) is bool for r in rows)
                    ):
                        mapping = {r["item_id"]: r["decision"] for r in rows}
            self._support_cache[qid] = mapping
        mapping = self._support_cache[qid]
        if mapping is None:
            return True
        path = self.root / item["image_path"]
        stat = path.stat()
        stamp = (str(path), stat.st_size, stat.st_mtime_ns)
        if stamp not in self._image_hashes:
            self._image_hashes[stamp] = hashlib.sha256(path.read_bytes()).hexdigest()
        if self._image_hashes[stamp] != item.get("image_sha256"):
            return True  # A changed photograph cannot reuse the frozen image index.
        return mapping[item["item_id"]]

    def _color(self, path, clause, plan):
        color = clause.split("/", 1)[1]
        if not color_observability(self.root / path)["observable"]:
            return False, 0.0, "color_not_observable"
        if color == "blue_white":
            # For the common plain query this reuses the previous full-gallery
            # whole-query VQA cache; no arbitrary Top-K truncation is applied.
            if "material/porcelain" in plan.clauses:
                raw = self.visual.score(path, compile_catalogue("青花瓷"))
                if raw["score"] < 0.5:
                    return False, raw["score"], "insufficient_blue_white_appearance"
            evidence = self.factor.score(path, "blue_white")
            h = self._color_heads()[0]["blue_white"]
            probability = float(
                expit(
                    ((np.array(evidence["features"]) - h["mean"]) / h["scale"])
                    @ np.array(h["coefficient"])
                    + h["intercept"]
                )
            )
            return (
                probability >= h["threshold"]
                and not evidence["color_excluded"]
                and evidence["positive"]["yes_no_logit"] > 0
                and evidence["counter"]["yes_no_logit"] < 0,
                probability,
                "blue_white_visual_evidence",
            )
        surface = self._surface(path)
        if not surface["valid"]:
            return False, 0.0, "surface_not_observable"
        if color == "red":
            head = self._color_heads()[1]
            pixel_score = float(
                expit(
                    np.sqrt(surface["features"]) @ np.array(head["coefficient"]) + head["intercept"]
                )
            )
            if pixel_score < head["threshold"]:
                return False, pixel_score, "surface_color_mismatch"
            e = self.factor.score(path, "red")
            yes = (
                not e["color_excluded"]
                and e["positive"]["yes_no_logit"] > 0
                and e["counter"]["yes_no_logit"] < 0
            )
            return yes, min(pixel_score, e["positive"]["score"]), "red_surface_and_visual_evidence"
        fraction = surface["fractions"].get(color, 0.0)
        if fraction < 0.5:
            return False, fraction, "surface_color_mismatch"
        evidence = self._visual(path, DESCRIPTIONS[clause], color=True)
        return (
            evidence["score"] >= 0.8,
            min(fraction, evidence["score"]),
            "surface_and_visual_color",
        )

    def score(self, image_path, plan, *, references=()):
        if image_path not in self.by_path:
            raise ValueError("Image is not in this catalogue")
        item = self.by_path[image_path]
        path = (self.root / image_path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Invalid catalogue image path")
        if plan.origin != VERSION:
            raw = self.visual.score(image_path, plan)
            return {
                **raw,
                "condition_evidence": [
                    {
                        "condition": "full_query",
                        "label": "完整查询",
                        "source": "image",
                        "status": "supported" if raw["score"] >= 0.5 else "contradicted",
                        "reason": raw["reason"],
                    }
                ],
            }
        evidence, supported, pending = [], set(), []

        def finish(yes, reason):
            return {
                "score": 1.0 if yes else 0.0,
                "yes_no_logit": 20.0 if yes else -20.0,
                "answer_mass": 1.0,
                "reason": reason,
                "decision_reason": reason,
                "cached": False,
                "condition_evidence": evidence,
            }

        for clause in plan.clauses:
            fact = catalogue_fact(item, clause)
            if fact["status"] != "unknown":
                evidence.append({**fact, "label": LABELS.get(clause, clause)})
                if fact["status"] == "contradicted":
                    return finish(False, "catalogue_conflict")
                supported.add(clause)
            else:
                pending.append(clause)
        # Test visible colors first, before spending inference on remaining shape
        # conditions. Metadata never authorizes a color match.
        pending.sort(key=lambda c: (c not in COLOR_KEYS, c))
        for clause in pending:
            measurement = {}
            if clause in COLOR_KEYS:
                yes, score, reason = self._color(image_path, clause, plan)
            else:
                if (
                    (clause == "form/meiping" or clause.startswith("form/gourd_"))
                    and not (
                        clause.startswith("form/gourd_")
                        and catalogue_fact(item, "form/gourd_generic")["status"] == "supported"
                    )
                    and not self._candidate_supported(item, clause)
                ):
                    evidence.append(
                        {
                            "condition": clause,
                            "label": LABELS.get(clause, clause),
                            "source": "image",
                            "status": "unknown",
                            "reason": "insufficient_visual_index_support",
                        }
                    )
                    return finish(False, "insufficient_visual_index_support")
                raw = self._visual(image_path, DESCRIPTIONS[clause])
                score, reason = raw["score"], "visual_condition"
                yes = score >= 0.8
                if yes and clause.startswith("form/gourd_"):
                    # Only unknown specialist identities need the contradiction
                    # fallback; an explicit museum category uses the broad photo
                    # consistency check below, not the unreliable counter alone.
                    counters = {k: self.visual.score(image_path, counter_plan(k)) for k in COUNTERS}
                    yes = all(s["yes_no_logit"] < 0 for s in counters.values())
                    reason = "gourd_visual_and_counterevidence"
                if yes and clause == "form/gourd_double":
                    triple = self._visual(image_path, DESCRIPTIONS["form/gourd_triple"])
                    yes = triple["score"] < 0.5
                    reason = "exactly_two_body_lobes"
                if yes and clause == "shape/narrow_mouth":
                    opposite = self._visual(
                        image_path,
                        "the actual mouth opening is wide compared with the body, or it is covered by a lid or hidden from view",
                    )
                    yes = opposite["score"] < 0.2
                    reason = "visible_narrow_opening"
                    if yes:
                        ratio = self._surface(image_path)["mouth_body_ratio"]
                        yes = ratio is not None and ratio <= 0.35
                        reason = (
                            "narrow_mouth_ratio"
                            if yes
                            else "mouth_ratio_exceeds_limit"
                            if ratio is not None
                            else "mouth_outline_not_observable"
                        )
                        measurement = {
                            "mouth_body_ratio": ratio,
                            "ratio_limit": 0.35,
                            "value": f"照片口缘/器身宽度约 {ratio:.0%}；严格上限 35%"
                            if ratio is not None
                            else "无法取得可用口部轮廓",
                        }
            if not math.isfinite(score):
                raise RuntimeError("无效图像证据；未生成完整判定。")
            evidence.append(
                {
                    "condition": clause,
                    "label": LABELS.get(clause, clause),
                    "source": "image",
                    "status": "supported" if yes else "contradicted",
                    "score": score,
                    "reason": reason,
                    **measurement,
                }
            )
            if not yes:
                return finish(False, reason)
        # A named gourd flask can still be pictured as a single-body flask.
        # The catalogue identifies the type; the photograph must support the
        # requested body lobes. Do not infer those from a title alone.
        for clause in sorted(c for c in supported if c.startswith("form/gourd_")):
            raw = self._visual(image_path, DESCRIPTIONS[clause])
            yes = raw["score"] >= 0.5
            evidence.append(
                {
                    "condition": "image_gourd_body_consistency",
                    "label": "图片支持葫芦腹部形态",
                    "source": "image",
                    "status": "supported" if yes else "contradicted",
                    "score": raw["score"],
                    "reason": "gourd_body_consistency",
                }
            )
            if not yes:
                return finish(False, "image_gourd_body_conflict")
        category = catalogue_category(item)
        if (
            any(c.startswith("form") for c in supported)
            and category
            and category["family"] in FAMILY_IMAGE
        ):
            raw = self._visual(image_path, FAMILY_IMAGE[category["family"]])
            yes = raw["score"] >= 0.5
            evidence.append(
                {
                    "condition": "image_category_consistency",
                    "label": "图片主体与馆藏器类一致",
                    "source": "image",
                    "status": "supported" if yes else "contradicted",
                    "score": raw["score"],
                    "reason": "main_object_consistency",
                }
            )
            if not yes:
                return finish(False, "image_category_conflict")
        for clause in sorted(supported & MATERIAL_IMAGE.keys()):
            raw = self._visual(image_path, MATERIAL_IMAGE[clause])
            yes = raw["score"] >= 0.5
            evidence.append(
                {
                    "condition": "image_material_consistency",
                    "label": "图片主体与馆藏材质一致",
                    "source": "image",
                    "status": "supported" if yes else "contradicted",
                    "score": raw["score"],
                    "reason": "material_appearance_consistency",
                }
            )
            if not yes:
                return finish(False, "image_material_conflict")
        return finish(True, "catalogue_and_image_supported")

    def close(self):
        # The owning JointSetJobs instance manages the shared visual worker.
        pass
