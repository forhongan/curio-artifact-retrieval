"""Conservative AND verification for membership, independent of aggregate ranking.

These explicit engineering thresholds are not a calibrated correctness guarantee.
Unresolved or unreadable conditions abstain; they never authorize a yes result.
"""

import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

from curio.represent.visual_parts import _foreground
from curio.retrieve.condition_boundary import CONDITIONS

VERSION = "strict-pixel-and-v4"
POLICY = {
    "mouth_yes_max": 0.45,
    "mouth_no_min": 0.60,
    "main_color_min": 0.50,
    "contrast_yes_min": 0.01,
    "contrast_no_max": -0.01,
}
COLORS = {"red", "green", "blue", "white", "black", "yellow", "brown", "pink", "blue_white"}
MOUTH = "shape/narrow_mouth"
CONDITION_LABELS = {
    MOUTH: "窄口",
    "material/porcelain": "瓷质",
    "material/bronze": "青铜质",
    "material/glass": "玻璃质",
    "broad_material/textile": "织物",
    "broad_material/ceramic": "陶瓷质",
    "form_family/bottle_vase": "瓶形",
    "form_family/bowl": "碗形",
    "form_family/plate_dish": "盘碟形",
    "form_family/pouring_vessel": "倾倒器形",
    "form/jar": "罐形",
    "vessel/yes": "容器",
    "tag/red": "红色主体",
    "tag/green": "绿色主体",
    "tag/blue": "蓝色主体",
    "tag/white": "白色主体",
    "tag/black": "黑色主体",
    "tag/yellow": "黄色主体",
    "tag/brown": "棕色主体",
    "tag/pink": "粉色主体",
    "tag/blue_white": "蓝白纹饰",
    "tag/long_neck": "细长颈",
    "tag/handles": "柄或耳",
    "tag/lidded": "带盖",
    "tag/spout": "流口",
    "tag/figures": "人物纹饰",
    "tag/floral": "花卉纹饰",
    "tag/lotus_motif": "莲花纹饰",
    "tag/dragon": "龙纹",
    "era/Qianlong": "乾隆年代",
    "era/ming": "明代年代",
    "blue_white_technique/yes": "青花工艺",
    "negation": "否定条件",
    "boolean_alternative": "或的组合条件",
    "specialist_form": "专业器型",
    "unparsed_text": "描述中的其他条件",
    "multiple_forms": "器型组合",
    "no_supported_conditions": "查询描述",
    "shape/open_mouth": "敞口",
    "shape/round_body": "圆腹",
}
COLOR_ALTERNATIVES = [
    "red",
    "green",
    "blue",
    "white",
    "black",
    "yellow",
    "brown",
    "pink",
    "purple",
    "orange",
    "gray",
]
FORM_ALTERNATIVES = {
    "form_family/bottle_vase": "a bottle or vase",
    "form_family/bowl": "a bowl or water bowl",
    "form_family/plate_dish": "a plate dish or saucer",
    "form_family/pouring_vessel": "a teapot jug pitcher or ewer with a pouring spout",
    "form/jar": "a storage jar with a short neck or a lid",
}


def pixel_evidence(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=float) / 255
    mask, diagnostic = _foreground(rgb)
    result = {
        "mask_valid": diagnostic["status"] == "valid",
        "mask_reasons": diagnostic["reasons"],
        "mouth_ratio": None,
        "rim_ratio": None,
        "color_fractions": {},
        "chromatic_fraction": 0.0,
    }
    if not result["mask_valid"]:
        return result
    yy, xx = np.where(mask)
    y0, y1, x0, x1 = yy.min(), yy.max() + 1, xx.min(), xx.max() + 1
    crop = mask[y0:y1, x0:x1]
    height, width = crop.shape
    # Use contour span rather than pixel count, so patterns/holes cannot narrow a rim.
    spans = np.array([np.ptp(np.flatnonzero(row)) + 1 if row.any() else 0 for row in crop])
    mouth = spans[max(1, int(height * 0.02)) : max(2, int(height * 0.18))]
    rim = spans[max(1, int(height * 0.02)) : max(2, int(height * 0.06))]
    body = spans[int(height * 0.2) : max(1, int(height * 0.9))]
    center = np.array([np.mean(np.flatnonzero(row)) if row.any() else width / 2 for row in crop])
    if height >= 40 and height / width >= 0.8 and len(mouth) and len(body) and body.max() > 0:
        # Off-axis/cropped/strongly asymmetric outlines are not reliable measurements.
        if abs(float(np.median(center[: max(1, int(height * 0.18))])) - width / 2) <= width * 0.12:
            result["mouth_ratio"] = float(mouth.max() / body.max())
            result["rim_ratio"] = float(rim.max() / body.max()) if len(rim) else None
    # Difference from the background alone can omit pale decoration within the
    # outline. Fill row spans, but abstain if this would invent too much area.
    surface = mask.copy()
    for row in surface:
        indices = np.flatnonzero(row)
        if len(indices):
            row[indices[0] : indices[-1] + 1] = True
    result["surface_fill_fraction"] = float((surface.sum() - mask.sum()) / surface.sum())
    if result["surface_fill_fraction"] > 0.15:
        return result
    interior = ndimage.binary_erosion(surface, iterations=2)
    # Remove the lowest 15% (often a stand/shadow) for surface-color evidence.
    interior[int(y0 + height * 0.85) :] = False
    if interior.sum() < 64:
        return result
    hsv = np.asarray(image.convert("HSV"), dtype=float) / 255
    h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    chromatic = (s >= 0.25) & (v >= 0.15)
    choices = {
        "red": chromatic & ((h <= 20) | (h >= 340)),
        "green": chromatic & (h >= 70) & (h <= 175),
        "blue": chromatic & (h >= 190) & (h <= 260),
        "yellow": chromatic & (h >= 40) & (h <= 70),
        "brown": (s >= 0.18) & (h >= 15) & (h <= 50) & (v >= 0.15) & (v <= 0.70),
        "pink": (s >= 0.12) & (s <= 0.55) & (v >= 0.55) & ((h >= 315) | (h <= 15)),
        "white": (s <= 0.16) & (v >= 0.65),
        "black": v <= 0.20,
    }
    result["chromatic_fraction"] = float(chromatic[interior].mean())
    result["color_fractions"] = {
        key: float(value[interior].mean()) for key, value in choices.items()
    }
    return result


def color_decision(evidence, color):
    fractions = evidence["color_fractions"]
    if not evidence["mask_valid"] or not fractions:
        return "unknown", None
    if color == "blue_white":
        blue, white = fractions["blue"], fractions["white"]
        if evidence["chromatic_fraction"] < 0.02:
            return "unknown", blue
        return ("supported" if blue >= 0.06 and white >= 0.30 else "contradicted"), blue
    fraction = fractions[color]
    if color not in {"white", "black"} and evidence["chromatic_fraction"] < 0.02:
        return "unknown", fraction  # grayscale/near-neutral photo cannot establish real color
    minimum = POLICY["main_color_min"]
    if fraction < minimum:
        return ("contradicted" if fraction < minimum * 0.5 else "unknown"), fraction
    competitors = [fractions[k] for k in COLORS - {"blue_white", color}]
    if fraction < max(competitors):
        return "contradicted", fraction
    return "supported", fraction


def mouth_decision(evidence):
    ratio = evidence["mouth_ratio"]
    if not evidence["mask_valid"] or ratio is None:
        return "unknown", ratio
    if ratio <= POLICY["mouth_yes_max"]:
        return "supported", ratio
    # A short neck can put the shoulder in the upper 18% profile. Only a
    # separately wide top rim supports rejection; otherwise abstain.
    rim = evidence.get("rim_ratio")
    if rim is not None and rim >= POLICY["mouth_no_min"]:
        return "contradicted", ratio
    return "unknown", ratio


class StrictBoundary:
    def __init__(self, service, condition_runtime):
        self.service, self.runtime = service, condition_runtime
        self._pixels = {}
        self._lock = threading.Lock()
        self.cache_dir = Path(service.root) / "artifacts/strict_pixel_cache"

    def _image(self, i):
        item = self.service.items[i]
        # Metadata supplies only image location; titles/facets never enter decisions.
        path = (Path(self.service.root) / item["image_path"]).resolve()
        if not path.is_relative_to(Path(self.service.root).resolve()):
            raise ValueError("Image path outside project")
        stat = path.stat()
        stamp = f"{VERSION}/{path}/{stat.st_size}/{stat.st_mtime_ns}"
        key = hashlib.sha256(stamp.encode()).hexdigest()
        if key not in self._pixels:
            cache = self.cache_dir / f"{key}.json"
            if cache.exists():
                value = json.loads(cache.read_text())
            else:
                value = pixel_evidence(path)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(value))
            self._pixels[key] = value
        return self._pixels[key]

    def apply(self, candidates, diagnostics):
        with self._lock, self.runtime._lock:
            count = len(candidates)
            reject = np.zeros(count, dtype=bool)
            unknown = np.zeros(count, dtype=bool)
            evidence = {int(i): [] for i in np.flatnonzero(candidates)}
            keys = ["/".join(c) for c in diagnostics["conditions"]]
            unresolved = list(diagnostics.get("unverified_modifiers", []))
            if diagnostics["fallback_reasons"] or not keys:
                unresolved += diagnostics["fallback_reasons"] or ["no_supported_conditions"]
            # Unknown Boolean syntax must not be interpreted as positive AND.
            for key in [] if diagnostics["fallback_reasons"] else keys:
                color = key[4:] if key.startswith("tag/") and key[4:] in COLORS else None
                if key == MOUTH:
                    positive = "a photograph of an upright bottle with a small narrow opening and a narrow mouth"
                    negatives = [
                        "a photograph of a vase with a wide open mouth and a flared rim",
                        "a photograph of a jar with a lid covering its mouth",
                        "a photograph of a closed bottle with a stopper",
                    ]
                elif color and color != "blue_white":
                    positive = f"a museum object with a {color} surface"
                    negatives = [
                        f"a museum object with a {other} surface"
                        for other in COLOR_ALTERNATIVES
                        if other != color
                    ]
                    # A white vessel with colored decoration can dominate a
                    # color-word embedding even when its ground is neutral.
                    # Explicitly test that confound for main-surface queries.
                    if color not in {"white", "black"}:
                        negatives += [
                            f"a museum object with a white surface and {color} decoration",
                            f"a museum object with a gray surface and {color} decoration",
                        ]
                elif key in FORM_ALTERNATIVES:
                    positive = "a museum object: " + FORM_ALTERNATIVES[key]
                    negatives = [
                        "a museum object: " + phrase
                        for other, phrase in FORM_ALTERNATIVES.items()
                        if other != key
                    ] + [
                        "a museum object: " + phrase
                        for phrase in [
                            "a cup or mug",
                            "a sculpture",
                            "a textile",
                            "a broken fragment",
                        ]
                    ]
                elif key in CONDITIONS:
                    # English prototypes avoid treating Chinese synonyms as new visual classes.
                    positive = "a museum object: " + CONDITIONS[key][0][0]
                    negatives = ["a museum object: " + v[0] for v in CONDITIONS[key][1:]]
                else:
                    unresolved.append(key)
                    continue
                margin = self.runtime._score(positive) - np.max(
                    np.stack([self.runtime._score(t) for t in negatives]), axis=0
                )
                for i in np.flatnonzero(candidates):
                    state = (
                        "supported"
                        if margin[i] >= POLICY["contrast_yes_min"]
                        else (
                            "contradicted" if margin[i] <= POLICY["contrast_no_max"] else "unknown"
                        )
                    )
                    value = {
                        "condition": key,
                        "label": CONDITION_LABELS.get(key, "查询条件"),
                        "contrast_margin": float(margin[i]),
                        "status": state,
                    }
                    if key == MOUTH or color:
                        try:
                            pixels = self._image(int(i))
                            pixel_state, measurement = (
                                mouth_decision(pixels)
                                if key == MOUTH
                                else color_decision(pixels, color)
                            )
                            value.update(
                                pixel_status=pixel_state,
                                measurement=measurement,
                                mask_valid=pixels["mask_valid"],
                            )
                            if key == MOUTH:
                                value["rim_ratio"] = pixels.get("rim_ratio")
                        except (OSError, ValueError, KeyError):
                            pixel_state = "unknown"
                            value.update(pixel_status="unknown", measurement=None, mask_valid=False)
                        if state == "contradicted" or pixel_state == "contradicted":
                            state = "contradicted"
                        elif state == "unknown" or pixel_state == "unknown":
                            state = "unknown"
                        value["status"] = state
                    reject[i] |= state == "contradicted"
                    unknown[i] |= state == "unknown"
                    evidence[int(i)].append(value)
            if unresolved:
                unknown |= candidates
                for values in evidence.values():
                    values.extend(
                        {
                            "condition": key,
                            "label": CONDITION_LABELS.get(key, "查询条件"),
                            "status": "unknown",
                        }
                        for key in dict.fromkeys(unresolved)
                    )
            pending = candidates & unknown & ~reject
            yes = candidates & ~reject & ~unknown
            report = {
                "version": VERSION,
                "policy": POLICY,
                "candidate_count": int(candidates.sum()),
                "accepted": int(yes.sum()),
                "contradicted": int((candidates & reject).sum()),
                "uncertain": int(pending.sum()),
                "unresolved_conditions": unresolved,
                "semantics": "all explicit checks must support; unknown is never yes; engineering thresholds, not accuracy guarantees",
            }
            return yes, pending, evidence, report
