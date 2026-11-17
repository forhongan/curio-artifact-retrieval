"""Learned component evidence; a query is accepted only if every atom passes.

Experimental finite concept set. Missing inference is an error, not a negative
label. Pixel-mask failure is a feature flag, not an automatic all-No fallback.
"""

import math

import numpy as np

from .joint_query import JointQuery, compile_known
from .strict_boundary import pixel_evidence

VERSION = "factor-contrast-v11"
PROBES = {
    "red": (
        "the main object's exterior surface is predominantly red, including coral-red or wine-red; judge the object itself, not its background or small decorative motifs",
        "the main object's exterior is predominantly a non-red color (white, yellow, green, blue, purple, pink, or brown), with at most small red decorative motifs",
    ),
    "bowl": (
        "the main object is an open bowl, with a broad open rim and a concave container body deep enough to hold food; not a flat plate, drinking cup, jar, or vase",
        "the main object is a drinking cup, a flat plate or saucer, a jar, a bottle, a vase, or a non-vessel object, rather than an open food bowl",
    ),
    "meiping": (
        "the main vessel has ALL of these visible shape features: a small mouth, a short neck, broad high shoulders close to the neck, and a body tapering toward a narrower foot (the meiping shape)",
        "the main object has a long tubular neck, or a wide jar opening, or a nearly straight cylindrical body, or is not a bottle or vase; these are alternatives to the short-necked high-shouldered tapering meiping shape",
    ),
    "gourd": (
        "the main object itself is a bottle with two distinct inflated chambers, an upper belly and a lower belly, separated by a clearly constricted waist; both are body chambers, not just a small collar on a long neck",
        "the main object has a single belly with a plain or decorated neck, or only a small raised collar on the neck; it does not have two distinct inflated body chambers joined at a narrow waist",
    ),
    "blue_white": (
        "the main object is a ceramic with visibly blue painted motifs on a white or pale ground, with no prominent added red, green, yellow, or other multicolor painted decoration; blue-and-white ceramic appearance",
        "the main object has prominent multicolored painted decoration or a colored yellow/green ground, or has a plain monochrome surface without blue painted motifs on a pale ground",
    ),
}
QUERY_ATOMS = {
    "red_bowl": ("red", "bowl"),
    "blue_white": ("blue_white",),
    "meiping": ("meiping",),
    "gourd": ("gourd",),
}
QUERY_TEXT = {
    "red_bowl": "a red bowl",
    "blue_white": "blue and white porcelain",
    "meiping": "meiping vase",
    "gourd": "double gourd vase",
}
COLOR_ATOMS = {"red", "blue_white"}
COLOR_FEATURES = ("red", "green", "blue", "white", "black", "yellow", "brown", "pink")


def query_id(text):
    plan = compile_known(text)
    if plan:
        for qid, canonical in QUERY_TEXT.items():
            if plan.key == compile_known(canonical).key:
                return qid
    raise ValueError("Experimental factor model does not cover this query")


def evidence_features(positive, counter, pixels, atom):
    missing_color = positive["reason"] == "color_not_observable"
    if missing_color != (counter["reason"] == "color_not_observable"):
        raise ValueError("Inconsistent color evidence")
    logits = []
    for score in (positive, counter):
        value = 0.0 if missing_color else score["yes_no_logit"]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Invalid component inference")
        logits.append(float(np.clip(value, -30, 30)))
    features = [*logits, logits[0] - logits[1], float(pixels["mask_valid"])]
    if atom in COLOR_ATOMS:
        features += [pixels["color_fractions"].get(c, 0.0) for c in COLOR_FEATURES]
        features += [pixels["chromatic_fraction"]]
    else:
        features += [pixels.get("mouth_ratio") or 0.0, pixels.get("rim_ratio") or 0.0]
    return features


class FactorEvidence:
    def __init__(self, runtime):
        self.runtime = runtime
        self.pixels = {}

    def score(self, image_path, atom):
        if atom not in PROBES:
            raise ValueError("Unsupported visual component")
        path = (self.runtime.root / image_path).resolve()
        if not path.is_relative_to(self.runtime.root) or not path.is_file():
            raise ValueError("Image must be an existing file inside the project")
        if image_path not in self.pixels:
            self.pixels[image_path] = pixel_evidence(path)
        positive, counter = [
            self.runtime.score(
                image_path,
                JointQuery(
                    canonical=text,
                    requires_color=atom in COLOR_ATOMS,
                    clauses=(VERSION + "/" + atom + "/" + str(i),),
                    origin=VERSION,
                ),
            )
            for i, text in enumerate(PROBES[atom])
        ]
        return {
            "features": evidence_features(positive, counter, self.pixels[image_path], atom),
            "positive": positive,
            "counter": counter,
            "color_excluded": positive["reason"] == "color_not_observable",
            "pixels": self.pixels[image_path],
        }


def and_decision(qid, components, thresholds):
    evidence = []
    for atom in QUERY_ATOMS[qid]:
        value = components[atom]
        score = value["score"]
        threshold = thresholds[atom]
        if not math.isfinite(score) or not 0 <= score <= 1 or not 0 < threshold <= 1:
            raise ValueError("Invalid component decision")
        yes = not value.get("color_excluded", False) and score >= threshold
        evidence.append(
            {
                "atom": atom,
                "score": score,
                "threshold": threshold,
                "yes": yes,
                "reason": "color_not_observable"
                if value.get("color_excluded")
                else "supported"
                if yes
                else "insufficient_support",
            }
        )
    return {"yes": all(e["yes"] for e in evidence), "evidence": evidence}
