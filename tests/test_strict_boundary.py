"""Regression checks for attribute vetoes and abstention, not accuracy estimates."""

import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pytest

from curio.retrieve.condition_parser import parse_conditions
from curio.retrieve.strict_boundary import (
    StrictBoundary,
    color_decision,
    mouth_decision,
    pixel_evidence,
)


def vessel(path, *, narrow=True, color=(5, 140, 30), background=(245, 245, 245)):
    image = Image.new("RGB", (240, 280), background)
    rim = 18 if narrow else 62
    ImageDraw.Draw(image).polygon(
        [
            (120 - rim, 25),
            (120 + rim, 25),
            (120 + rim, 85),
            (185, 145),
            (180, 235),
            (60, 235),
            (55, 145),
            (120 - rim, 85),
        ],
        fill=color,
    )
    image.save(path)
    return path


def test_pixel_shape_distinguishes_narrow_and_wide_rims(tmp_path):
    narrow = pixel_evidence(vessel(tmp_path / "narrow.png"))
    wide = pixel_evidence(vessel(tmp_path / "wide.png", narrow=False))
    assert mouth_decision(narrow)[0] == "supported"
    assert mouth_decision(wide)[0] == "contradicted"


def test_short_neck_shoulder_width_does_not_certify_a_wide_mouth():
    evidence = {"mask_valid": True, "mouth_ratio": 0.85, "rim_ratio": 0.30}
    assert mouth_decision(evidence)[0] == "unknown"


def test_background_color_cannot_satisfy_surface_color(tmp_path):
    pixels = pixel_evidence(
        vessel(tmp_path / "red.png", color=(190, 5, 5), background=(10, 160, 30))
    )
    assert color_decision(pixels, "red")[0] == "supported"
    assert color_decision(pixels, "green")[0] == "contradicted"


def test_grayscale_and_invalid_masks_abstain(tmp_path):
    gray = pixel_evidence(vessel(tmp_path / "gray.png", color=(90, 90, 90)))
    assert color_decision(gray, "green")[0] == "unknown"
    path = tmp_path / "blank.png"
    Image.new("RGB", (240, 280), "white").save(path)
    blank = pixel_evidence(path)
    assert not blank["mask_valid"]
    assert mouth_decision(blank)[0] == color_decision(blank, "red")[0] == "unknown"


def test_small_colored_decoration_is_not_main_color(tmp_path):
    path = vessel(tmp_path / "decoration.png", color=(215, 215, 215), background=(25, 25, 25))
    image = Image.open(path)
    ImageDraw.Draw(image).rectangle((98, 170, 138, 195), fill=(10, 150, 20))
    image.save(path)
    assert color_decision(pixel_evidence(path), "green")[0] == "contradicted"


class StrongText:
    """All requested text conditions have overwhelming support."""

    def __init__(self, count):
        self.count = count
        self._lock = threading.Lock()

    def _score(self, text):
        positive = text in {
            "a museum object: a bottle or vase",
            "a museum object with a green surface",
            "a photograph of an upright bottle with a small narrow opening and a narrow mouth",
        }
        return np.full(self.count, 0.9 if positive else 0.1)


def test_and_gate_vetoes_even_strong_semantic_matches_and_keeps_unknown_separate(tmp_path):
    paths = [
        vessel(tmp_path / "yes.png"),
        vessel(tmp_path / "wide.png", narrow=False),
        vessel(tmp_path / "wrong_color.png", color=(190, 5, 5)),
        vessel(tmp_path / "gray.png", color=(90, 90, 90)),
    ]
    service = SimpleNamespace(root=tmp_path, items=[{"image_path": p.name} for p in paths])
    boundary = StrictBoundary(service, StrongText(4))
    yes, pending, evidence, report = boundary.apply(
        np.ones(4, bool), parse_conditions("绿色窄口瓶子")
    )
    assert yes.tolist() == [True, False, False, False]
    assert pending.tolist() == [False, False, False, True]
    assert report["contradicted"] == 2
    assert any(e["status"] == "contradicted" for e in evidence[1])
    assert all(e["status"] == "supported" for e in evidence[0])
    # Changes to catalogue descriptions cannot affect membership.
    for item in service.items:
        item.update(title="green narrow-mouth vase", material="porcelain")
    again = boundary.apply(np.ones(4, bool), parse_conditions("绿色窄口瓶子"))
    np.testing.assert_array_equal(yes, again[0])


@pytest.mark.parametrize(
    "query", ["a green vase without handles", "乾隆瓷瓶", "窄口梅瓶", "vase with elephants"]
)
def test_unverified_clauses_never_get_a_yes(tmp_path, query):
    path = vessel(tmp_path / "vase.png")
    boundary = StrictBoundary(
        SimpleNamespace(root=tmp_path, items=[{"image_path": path.name}]), StrongText(1)
    )
    yes, _, _, report = boundary.apply(np.ones(1, bool), parse_conditions(query))
    assert not yes.any()
    assert report["unresolved_conditions"]


def test_missing_image_abstains_instead_of_allowing_text_only_yes(tmp_path):
    boundary = StrictBoundary(SimpleNamespace(root=tmp_path, items=[{}]), StrongText(1))
    yes, pending, _, _ = boundary.apply(np.ones(1, bool), parse_conditions("green vase"))
    assert not yes.any()
    assert pending.all()


@pytest.mark.parametrize(
    "color,word",
    [
        ("blue", "蓝色"),
        ("white", "白色"),
        ("black", "黑色"),
        ("yellow", "黄色"),
        ("brown", "棕色"),
        ("pink", "粉色"),
    ],
)
def test_additional_colors_are_explicit_conditions(color, word):
    for query in [f"{color} vase", f"{word}瓶子"]:
        parsed = parse_conditions(query)
        assert ["tag", color] in parsed["conditions"]
        assert not parsed["fallback_reasons"]
