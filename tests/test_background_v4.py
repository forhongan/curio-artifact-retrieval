import numpy as np
from PIL import Image, ImageDraw

from curio.represent.background_v4 import isolate_object


def _iou(left, right):
    return np.logical_and(left, right).sum() / np.logical_or(left, right).sum()


def test_simple_object_is_deterministic_and_returns_valid_mask():
    image = Image.new("RGB", (260, 220), "#eeeeee")
    ImageDraw.Draw(image).ellipse((50, 30, 210, 190), fill="#336699")
    first, second = isolate_object(image), isolate_object(image)
    assert first["status"] == "accepted"
    assert first["mask_binary"].shape == first["score"].shape
    np.testing.assert_array_equal(first["mask_binary"], second["mask_binary"])
    height, width = first["mask_binary"].shape
    scale = width / 260
    truth = np.zeros((height, width), bool)
    yy, xx = np.mgrid[:height, :width]
    truth[((xx - 130 * scale) / (80 * scale)) ** 2 + ((yy - 110 * scale) / (80 * scale)) ** 2 <= 1] = True
    assert _iou(first["mask_binary"], truth) > 0.5


def test_flat_rectangle_uses_rectangular_archetype():
    image = Image.new("RGB", (300, 280), "#eeeeee")
    ImageDraw.Draw(image).rectangle((35, 30, 270, 240), fill="#336699")
    result = isolate_object(image)
    ys, xs = np.where(result["mask_binary"])
    bbox = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    assert result["status"] == "accepted"
    assert 18 <= bbox[0] <= 50 and 235 <= bbox[2] <= 280


def test_tiny_image_abstains_instead_of_inventing_mask():
    result = isolate_object(Image.new("RGB", (8, 8), "white"))
    assert result["status"] == "abstain"
    assert result["reasons"]
