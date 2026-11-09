import numpy as np
from PIL import Image, ImageDraw

from curio.represent.background_benchmark import (
    FEATURE_CHANNELS,
    METHODS,
    descriptors,
    masks_for_image,
    shape_features,
)


def test_shape_features_encode_a_silhouette_and_handle_empty_mask():
    full = np.ones((100, 100), dtype=bool)
    narrow = np.zeros((100, 100), dtype=bool)
    narrow[5:95, 40:60] = True
    left = shape_features(full)
    right = shape_features(narrow)
    empty = shape_features(np.zeros_like(full))
    assert left.shape == right.shape == empty.shape == (302,)
    assert not np.allclose(left, right)
    assert not empty.any()


def test_all_channel_actually_contains_shape_and_dimensions_are_stable():
    image = np.full((50, 300, 3), 127, np.uint8)
    mask = np.zeros((50, 300), bool)
    mask[10:40, 100:200] = True
    valid = descriptors(image, mask, {"valid": True})
    invalid = descriptors(image, mask, {"valid": False})
    assert set(valid) == set(FEATURE_CHANNELS)
    assert valid["texture"].shape == invalid["texture"].shape
    assert valid["all"].shape[0] == valid["all_without_shape"].shape[0] + 302
    np.testing.assert_array_equal(valid["all"][:302], valid["shape"])
    assert not invalid["shape"].any()


def test_all_six_masks_use_the_same_image_and_report_status(tmp_path):
    image = Image.new("RGB", (120, 100), "white")
    ImageDraw.Draw(image).ellipse((25, 10, 95, 90), fill="blue")
    path = tmp_path / "object.png"
    image.save(path)
    pixels, masks = masks_for_image(path)
    assert set(masks) == set(METHODS)
    assert all(mask.shape == pixels.shape[:2] for mask, _ in masks.values())
    assert masks["no_mask"][0].all()
    assert all("valid" in info and "seconds" in info for _, info in masks.values())
