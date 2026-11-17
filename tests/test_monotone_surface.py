import numpy as np
from PIL import Image, ImageDraw
import pytest

from curio.evaluate.monotone_evidence import fit_monotone, predict_monotone
from curio.represent.object_surface import extract_surface


def test_stronger_opposition_never_increases_acceptance():
    # Features are positive support and *negated* counter-support.
    x = np.array([[2, 2], [3, 1], [-2, -2], [-3, -1]], dtype=float)
    head = fit_monotone(x, [1, 1, 0, 0])
    scores = predict_monotone(head, [[2, 2], [2, 0], [2, -2]])
    assert scores[0] >= scores[1] >= scores[2]
    assert min(head["coefficient"]) >= 0


def test_inverse_training_correlation_cannot_flip_evidence_direction():
    head = fit_monotone([[0], [1], [2], [3]], [1, 1, 0, 0])
    scores = predict_monotone(head, [[0], [3]])
    assert scores[1] >= scores[0]
    with pytest.raises(ValueError):
        predict_monotone(head, [[float("nan")]])


def test_green_object_on_red_background_does_not_become_red_surface(tmp_path):
    image = Image.new("RGB", (200, 240), (200, 30, 30))
    ImageDraw.Draw(image).ellipse((45, 35, 155, 210), fill=(30, 170, 60))
    path = tmp_path / "object.png"
    image.save(path)
    result = extract_surface(path)
    assert result["valid"]
    assert result["mask"][100, 100] and not result["mask"][5, 5]
    assert sum(result["features"][5:12]) > 0.9  # green hue bins
    assert result["features"][0] + result["features"][23] < 0.05


def test_small_motif_is_normalized_by_whole_object_area(tmp_path):
    image = Image.new("RGB", (200, 240), (40, 40, 40))
    draw = ImageDraw.Draw(image)
    draw.ellipse((35, 35, 165, 215), fill=(230, 230, 230))
    draw.ellipse((90, 90, 110, 110), fill=(210, 20, 20))
    path = tmp_path / "motif.png"
    image.save(path)
    result = extract_surface(path)
    assert result["valid"]
    assert sum(result["features"][:24]) < 0.1
