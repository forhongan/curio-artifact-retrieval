import numpy as np
from PIL import Image, ImageDraw

from curio.represent.classical_cv import extract_classical


def test_deterministic_finite_features_and_foreground():
    image = Image.new('RGB', (150, 180), 'white')
    ImageDraw.Draw(image).ellipse((35, 25, 115, 150), fill='blue')
    first = extract_classical(image)
    second = extract_classical(image)
    assert first['diagnostics']['grabcut_accepted']
    assert np.array_equal(first['mask'], second['mask'])
    for key, value in first['features'].items():
        assert np.isfinite(value).all()
        np.testing.assert_array_equal(value, second['features'][key])
        assert np.linalg.norm(value) <= 1.00001


def test_rejected_grabcut_explicitly_falls_back_to_raw():
    image = Image.new('RGB', (10, 10), 'white')
    result = extract_classical(image)
    assert result['diagnostics']['fallback']
    assert result['diagnostics']['reason'] == 'image_too_small'
    np.testing.assert_array_equal(result['features']['raw_hog'], result['features']['gc_hog'])
    np.testing.assert_array_equal(result['features']['raw_color'], result['features']['gc_color'])
    assert not result['features']['gc_shape'].any()
