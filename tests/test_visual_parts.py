import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw
import pytest

from curio.represent.visual_parts import (
    FEATURE_DIMENSIONS, extract_visual_parts, load_visual_parts,
)


def studio(color="navy", shift=0, scale=1):
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((128 - 40 * scale + shift, 128 - 85 * scale,
                  128 + 40 * scale + shift, 128 + 85 * scale), fill=color)
    return image


def test_mask_accuracy_and_abstention():
    image = studio()
    result = extract_visual_parts(image)
    truth = np.asarray(image)[..., 0] < 100
    predicted = result["mask"] > 0
    assert (truth & predicted).sum() / (truth | predicted).sum() > 0.98
    assert all(result["valid"].values())
    for feature, dimension in FEATURE_DIMENSIONS.items():
        assert result[feature].shape == (dimension,)
        assert np.linalg.norm(result[feature]) == pytest.approx(1)
    blank = extract_visual_parts(Image.new("RGB", (200, 200), "white"))
    assert not any(blank["valid"].values())
    assert "foreground_area_out_of_range" in blank["diagnostics"]["reasons"]
    random = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    scene = extract_visual_parts(Image.fromarray(random))
    assert not any(scene["valid"].values())
    assert "complex_or_inconsistent_border" in scene["diagnostics"]["reasons"]


def test_shape_is_translation_scale_stable_and_color_distinguishes():
    source = extract_visual_parts(studio())
    moved = extract_visual_parts(studio(shift=30, scale=0.7))
    recolored = extract_visual_parts(studio(color="red"))
    assert source["shape"] @ moved["shape"] > 0.995
    assert source["shape"] @ recolored["shape"] > 0.999
    assert source["color"] @ recolored["color"] < 0.05


def test_multiple_objects_and_cropped_foregrounds_are_unknown():
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 50, 90, 190), fill="black")
    draw.rectangle((160, 50, 230, 190), fill="black")
    result = extract_visual_parts(image)
    assert not result["valid"]["shape"]
    assert "multiple_or_fragmented_foregrounds" in result["diagnostics"]["reasons"]
    image = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(image).rectangle((0, 50, 180, 190), fill="black")
    assert not extract_visual_parts(image)["valid"]["shape"]


def test_texture_responds_to_local_patterns_and_repeats_exactly():
    smooth = studio(color="gray")
    patterned = smooth.copy()
    pixels = np.asarray(patterned).copy()
    foreground = np.asarray(smooth)[..., 0] < 200
    yy, xx = np.indices(foreground.shape)
    checker = ((xx // 3 + yy // 3) % 2).astype(bool)
    pixels[foreground & checker] = 40
    pixels[foreground & ~checker] = 180
    first = extract_visual_parts(Image.fromarray(pixels))
    repeated = extract_visual_parts(Image.fromarray(pixels))
    baseline = extract_visual_parts(smooth)
    assert first["valid"]["texture"]
    assert first["texture"] @ baseline["texture"] < 0.9
    assert np.array_equal(first["texture"], repeated["texture"])
    assert np.array_equal(first["mask"], repeated["mask"])


def test_builder_provenance_and_load_rejects_tampering(tmp_path):
    # Import by file path because scripts is not installed as a package.
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("build_visual_parts", Path(__file__).parents[1] / "scripts/build_visual_parts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for index, color in enumerate(("navy", "red")):
        path = tmp_path / f"{index}.png"
        studio(color).save(path)
        rows.append({"item_id": str(index), "image_path": path.name,
                     "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = tmp_path / "catalog.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in rows))
    manifest = module.build_visual_parts(data, "parts", root=tmp_path, workers=2)
    assert manifest["items"] == 2
    assert manifest["valid_counts"] == {"shape": 2, "color": 2, "texture": 2}
    assert not manifest["trained_segmentation"]
    loaded = load_visual_parts(tmp_path / "parts")
    assert loaded["item_ids"] == ["0", "1"]
    assert loaded["valid_shape"].all()
    (tmp_path / "parts/item_ids.json").write_text('["changed"]')
    with pytest.raises(ValueError, match="hash mismatch"):
        load_visual_parts(tmp_path / "parts")
    studio("green").save(tmp_path / "0.png")
    with pytest.raises(ValueError, match="Frozen processed image changed"):
        module.build_visual_parts(data, "bad", root=tmp_path, workers=1)
