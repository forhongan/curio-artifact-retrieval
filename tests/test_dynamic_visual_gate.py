from pathlib import Path
import importlib.util

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dynamic_runner", ROOT / "scripts/run_dynamic_reference_v4.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_meiping_shape_gate_accepts_silhouette_and_rejects_wide_form(tmp_path):
    reference = Image.new("RGB", (300, 440), "#eeeeeb")
    draw = ImageDraw.Draw(reference)
    draw.polygon([(145, 70), (155, 70), (162, 110), (200, 160),
                  (175, 330), (125, 330), (100, 160), (138, 110)], fill="#6a8f82")
    tall = tmp_path / "tall.png"
    reference.save(tall)
    accepted = runner.object_shape_evidence(tall, "meiping")
    assert accepted["status"] == "available"
    assert accepted["value"]["aspect"] >= 1.15
    assert accepted["value"]["shoulder_to_waist"] < 1.5

    wide = Image.new("RGB", (440, 300), "#eeeeeb")
    ImageDraw.Draw(wide).ellipse((40, 50, 400, 250), fill="#6a8f82")
    wide_path = tmp_path / "wide.png"
    wide.save(wide_path)
    rejected = runner.object_shape_evidence(wide_path, "meiping")
    assert rejected["value"]["aspect"] < 1.15


def test_color_gate_has_numeric_visible_evidence(tmp_path):
    image = Image.new("RGB", (320, 320), "#eeeeee")
    ImageDraw.Draw(image).rectangle((60, 60, 260, 260), fill="#253b9b")
    path = tmp_path / "blue.png"
    image.save(path)
    evidence = runner.image_evidence(path, "blue_white")
    assert evidence["status"] == "available"
    assert evidence["value"] > evidence["threshold"]
    assert evidence["visual_gate"] == "pass"
