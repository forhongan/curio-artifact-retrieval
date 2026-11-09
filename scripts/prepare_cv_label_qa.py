"""Select an independently relabeled subset without consulting model results."""

import hashlib
import json

from PIL import Image, ImageDraw, ImageOps

from run_cv_research import BASE, ROOT, dump


def main():
    mapping = json.loads((BASE / "blind_map.json").read_text())
    chosen = sorted(
        mapping,
        key=lambda r: hashlib.sha256(
            ("cv-label-agreement-fixed-20261104" + r["alias"]).encode()
        ).hexdigest(),
    )[:24]
    directory = BASE / "annotator_qa"
    directory.mkdir(exist_ok=True)
    dump(directory / "aliases.json", [r["alias"] for r in chosen])
    for start in range(0, len(chosen), 6):
        sheet = Image.new("RGB", (1200, 1220), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(chosen[start : start + 6]):
            with Image.open(ROOT / row["image_path"]) as im:
                tile = ImageOps.contain(ImageOps.exif_transpose(im).convert("RGB"), (560, 355))
            x, y = (slot % 2) * 600, (slot // 2) * 405
            sheet.paste(tile, (x + (600 - tile.width) // 2, y + 35 + (355 - tile.height) // 2))
            draw.text((x + 12, y + 8), row["alias"], fill="black", font_size=22)
        sheet.save(directory / f"sheet_{start // 6 + 1:02d}.jpg", quality=94)
    prompt = (BASE / "annotator_a/TASK.md").read_text()
    prompt = (
        prompt.replace("annotation A", "annotation QA")
        .replace("There are72 images (6 per sheet)", "There are24 images (6 per sheet)")
        .replace("sheet_12.jpg", "sheet_04.jpg")
        .replace("gpt-6.1-terra-a", "gpt-6.1-terra-qa")
    )
    (directory / "TASK.md").write_text(
        prompt
        + "\nThis is an independent annotation repeatability audit. Do not read any previous annotations. Your labels do not replace or tune the frozen evaluation labels.\n",
        encoding="utf8",
    )
    print("24-image independent repeatability subset prepared")


if __name__ == "__main__":
    main()
