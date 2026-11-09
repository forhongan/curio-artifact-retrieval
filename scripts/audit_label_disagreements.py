"""Read-only audit of frozen labels against official records and prompt evidence."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/cv_research_20261104"
OUT = ROOT / "artifacts/label_disagreements_20261108"
ALIASES = ["V036", "D037", "V052", "V012"]
URLS = {
    "cma_170279": "https://openaccess-api.clevelandart.org/api/artworks/170279",
    "met_47660": "https://collectionapi.metmuseum.org/public/collection/v1/objects/47660",
    "cma_125743": "https://openaccess-api.clevelandart.org/api/artworks/125743",
    "cma_101807": "https://openaccess-api.clevelandart.org/api/artworks/101807",
}


def fetch(pair):
    key, url = pair
    record = {"item_id": key, "url": url, "fetched_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=35)
        target = OUT / f"{key}_official_response.txt"
        target.write_bytes(response.content)
        record.update(
            status=response.status_code, sha256=hashlib.sha256(response.content).hexdigest()
        )
        if response.ok:
            data = response.json()
            if key.startswith("cma"):
                data = data["data"]
            (OUT / f"{key}_official.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8"
            )
            record["title"] = data.get("title")
    except (requests.RequestException, ValueError, KeyError) as error:
        record["error"] = str(error)
    return record


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    mapping = {
        r["alias"]: r for r in json.loads((BASE / "blind_map.json").read_text(encoding="utf8"))
    }
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    sheet = Image.new("RGB", (1200, 1040), "white")
    draw = ImageDraw.Draw(sheet)
    for i, alias in enumerate(ALIASES):
        row = mapping[alias]
        path = ROOT / row["image_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["image_sha256"]
        with Image.open(path) as im:
            tile = ImageOps.contain(im.convert("RGB"), (570, 455))
        x, y = i % 2 * 600, i // 2 * 520
        sheet.paste(tile, (x + (600 - tile.width) // 2, y + 55 + (455 - tile.height) // 2))
        draw.text((x + 20, y + 15), alias + " / " + row["item_id"], font=font, fill="black")
    sheet.save(OUT / "four_disagreements.jpg", quality=94)
    records = list(ThreadPoolExecutor(max_workers=4).map(fetch, URLS.items()))
    (OUT / "official_fetch_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf8"
    )
    parts = [
        "# Reconstructed actual annotation prompts\n\nReconstructed from preserved TASK.md and the literal suffix in scripts/run_terra_cv_labels.py. The suffix forbids tools; docs/VISUAL_LABELING.md was referenced but not inlined. Logs contain no file-reading tool calls. Do not imply that this guideline was actually read.\n"
    ]
    for person in ("a", "b", "qa"):
        directory = BASE / f"annotator_{person}"
        aliases = json.loads((directory / "aliases.json").read_text())
        sheets = sorted(directory.glob("sheet_*.jpg"))
        prompt = (directory / "TASK.md").read_text(encoding="utf8")
        prompt += f"\nAll {len(sheets)} sheets are attached in numbered order, with aliases above the images. Work only from these images, without using tools or reading any files. Return all {len(aliases)} labels as the schema JSON. Do not write files: the calling process saves your response.\n"
        (OUT / f"actual_prompt_{person}.txt").write_text(prompt, encoding="utf8")
        parts.append(f"\n## {person}\n\n```text\n{prompt}```\n")
    (OUT / "ACTUAL_PROMPTS.md").write_text("".join(parts), encoding="utf8")
    print(json.dumps(records, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
