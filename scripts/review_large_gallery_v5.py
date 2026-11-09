"""Download only a declared top-result pool, retaining failures and provenance."""

import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.retrieve.dynamic_reference_runtime import make_sheet  # noqa: E402


def main():
    out = ROOT / "artifacts/relevance_v5/large_gallery_review"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "research/teammate_review/resources/data/snapshots/collection_10k.jsonl"
    catalog = {
        r["item_id"]: r for r in map(json.loads, source.read_text(encoding="utf-8").splitlines())
    }
    records = json.loads(
        (ROOT / "artifacts/relevance_v5/team_workflow/results.json").read_text(encoding="utf-8")
    )["records"]
    chosen = ["v5_green_vase", "v5_blue_white", "v5_ming_blue_white", "v5_qianlong"]
    rankings = {}
    for r in records:
        if r["gallery"] == 10200 and r["language"] == "en" and r["method"] == "image_only":
            for q in r["per_query"]:
                if q["qid"] in chosen:
                    rankings[r["model"] + "/" + q["qid"]] = q["top10"][:5]
    ids = sorted({key for keys in rankings.values() for key in keys})

    def fetch(key):
        r = dict(catalog[key])
        original = ROOT / r["image_path"]
        path = original if original.is_file() else out / (key + ".jpg")
        try:
            if not path.exists():
                response = requests.get(r["image_url"], timeout=(8, 30))
                response.raise_for_status()
                path.write_bytes(response.content)
            with Image.open(path) as image:
                image.load()
                r["decoded_size"] = list(image.size)
            r.update(
                review_image_path=str(path.relative_to(ROOT)),
                review_image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                status="available",
            )
        except Exception as exc:
            r.update(status="unavailable", failure=str(exc))
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(fetch, ids))
    lookup = {r["item_id"]: r for r in rows}
    for key, keys in rankings.items():
        make_sheet(
            out / (key.replace("/", "_") + ".jpg"),
            key,
            [
                (item, ROOT / lookup[item]["review_image_path"])
                for item in keys
                if lookup[item]["status"] == "available"
            ],
            columns=5,
        )
    (out / "pool.json").write_text(
        json.dumps(
            {
                "selection": "4 declared intents x 2 models x top5, image_only EN, 10200 gallery",
                "rankings": rankings,
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        len(rows),
        "unique pooled images;",
        sum(r["status"] == "available" for r in rows),
        "available",
    )


if __name__ == "__main__":
    main()
