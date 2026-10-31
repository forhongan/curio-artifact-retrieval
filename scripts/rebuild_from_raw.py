"""Rebuild the validated table from every cached source JSON and image."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_ingest import fetch_one, ensure_dirs, finalize

ROOT = Path(__file__).resolve().parents[1]


def main():
    ensure_dirs()
    rows = {}
    failures = []
    for source in ("met", "cleveland"):
        for p in sorted((ROOT / "data/raw" / source).glob("*.json")):
            try:
                sid = int(p.stem)
            except ValueError:
                continue
            item, fail = fetch_one(source, sid, cached_only=True)
            if item:
                rows[item["item_id"]] = item
            if fail:
                failures.append(fail)
    rows = finalize(list(rows.values()))
    out = ROOT / "data/processed/items.jsonl"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf8"
    )
    tmp.replace(out)
    (ROOT / "data/processed/download_failures.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in failures) + "\n", encoding="utf8"
    )
    print(f"rebuilt rows={len(rows)} failures={len(failures)}")


if __name__ == "__main__":
    main()
