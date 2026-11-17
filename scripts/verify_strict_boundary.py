"""Exercise the default HTTP contract and save reviewable image contact sheets.

--benchmark replays the existing, previously inspected 140-image test core.
Its numbers are exploratory regression diagnostics, not independent validation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import urllib.request

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "mouth": "窄口的瓶子",
    "green_mouth": "绿色窄口瓷瓶",
    "green": "绿色瓶子",
    "red": "红色瓶子",
    "blue": "蓝色瓶子",
    "negation": "没有龙纹的瓷瓶",
    "historical": "乾隆时期瓷器",
}


def request(base, query):
    req = urllib.request.Request(
        base + "/api/search/set",
        data=json.dumps({"query": query, "top_k": 1}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    assert result["method"] == "strict_conditions"
    assert result["matched_count"] == result["returned"] == len(result["results"])
    assert result["uncertain_count"] == len(result["uncertain_results"])
    assert (
        result["matched_count"] + result["uncertain_count"] + result["rejected_count"]
        == result["gallery_count"]
    )
    assert len(result["preview"]) == min(1, result["matched_count"])
    yes = set(result["matched_item_ids"])
    pending = {row["item_id"] for row in result["uncertain_results"]}
    assert not yes & pending
    for row in result["results"]:
        assert row["condition_evidence"] and all(
            e["status"] == "supported" for e in row["condition_evidence"]
        )
    for row in result["uncertain_results"]:
        assert row["yes"] is None
        assert any(e["status"] == "unknown" for e in row["condition_evidence"])
        assert not any(e["status"] == "contradicted" for e in row["condition_evidence"])
    return result


def sheets(out, name, result, items):
    fonts = [
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    font = next(
        (ImageFont.truetype(str(p), 13) for p in fonts if p.exists()), ImageFont.load_default()
    )
    rows = result["results"]
    for part in range(math.ceil(len(rows) / 40)):
        batch = rows[part * 40 : (part + 1) * 40]
        sheet = Image.new("RGB", (1000, 250 * math.ceil(len(batch) / 5)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(batch):
            x, y = j % 5 * 200, j // 5 * 250
            with Image.open(ROOT / items[row["item_id"]]["image_path"]) as source:
                im = ImageOps.contain(ImageOps.exif_transpose(source).convert("RGB"), (190, 195))
            sheet.paste(im, (x + (200 - im.width) // 2, y))
            measures = [
                f"{e['condition'].split('/')[-1]}={e['measurement']:.2f}"
                for e in row["condition_evidence"]
                if e.get("measurement") is not None
            ]
            draw.text((x + 5, y + 198), f"{row['rank']} {row['item_id']}", font=font, fill="black")
            draw.text((x + 5, y + 216), ", ".join(measures), font=font, fill="black")
            draw.text((x + 5, y + 233), row["title"][:26], font=font, fill="black")
        sheet.save(out / f"after_{name}_{part + 1}.jpg")


def benchmark(base, out):
    dataset = ROOT / "eval/relevance_v5"
    freeze = json.loads((dataset / "freeze.json").read_text())
    for name, digest in freeze["hashes"].items():
        assert hashlib.sha256((dataset / name).read_bytes()).hexdigest() == digest

    def read(name):
        return [json.loads(line) for line in (dataset / name).read_text().splitlines()]

    queries, items, qrels = map(read, ["queries.jsonl", "items.jsonl", "qrels.jsonl"])
    test_ids = {r["item_id"] for r in items if r["split"] == "test"}
    reports = {}
    for lang in ["en", "zh"]:
        records = []
        for q in queries:
            result = request(base, q[f"text_{lang}"])
            (out / f"benchmark_{q['qid']}_{lang}.json").write_text(
                json.dumps(result, ensure_ascii=False)
            )
            yes = set(result["matched_item_ids"])
            labels = [r for r in qrels if r["qid"] == q["qid"] and r["item_id"] in test_ids]
            tp = sum(r["grade"] == 3 and r["item_id"] in yes for r in labels)
            fp = sum(0 <= r["grade"] < 3 and r["item_id"] in yes for r in labels)
            fn = sum(r["grade"] == 3 and r["item_id"] not in yes for r in labels)
            records.append(
                {
                    "qid": q["qid"],
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "unknown_selected": sum(r["grade"] < 0 and r["item_id"] in yes for r in labels),
                    "full_gallery_selected": len(yes),
                    "pending": result["uncertain_count"],
                }
            )
        tp, fp, fn = [sum(r[k] for r in records) for k in ["tp", "fp", "fn"]]
        reports[lang] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "micro_precision": tp / (tp + fp) if tp + fp else None,
            "micro_recall": tp / (tp + fn) if tp + fn else None,
            "macro_f1": sum(
                2 * r["tp"] / (2 * r["tp"] + r["fp"] + r["fn"])
                if 2 * r["tp"] + r["fp"] + r["fn"]
                else 0
                for r in records
            )
            / len(records),
            "per_query": records,
        }
        print(
            "Historical core",
            lang,
            {k: v for k, v in reports[lang].items() if k != "per_query"},
            flush=True,
        )
    return {
        "scope": "Previously inspected 140-image core; 20 paired intents; not independent, not full-gallery accuracy; uncertain is not accepted; grade -1 excluded",
        "results": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    out = ROOT / "artifacts/strict_boundary_v8"
    out.mkdir(parents=True, exist_ok=True)
    items = {
        r["item_id"]: r
        for r in map(
            json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines()
        )
    }
    report = {"method": "strict_conditions", "real_http": True, "checks": [], "source_hashes": {}}
    for name, query in CASES.items():
        result = request(args.url, query)
        (out / f"after_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        if name in {"negation", "historical"}:
            assert result["matched_count"] == 0
        else:
            sheets(out, name, result, items)
        check = {
            "query": query,
            **{
                k: result[k]
                for k in ["matched_count", "uncertain_count", "rejected_count", "latency_ms"]
            },
            "policy": result["boundary"]["strict_verification"],
        }
        report["checks"].append(check)
        print(query, check["matched_count"], "yes", check["uncertain_count"], "pending", flush=True)
    if args.benchmark:
        report["historical_core"] = benchmark(args.url, out)
    for name in [
        "src/curio/retrieve/strict_boundary.py",
        "src/curio/retrieve/condition_parser.py",
        "src/curio/retrieve/condition_boundary_v7.json",
        "src/curio/retrieve/visual_sets.py",
        "eval/relevance_v5/freeze.json",
    ]:
        report["source_hashes"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    (out / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
