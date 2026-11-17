"""Prepare Chinese failure review and a frozen, unlabeled next-round sample.

No model training, label inference, or test-set tuning occurs in this script.
All pending labels remain blank. Sampling uses IDs/groups, never model scores.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/set_boundary_followup"
QUERIES = [
    ("v8_red_bowl", "dev", "a red bowl", "红色碗"),
    ("v8_green_plate", "dev", "a green plate", "绿色盘"),
    ("v8_glass_bottle", "dev", "a glass bottle", "玻璃瓶"),
    ("v8_bronze_animal", "dev", "a bronze animal sculpture", "青铜动物雕塑"),
    ("v8_floral_bowl", "dev", "a bowl with flower decoration", "带花卉纹饰的碗"),
    ("v8_lidded_porcelain_jar", "dev", "a porcelain jar with a lid", "带盖瓷罐"),
    ("v8_green_bowl", "test", "a green bowl", "绿色碗"),
    ("v8_red_plate", "test", "a red plate", "红色盘"),
    ("v8_porcelain_cup", "test", "a porcelain cup", "瓷杯"),
    ("v8_blue_white_vase", "test", "a blue and white decorated vase", "带蓝白纹饰的瓶"),
    ("v8_dragon_textile", "test", "a textile depicting a dragon", "描绘龙的织物"),
    ("v8_bronze_handles", "test", "a bronze vessel with handles", "带柄或耳的青铜容器"),
]


def readl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def token(value):
    return hashlib.sha256(("boundary-v8-20261115/" + value).encode()).hexdigest()


def sheet(rows, filename, title):
    width, height = 300, 300
    canvas = Image.new("RGB", (width * 4, 54 + height * ((len(rows) + 3) // 4)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for path in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).is_file():
            font = ImageFont.truetype(path, 16)
            break
    draw.text((15, 15), title, font=font, fill="black")
    for n, row in enumerate(rows):
        x, y = n % 4 * width, 54 + n // 4 * height
        with Image.open(ROOT / row["image_path"]) as original:
            pic = ImageOps.contain(original.convert("RGB"), (280, 235))
        canvas.paste(pic, (x + (width - pic.width) // 2, y))
        draw.text((x + 8, y + 239), row["item_id"], font=font, fill="black")
        draw.text((x + 8, y + 261), row.get("caption", "UNLABELED"), font=font, fill="black")
    canvas.save(filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pending = OUT / "independent_labels_pending.csv"
    if pending.exists():
        with pending.open(encoding="utf-8-sig", newline="") as stream:
            if any(
                row.get("human_grade", "").strip() or row.get("notes", "").strip()
                for row in csv.DictReader(stream)
            ):
                raise ValueError("Refusing to overwrite existing human annotations")
    core = readl(ROOT / "eval/relevance_v5/items.jsonl")
    queries = readl(ROOT / "eval/relevance_v5/queries.jsonl")
    gallery = readl(ROOT / "data/snapshots/collection_v2.jsonl")
    by_id = {r["item_id"]: r for r in gallery}
    ids = json.loads((ROOT / "artifacts/siglip/item_ids.json").read_text())
    index = {item: i for i, item in enumerate(ids)}
    grades = {
        (r["qid"], r["item_id"]): r["grade"] for r in readl(ROOT / "eval/relevance_v5/qrels.jsonl")
    }
    models = json.loads((ROOT / "artifacts/set_boundary_v7/models.json").read_text())["models"]
    scores = np.load(ROOT / "artifacts/set_boundary_v7/scores.npz")
    result = json.loads((ROOT / "artifacts/set_boundary_v7/results.json").read_text())["records"]
    base = {r["qid"]: r for r in result["refit_base/zh"]["per_query"]}
    worse = sorted(
        result["conditions/zh"]["per_query"],
        key=lambda r: (r["set_f1"] - base[r["qid"]]["set_f1"], r["qid"]),
    )[:5]
    failures = []
    threshold = models["conditions/zh"]["threshold"]
    base_threshold = models["refit_base/zh"]["threshold"]
    for row in worse:
        j = next(j for j, q in enumerate(queries) if q["qid"] == row["qid"])
        rows = []
        for item in core:
            if item["split"] != "test":
                continue
            grade = grades[row["qid"], item["item_id"]]
            if grade < 0:
                continue
            i = index[item["item_id"]]
            score = float(scores["conditions/zh"][i, j])
            previous = float(scores["refit_base/zh"][i, j])
            predicted = score >= threshold
            if predicted == (grade == 3):
                continue
            rows.append(
                {
                    "qid": row["qid"],
                    "query_zh": queries[j]["text_zh"],
                    "item_id": item["item_id"],
                    "error": "FN" if grade == 3 else "FP",
                    "frozen_grade": grade,
                    "score": score,
                    "threshold": threshold,
                    "baseline_score": previous,
                    "baseline_correct": (previous >= base_threshold) == (grade == 3),
                    "image_path": by_id[item["item_id"]]["image_path"],
                    "source_url": by_id[item["item_id"]].get("source_url"),
                    "caption": f"{'FN' if grade == 3 else 'FP'} grade={grade}; score={score:.3f}",
                }
            )
        rows.sort(
            key=lambda r: (
                not r["baseline_correct"],
                r["error"] != "FN",
                -abs(r["score"] - threshold),
                r["item_id"],
            )
        )
        failures.extend(rows)
        sheet(
            rows[:12],
            OUT / (row["qid"] + ".png"),
            row["qid"] + " | frozen test failure diagnosis (ZH)",
        )
    dump(OUT / "chinese_failures.json", failures)

    excluded_groups = {by_id[r["item_id"]]["object_group"] for r in core}
    excluded_hashes = {by_id[r["item_id"]]["image_sha256"] for r in core}
    selected, used_hashes = [], set(excluded_hashes)
    for split, count in [("dev", 40), ("test", 80)]:
        groups = {}
        for row in gallery:
            if (
                row["split"] == split
                and row["object_group"] not in excluded_groups
                and row["image_sha256"] not in used_hashes
            ):
                groups.setdefault(row["object_group"], []).append(row)
        for group in sorted(groups, key=token):
            candidates = [r for r in groups[group] if r["image_sha256"] not in used_hashes]
            if not candidates:
                continue
            item = min(candidates, key=lambda r: token(r["item_id"]))
            selected.append(
                {
                    k: item[k]
                    for k in [
                        "item_id",
                        "object_group",
                        "split",
                        "image_path",
                        "image_sha256",
                        "source_url",
                    ]
                }
            )
            used_hashes.add(item["image_sha256"])
            if sum(r["split"] == split for r in selected) == count:
                break
        if sum(r["split"] == split for r in selected) != count:
            raise ValueError("Insufficient unused independent groups")
    assert not {r["object_group"] for r in selected if r["split"] == "dev"} & {
        r["object_group"] for r in selected if r["split"] == "test"
    }
    new_queries = [dict(zip(["qid", "split", "text_en", "text_zh"], q)) for q in QUERIES]
    dump(OUT / "new_queries.json", new_queries)
    dump(OUT / "new_items.json", selected)
    for split in ["dev", "test"]:
        subset = [r for r in selected if r["split"] == split]
        for page in range(0, len(subset), 20):
            sheet(
                subset[page : page + 20],
                OUT / f"{split}_{page // 20 + 1:02}.png",
                f"{split.upper()} independent sample | UNLABELED | page {page // 20 + 1}",
            )
    fields = [
        "qid",
        "split",
        "text_zh",
        "text_en",
        "item_id",
        "image_path",
        "source_url",
        "human_grade",
        "notes",
    ]
    with (OUT / "independent_labels_pending.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for q in new_queries:
            for item in selected:
                if q["split"] == item["split"]:
                    writer.writerow(
                        {**q, **{k: item[k] for k in ["item_id", "image_path", "source_url"]}}
                    )
    dump(
        OUT / "manifest.json",
        {
            "version": "boundary-followup-v1",
            "status": "unlabeled; not an evaluation result",
            "sampling": "SHA256 order by object group; no score-based selection; one image per group",
            "new_dev_images": 40,
            "new_test_images": 80,
            "new_intents": 12,
            "pending_judgments": 720,
            "excluded_reviewed_groups": len(excluded_groups),
            "group_overlap": 0,
            "hashes": {
                name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                for name in ["new_queries.json", "new_items.json", "independent_labels_pending.csv"]
            },
            "limitations": [
                "New combinations of known concepts, not entirely unseen concepts.",
                "Source split is preserved; finite sample may contain few/no positives.",
                "Sample recall is not full-gallery recall. Labels require independent human review.",
            ],
        },
    )
    print(OUT, "failures:", len(failures), "new images:", len(selected), "pending judgments:", 720)


if __name__ == "__main__":
    main()
