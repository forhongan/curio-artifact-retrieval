"""Full-gallery reference-image ablation and method-blinded pixel judgment pool.

The intent/reference protocol is frozen before generating any rankings. Grades
are entered separately: no catalogue fields are used as visual relevance truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

from PIL import Image, ImageDraw, ImageOps

from curio.retrieve.engine import SearchService, load_items
from curio.retrieve.image_search import ImageSearchService
from curio.retrieve.references import load_reference_bank


PROTOCOL = [
    ("meiping", "梅瓶", "meiping vase, broad shoulders, small mouth, tapered body",
     ["cma_154732", "met_46140", "met_46515", "met_46527", "met_46462"],
     "Meiping silhouette: small short neck, broad high shoulders, body tapering toward base. Ignore glaze/date."),
    ("gourd", "葫芦瓶", "double gourd vase, two rounded lobes and a narrow waist",
     ["met_46057", "met_45930", "met_40689", "met_46121"],
     "A vessel with two rounded stacked lobes separated by a narrow waist. Ignore color/date/material."),
    ("blue_white", "青花瓷", "blue and white decorated porcelain",
     ["met_49216", "met_48430", "met_53849", "met_46057"],
     "White or pale ceramic surface with clearly blue painted decoration; monochrome images cannot verify blue."),
    ("celadon", "青绿色釉陶瓷", "green celadon glazed ceramic vessel",
     ["cma_103701", "met_51045", "met_46527", "cma_140223"],
     "A ceramic vessel visibly coated in pale gray-green or green glaze. Ignore dynasty; bronze green patina is not this target."),
    ("bronze", "青铜礼器", "bronze ritual vessel",
     ["met_460557", "met_460459", "met_460460", "met_61034"],
     "An archaic-style cast metal ritual vessel with bronze/patinated surface; not a figurine, textile, or ceramic imitation."),
    ("jade", "玉牌", "rectangular carved jade plaque",
     ["met_141624", "met_141625", "met_141627", "met_141628"],
     "A small flat rectangular or square plaque visibly of translucent/opaque stone with jade-like surface. Pixels do not prove material."),
    ("lacquer", "红色雕漆器物", "red carved lacquer object with dense relief ornament",
     ["cma_144690", "cma_663153", "met_60916"],
     "A predominantly cinnabar-red object with dense carved/raised ornament; ordinary smooth red glazed ceramic is not sufficient."),
    ("textile", "有图案的丝织物", "patterned woven silk textile fabric",
     ["cma_101332", "cma_101333", "cma_101334"],
     "A visibly patterned fabric/textile surface or garment; ignore period, and do not infer silk fiber from pixels alone."),
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    protocol = [{"qid": q, "query_zh": zh, "query_en": en, "reference_ids": refs, "visual_target": target}
                for q, zh, en, refs, target in PROTOCOL]
    protocol_bytes = json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()
    write_json(output / "protocol.json", {"sha256": hashlib.sha256(protocol_bytes).hexdigest(), "intents": protocol,
               "predeclared": True, "grade_scale": {"0": "not visually relevant", "1": "partial/uncertain", "2": "visually relevant"},
               "evaluation": "Full-gallery ranking; pooled top5 AI pixel judgments. No full-gallery recall or historic attribution claim."})
    items = load_items("data/snapshots/collection_v2.jsonl")
    service = SearchService(items, index_dir="artifacts/siglip_adapted", encoder_name="siglip", root=".", probe_weight=0)
    image_search = ImageSearchService(service)
    bank = {row["reference_id"]: row for row in load_reference_bank(Path.cwd())}
    records = []
    for intent in protocol:
        qid = intent["qid"]
        paths = [Path(bank[key]["image_path"]) for key in intent["reference_ids"]]
        methods = [("siglip_single", "siglip", "mean", paths[:1]), ("siglip_mean", "siglip", "mean", paths),
                   ("siglip_max", "siglip", "max", paths), ("dinov2", "dinov2", "mean", paths),
                   *[(name, name, "mean", paths) for name in ("shape", "color", "texture", "siglip_shape", "siglip_color", "siglip_texture", "ensemble")]]
        for name, method, aggregation, references in methods:
            started = time.perf_counter()
            try:
                response = image_search.search(references, top_k=10, method=method, aggregation=aggregation)
                record = {"qid": qid, "method": name, "ranking": [r["item_id"] for r in response["results"]],
                          "scores": [r["score"] for r in response["results"]], "warnings": response["warnings"], "status": "ok"}
            except (ValueError, RuntimeError) as exc:
                record = {"qid": qid, "method": name, "ranking": [], "status": "abstained", "reason": str(exc)}
            record["elapsed_ms"] = (time.perf_counter() - started) * 1000
            records.append(record)
            print(qid, name, record["status"], round(record["elapsed_ms"]), flush=True)
        for language in ("zh", "en"):
            response = service.search(intent[f"query_{language}"], top_k=10, variant="image_only")
            records.append({"qid": qid, "method": f"text_{language}", "status": "ok",
                            "ranking": [r["item_id"] for r in response["results"]], "elapsed_ms": response["timings"]["total_ms"]})
        write_json(output / "rankings.json", records)
    write_json(output / "run_manifest.json", {"created_at": datetime.now(timezone.utc).isoformat(),
               "gallery_count": len(items), "gallery_sha256": hashlib.sha256(Path("data/snapshots/collection_v2.jsonl").read_bytes()).hexdigest(),
               "reference_bank_sha256": hashlib.sha256(Path("data/references/bank.jsonl").read_bytes()).hexdigest(),
               "index_sha256": service.index_sha256,
               "judgment_status": "pending", "fusion_weights": "research RRF .75 SigLIP/.25 specialist; fixed before judgments"})
    make_pool(output, items, records)


def make_pool(output, items, records):
    lookup = {str(r["item_id"]): r for r in items}
    aliases = []
    for qi, (qid, _, _, refs, target) in enumerate(PROTOCOL):
        ids = sorted({key for row in records if row["qid"] == qid for key in row["ranking"][:5]})
        random.Random(9017 + qi).shuffle(ids)
        for start in range(0, len(ids), 30):
            batch = ids[start:start + 30]
            sheet = Image.new("RGB", (1200, 80 + ((len(batch) + 5) // 6) * 210), "#eeeeee")
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 8), f"Target {qid}: {target[:130]}", fill="black")
            draw.text((12, 30), "Grade 0=not relevant; 1=partial/uncertain; 2=visually relevant. No catalogue or method shown.", fill="black")
            for n, key in enumerate(batch, start):
                alias = f"{qid}_{n:03d}"
                aliases.append({"qid": qid, "alias": alias, "item_id": key})
                image = Image.open(lookup[key]["image_path"]).convert("RGB")
                thumb = ImageOps.contain(image, (194, 181))
                x, y = (n - start) % 6 * 200, 80 + (n - start) // 6 * 210
                sheet.paste(thumb, (x + (200 - thumb.width) // 2, y))
                draw.text((x + 6, y + 185), alias, fill="black")
            sheet.save(output / f"blind_{qid}_{start//30}.jpg", quality=93)
    write_json(output / "pool_private_mapping.json", aliases)
    write_json(output / "grades_template.json", [{"alias": row["alias"], "grade": None} for row in aliases])
    print("Blinded pool size", len(aliases), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/reference_eval"))
    run(parser.parse_args().output)
