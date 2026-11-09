"""Turn the root's image/source review into reproducible exhaustive core qrels."""

import csv
import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.evaluate.relevance import judge  # noqa: E402

OUT = ROOT / "eval/relevance_v5"
QUERY_DEFINITIONS = [
    ("porcelain", "porcelain artifacts", "瓷器", [("material", "porcelain")]),
    ("bronze", "bronze artifacts", "青铜器", [("material", "bronze")]),
    ("textile", "textile artifacts and garments", "织物与衣物", [("broad_material", "textile")]),
    ("glass", "glass artifacts", "玻璃器物", [("material", "glass")]),
    ("bottle", "a bottle or vase", "瓶或花瓶", [("form_family", "bottle_vase")]),
    ("bowl", "an open bowl or water bowl", "敞口碗或水钵", [("form_family", "bowl")]),
    ("plate", "a plate dish or saucer", "盘或碟", [("form_family", "plate_dish")]),
    ("people", "an artifact depicting human figures", "描绘人物的文物", [("tag", "figures")]),
    (
        "green_vase",
        "a green bottle or vase",
        "绿色瓶或花瓶",
        [("form_family", "bottle_vase"), ("tag", "green")],
    ),
    (
        "red_vase",
        "a red bottle or vase",
        "红色瓶或花瓶",
        [("form_family", "bottle_vase"), ("tag", "red")],
    ),
    (
        "long_neck",
        "a bottle or vase with a long narrow neck",
        "细长颈瓶",
        [("form_family", "bottle_vase"), ("tag", "long_neck")],
    ),
    ("handles", "a vessel with handles", "带柄或耳的容器", [("vessel", "yes"), ("tag", "handles")]),
    (
        "covered_jar",
        "a round covered jar with its lid",
        "带盖的圆腹罐",
        [("form", "jar"), ("tag", "lidded")],
    ),
    (
        "blue_white",
        "a blue and white decorated ceramic artifact",
        "蓝白纹饰陶瓷器",
        [("broad_material", "ceramic"), ("tag", "blue_white")],
    ),
    ("flowers", "an artifact with flower decoration", "带花卉装饰的文物", [("tag", "floral")]),
    (
        "lotus",
        "an artifact decorated with lotus flowers or lotus leaves",
        "饰有荷花或荷叶纹的文物",
        [("tag", "lotus_motif")],
    ),
    ("dragon", "an artifact depicting a dragon", "描绘龙的文物", [("tag", "dragon")]),
    (
        "pouring",
        "a pouring vessel with a spout and a handle",
        "带流和柄的倾倒器",
        [("form_family", "pouring_vessel"), ("tag", "handles"), ("tag", "spout")],
    ),
    (
        "ming_blue_white",
        "a Ming dynasty blue and white porcelain artifact",
        "明代青花瓷器",
        [("material", "porcelain"), ("era", "ming"), ("blue_white_technique", "yes")],
    ),
    (
        "qianlong",
        "a Qianlong period porcelain artifact",
        "乾隆时期瓷器",
        [("material", "porcelain"), ("era", "Qianlong")],
    ),
]

# Primary body overrides recorded after inspecting original pictures and full official fields.
OVERRIDES = {
    "R028": (None, None),
    "R044": (["bronze"], ["metal"]),
    "R053": (["metal"], ["metal"]),
    "R117": (["jade"], ["stone"]),
    "R121": (["stoneware"], ["ceramic"]),
    "R125": (["brass"], ["metal"]),
    "R129": (["metal"], ["metal"]),
    "R130": (["jade", "coral"], ["stone"]),
    "R132": (["metal"], ["metal"]),
    "R133": (["bronze"], ["metal"]),
    "R136": (None, None),
    "R138": (["wood"], ["wood"]),
    "R141": (["brass"], ["metal"]),
    "R142": (["silk"], ["textile"]),
    "R152": (["porcelain"], ["ceramic"]),
    "R154": (["bronze"], ["metal"]),
    "R157": (["brass"], ["metal"]),
    "R167": (["lacquer"], ["lacquer"]),
    "R181": (["earthenware"], ["ceramic"]),
    "R185": (["copper"], ["metal"]),
}


def write_lines(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def main():
    catalog = {
        r["item_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    items = [json.loads(s) for s in (OUT / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    reviewed = {
        r["review_id"]: r
        for r in csv.DictReader((OUT / "visual_review.tsv").open(encoding="utf-8"), delimiter="\t")
    }
    if len(reviewed) != len(items) or {r["review_id"] for r in items} != set(reviewed):
        raise ValueError("Every core image must have exactly one completed review")
    facts, qrels = [], []
    queries = [
        {
            "qid": f"v5_{key}",
            "text_en": en,
            "text_zh": zh,
            "conditions": atoms,
            "tier": "professional" if key in {"ming_blue_white", "qianlong"} else "visual",
            "acceptable_variants": "FORM_FAMILIES in evaluate/relevance.py; partial never counts as fully relevant",
        }
        for key, en, zh, atoms in QUERY_DEFINITIONS
    ]
    vessel_forms = {
        "vase",
        "bottle",
        "jar",
        "bowl",
        "waterpot",
        "cup",
        "mug",
        "teapot",
        "ewer",
        "cauldron",
        "censer",
        "brushpot",
        "vessel",
        "flask",
    }
    for item in items:
        reviewed_item = reviewed[item["review_id"]]
        tags = reviewed_item["visible_tags"].split(",")
        facet = item["catalogue_facets"]
        # The previous CMA extractor sometimes stopped at its generic type and
        # missed the explicit technique. Use the complete reviewed material text.
        raw_material = (
            item["official_catalog"].get("medium")
            or item["official_catalog"].get("technique")
            or ""
        ).lower()
        parent = {
            "porcelain": "ceramic",
            "earthenware": "ceramic",
            "stoneware": "ceramic",
            "pottery": "ceramic",
            "clay": "ceramic",
            "bronze": "metal",
            "brass": "metal",
            "copper": "metal",
            "silver": "metal",
            "gold": "metal",
            "silk": "textile",
            "textile": "textile",
            "velvet": "textile",
            "weave": "textile",
            "embroidery": "textile",
            "jade": "stone",
            "marble": "stone",
            "steatite": "stone",
            "turquoise": "stone",
            "wood": "wood",
            "bamboo": "wood",
            "sandalwood": "wood",
            "lacquer": "lacquer",
            "glass": "glass",
            "ivory": "ivory",
            "amber": "amber",
            "tortoiseshell": "tortoiseshell",
            "ink": "ink",
            "paper": "paper",
        }
        hits = [
            (m.start(), key)
            for key in parent
            for m in [re.search(r"\b" + key + r"\b", raw_material)]
            if m
        ]
        first = min(hits)[1] if hits else None
        primary = (
            ([first], [parent[first]]) if first else (facet["material"], facet["broad_material"])
        )
        mat, broad = OVERRIDES.get(item["review_id"], primary)
        if mat and len(set(broad or [])) > 1:
            mat, broad = None, None
        era = facet["era"][:]
        raw_era = item["official_catalog"]["era"]
        if "Qianlong" in raw_era and not any(t in raw_era.lower() for t in [" or ", "later", "?"]):
            era = list(set(era + ["Qianlong"]))
        elif "Qianlong" in raw_era:
            era = None
        raw_technique = " ".join(
            str(item["official_catalog"].get(k, "")) for k in ["medium", "technique"]
        ).lower()
        blue_technique = (
            "underglaze cobalt blue" in raw_technique
            or "cobalt blue under" in raw_technique
            or "blue and white" in raw_technique
        )
        # Dated painted-blue ceramics in this core have explicit catalogue wording.
        f = {
            **{
                k: item[k]
                for k in ["review_id", "item_id", "split", "object_group", "image_sha256"]
            },
            "form": [reviewed_item["form"]],
            "visible_tags": tags,
            "material": mat,
            "broad_material": broad,
            "era": era,
            "blue_white_technique": ["yes"] if blue_technique else ["no"],
            "vessel": ["yes"] if reviewed_item["form"] in vessel_forms else ["no"],
            "unknown_tag_families": [k for k in ["color", "motif"] if k + "_unknown" in tags],
            "review_note": reviewed_item["note"],
            "reviewer": "root AI, original contact sheets and complete museum record",
            "source_url": item["official_catalog"]["source_url"],
        }
        source = catalog[item["item_id"]]
        start, end = source.get("year_from"), source.get("year_to")
        if era and "Qianlong" in era:
            period_state = "yes"
        elif re.search(
            r"Kangxi|Yongzheng|Jiaqing|Daoguang|Xianfeng|Tongzhi|Guangxu", raw_era, re.I
        ):
            period_state = "no"
        elif start is not None and end is not None:
            period_state = (
                "no"
                if end < 1736 or start > 1795
                else "yes"
                if 1736 <= start <= end <= 1795
                else "unknown"
            )
        elif era and "qing" not in era:
            period_state = "no"
        else:
            period_state = "unknown"
        f.update(qianlong_period_state=period_state, source_year_interval=[start, end])
        # form-family comparisons use a string; exact fact conditions use the list below.
        facts.append(f)
        for q in queries:
            jf = {**f, "form": reviewed_item["form"]}
            # Exact-form condition uses membership; wrap separately for that atom.
            states = []
            for field, value in q["conditions"]:
                if field == "form":
                    states.append(("exact_form", [value]))
                else:
                    states.append((field, value))
            jf["exact_form"] = [reviewed_item["form"]]
            # Keep conditions as plain field/value pairs in saved qrels.
            resolved = [("exact_form", v[0]) if k == "exact_form" else (k, v) for k, v in states]
            judgment = judge(jf, resolved)
            visual = (
                judge(
                    jf,
                    [
                        a
                        for a in resolved
                        if a[0] not in {"era", "material", "blue_white_technique"}
                    ],
                )
                if q["tier"] == "professional"
                else judgment
            )
            qrels.append(
                {
                    "qid": q["qid"],
                    "item_id": f["item_id"],
                    "split": f["split"],
                    **judgment,
                    "visual_only_status": visual["status"],
                    "evidence_review_id": f["review_id"],
                }
            )
    write_lines(OUT / "facts.jsonl", facts)
    write_lines(OUT / "queries.jsonl", queries)
    write_lines(OUT / "qrels.jsonl", qrels)
    hashes = {
        name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        for name in [
            "items.jsonl",
            "visual_review.tsv",
            "facts.jsonl",
            "queries.jsonl",
            "qrels.jsonl",
        ]
    }
    summary = {
        q["qid"]: {
            sp: {
                status: sum(
                    r["qid"] == q["qid"] and r["split"] == sp and r["status"] == status
                    for r in qrels
                )
                for status in ["relevant", "partial", "irrelevant", "unknown"]
            }
            for sp in ["dev", "test"]
        }
        for q in queries
    }
    (OUT / "freeze.json").write_text(
        json.dumps(
            {
                "hashes": hashes,
                "query_counts": summary,
                "relevance": "All required atoms yes; compatible family variants yes; partial=grade1 and binary negative; unknown=-1 excluded with coverage and precision bounds.",
                "review": "200 root image/source reviews, expanded deterministically into 4000 query-image judgments; not 4000 independent expert annotations.",
                "freeze_stage": "Initial image/source review before new model rankings. v5.1 protocol correction after rankings: broad dates/materials do not establish negative fine labels; empty visual conjunction unknown. No image review retuned to model outputs.",
                "version": "5.1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
