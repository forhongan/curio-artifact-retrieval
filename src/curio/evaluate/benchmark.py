"""Transparent metadata-derived silver relevance set, never called human gold."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

from curio.represent.train import read_items

ZH_TYPES = {
    "vase": "瓶",
    "bowl": "碗",
    "plate": "盘",
    "dish": "碟",
    "jar": "罐",
    "cup": "杯",
    "figure": "人物雕像",
    "figurine": "小雕像",
    "sculpture": "雕塑",
    "box": "盒",
    "bottle": "瓶",
    "ewer": "执壶",
    "teapot": "茶壶",
    "pot": "壶",
    "tripod": "三足器",
    "incense burner": "香炉",
    "censer": "香炉",
    "mirror": "镜",
    "bell": "钟",
    "pendant": "佩饰",
    "stele": "石碑",
    "ornament": "饰件",
    "textile": "织物",
    "robe": "袍",
    "brush holder": "笔筒",
    "jue": "爵",
    "gu": "觚",
    "ding": "鼎",
    "bi": "璧",
    "cong": "琮",
    "container": "容器",
    "seal": "印",
    "tile": "砖",
    "bead": "珠",
}
ZH_MATERIALS = {
    "porcelain": "瓷",
    "bronze": "青铜",
    "jade": "玉",
    "silk": "丝绸",
    "textile": "纺织品",
    "ceramic": "陶瓷",
    "ceramics": "陶瓷",
    "earthenware": "陶",
    "stoneware": "炻器",
    "stone": "石",
    "wood": "木",
    "lacquer": "漆",
    "glass": "玻璃",
    "silver": "银",
    "gold": "金",
    "iron": "铁",
    "copper": "铜",
    "ivory": "象牙",
    "metal": "金属",
    "clay": "陶土",
}
ZH_MOTIFS = {
    "dragon": "龙",
    "phoenix": "凤",
    "lotus": "莲花",
    "flower": "花卉",
    "flowers": "花卉",
    "bird": "鸟",
    "birds": "鸟",
    "fish": "鱼",
    "cloud": "云",
    "clouds": "云",
    "animal": "动物",
    "lion": "狮",
    "horse": "马",
    "butterfly": "蝴蝶",
    "bamboo": "竹",
    "peony": "牡丹",
    "landscape": "山水",
    "geometric": "几何",
    "taotie": "饕餮",
}
ERAS = {
    "Han": (-206, 220, "汉代"),
    "Tang": (618, 907, "唐代"),
    "Song": (960, 1279, "宋代"),
    "Yuan": (1271, 1368, "元代"),
    "Ming": (1368, 1644, "明代"),
    "Qing": (1644, 1912, "清代"),
}


def attributes(item):
    return {
        "object_type": {str(item.get("object_type", "")).lower()},
        "material": {str(item.get("material", "")).lower()},
        "motifs": {str(x).lower() for x in item.get("motifs", [])},
    }


def grade(item, predicate):
    values = attributes(item)
    type_match = predicate["object_type"] in values["object_type"]
    material_match = predicate["material"] in values["material"]
    optional = []
    if predicate.get("motif"):
        optional.append(predicate["motif"] in values["motifs"])
    if predicate.get("era"):
        start, end = ERAS[predicate["era"]][:2]
        optional.append(
            item.get("year_from") is not None
            and item.get("year_to") is not None
            and item["year_to"] >= start
            and item["year_from"] <= end
        )
    if type_match and material_match and all(optional):
        return 3
    if type_match and material_match:
        # A core type/material match remains strongly related even if a motif
        # or era disagrees. This is deliberately broad catalog-silver grading,
        # not a claim that the required detail is visible in the image.
        return 2
    if type_match or material_match and any(optional):
        return 1
    return 0


def query_text(predicate):
    typ, mat = predicate["object_type"], predicate["material"]
    en = f"a {mat} {typ}"
    zh = f"{ZH_MATERIALS.get(mat, mat)}{ZH_TYPES.get(typ, typ)}"
    if predicate.get("motif"):
        motif = predicate["motif"]
        en += f" decorated with {motif}"
        zh += f"，带有{ZH_MOTIFS.get(motif, motif)}纹饰"
    if predicate.get("era"):
        en += f" from the {predicate['era']} dynasty"
        zh = ERAS[predicate["era"]][2] + zh
    return zh, en


def candidate_pool(items):
    """Keep one deterministic representative per informative graded label vector."""
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("Duplicate item_id in benchmark catalog")
    pairs = Counter((x.get("object_type"), x.get("material")) for x in items)
    candidates = []
    for (typ, mat), count in sorted(pairs.items()):
        if (
            count < 3
            or not typ
            or not mat
            or typ in {"other", "unknown", "object"}
            or mat in {"other", "unknown"}
        ):
            continue
        base = {"object_type": typ, "material": mat}
        candidates.append(base)
        subset = [x for x in items if x.get("object_type") == typ and x.get("material") == mat]
        motifs = Counter(itertools.chain.from_iterable(set(x.get("motifs", [])) for x in subset))
        candidates.extend({**base, "motif": motif} for motif, n in sorted(motifs.items()) if n >= 3)
        candidates.extend(
            {**base, "era": era}
            for era in ERAS
            if sum(grade(x, {**base, "era": era}) == 3 for x in subset) >= 3
        )
    # Prefer the simpler intent when optional constraints convey no extra label
    # information. Different wording/eras with identical qrels are one group.
    def predicate_hash(predicate):
        return hashlib.sha256(json.dumps(predicate, sort_keys=True).encode()).hexdigest()
    groups, excluded = {}, []
    for predicate in sorted(candidates, key=lambda value: (len(value), predicate_hash(value))):
        labels = {item["item_id"]: grade(item, predicate) for item in items}
        if labels and all(value >= 2 for value in labels.values()):
            excluded.append(predicate)
            continue
        fingerprint = hashlib.sha256(json.dumps(sorted(labels.items()), separators=(",", ":")).encode()).hexdigest()
        if fingerprint in groups:
            groups[fingerprint]["equivalent_predicates"].append(predicate)
        else:
            groups[fingerprint] = {"predicate": predicate, "labels": labels,
                                   "graded_qrels_sha256": fingerprint, "equivalent_predicates": []}
    selected = sorted(groups.values(), key=lambda row: predicate_hash(row["predicate"]))
    return {"candidates": selected, "raw_supported_predicates": len(candidates),
            "distinct_graded_qrels": len(selected), "excluded_all_relevant_predicates": excluded,
            "equivalent_predicate_groups": [
                {"retained": row["predicate"], "equivalent_predicates": row["equivalent_predicates"],
                 "graded_qrels_sha256": row["graded_qrels_sha256"]}
                for row in selected if row["equivalent_predicates"]]}


def build(data_path, output="eval", n_dev=40, n_test=80):
    if n_dev < 0 or n_test < 0 or n_dev + n_test < 1:
        raise ValueError("Requested dev/test counts must be nonnegative with at least one query")
    items = read_items(data_path)
    pool = candidate_pool(items)
    candidates = pool["candidates"]
    if len(candidates) < n_dev + n_test:
        raise ValueError(
            f"Only {len(candidates)} distinct informative graded-qrels groups from "
            f"{pool['raw_supported_predicates']} supported predicates; cannot fabricate {n_dev + n_test} queries. "
            "Choose explicit smaller dev/test counts or collect genuinely different data/intents."
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, selected in [
        ("dev", candidates[:n_dev]),
        ("test", candidates[n_dev : n_dev + n_test]),
    ]:
        queries, qrels = [], []
        for index, candidate in enumerate(selected, 1):
            predicate = candidate["predicate"]
            qid = f"{name}_{index:03d}"
            zh, en = query_text(predicate)
            queries.append(
                {
                    "qid": qid,
                    "text_zh": zh,
                    "text_en": en,
                    "predicate": predicate,
                    "type": "metadata_constraint" if "era" in predicate else "visual_attributes",
                    "label_source": "metadata_silver",
                    "author": "AI-assisted deterministic ontology templates",
                    "graded_qrels_sha256": candidate["graded_qrels_sha256"],
                }
            )
            labels = candidate["labels"]
            qrels.append(
                {
                    "qid": qid,
                    "grades": {i: g for i, g in labels.items() if g},
                    "label_source": "metadata_silver",
                }
            )
        for filename, records in [
            (f"queries_{name}.jsonl", queries),
            (f"qrels_{name}.jsonl", qrels),
        ]:
            (output / filename).write_text(
                "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8"
            )
    manifest = {
        "protocol_version": "metadata-silver-v2-informative-qrels-groups",
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "items": len(items),
        "dev_queries": n_dev,
        "test_queries": n_test,
        "languages_per_intent": ["zh", "en"],
        "label_source": "metadata_silver",
        "independence_warning": "Metadata-based systems share information with these labels; this evaluates catalog constraint compliance, not independently judged visual relevance. Incomplete museum fields cause false negatives.",
        "split_policy": "one representative per distinct informative graded-qrels group, deterministic SHA256 order; bilingual forms stay together; do not tune on test",
        "raw_supported_predicates": pool["raw_supported_predicates"],
        "distinct_graded_qrels": pool["distinct_graded_qrels"],
        "equivalent_predicate_groups": pool["equivalent_predicate_groups"],
        "excluded_all_relevant_predicates": pool["excluded_all_relevant_predicates"],
        "human_annotations": 0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    p.add_argument("--output", default="eval")
    p.add_argument("--dev", type=int, default=40)
    p.add_argument("--test", type=int, default=80)
    a = p.parse_args()
    print(json.dumps(build(a.data, a.output, a.dev, a.test), ensure_ascii=False, indent=2))
