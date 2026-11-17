"""Exploratory named-form and grayscale cases; these are AI-reviewed, not human gold."""

import json
from pathlib import Path
from curio.retrieve.engine import load_items
from curio.retrieve.joint_sets import REFERENCE_SEEDS, binary_decision
from curio.retrieve.joint_vlm import LocalVQA

ROOT = Path(__file__).resolve().parents[1]


def main():
    items = {r["item_id"]: r for r in load_items(ROOT / "data/snapshots/collection_v2.jsonl")}
    meiping = ["cma_140168", "cma_284746", "met_49836", "met_42242"]
    other = ["met_50235", "met_46958", "met_46557", "met_50886"]
    gourd = ["met_48686", "cma_94798", "met_48516", "met_47511"]
    gray = ["met_47334", "met_48328", "met_48308", "met_48637"]
    groups = [
        ("梅瓶", "meiping vase", meiping, other),
        ("葫芦瓶", "double gourd vase", gourd, meiping[:2] + other[:2]),
        ("白色瓶子", "white vase", [], gray),
    ]
    out = ROOT / "artifacts/joint_vlm_v9"
    protocol = {
        "label_source": "root AI contact-sheet review and prior v6 review; not independent human labels",
        "selection": "four previously reviewed positives and four near-shape negatives per form, plus four reported grayscale errors",
        "groups": groups,
        "release_gate": False,
    }
    (out / "specialty_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    )
    scorer = LocalVQA(ROOT)
    rows = []
    try:
        for zh, en, positives, negatives in groups:
            plan = scorer.plan(zh)
            assert scorer.plan(en).key == plan.key
            refs = [
                {"item_id": i, "image_path": items[i]["image_path"], "yes": yes}
                for clause in plan.clauses
                for i, yes in REFERENCE_SEEDS.get(clause, [])
            ]
            for ident in positives + negatives:
                item = items[ident]
                assert all(
                    r["item_id"] != ident
                    and not any(
                        item.get(k) and item[k] == items[r["item_id"]].get(k)
                        for k in ["object_group", "image_sha256"]
                    )
                    for r in refs
                )
                raw = scorer.score(item["image_path"], plan)
                adapted = scorer.score(item["image_path"], plan, references=refs) if refs else raw
                row = {
                    "query": zh,
                    "paired_query": en,
                    "query_key": plan.key,
                    "item_id": ident,
                    "expected": ident in positives,
                    "raw_yes": binary_decision(raw)[0],
                    "reference_yes": binary_decision(adapted)[0],
                    "raw": raw,
                    "with_references": adapted,
                }
                rows.append(row)
                print(
                    zh,
                    ident,
                    "expected",
                    row["expected"],
                    "raw",
                    row["raw_yes"],
                    "reference",
                    row["reference_yes"],
                    flush=True,
                )
        free_plans = []
        for text in [
            "玉琮",
            "jade cong",
            "抱月瓶",
            "moonflask",
            "红色没有把手的罐子",
            "a red jar without handles",
        ]:
            try:
                plan = scorer.plan(text)
                free_plans.append({"query": text, **plan.payload()})
                print("PLAN", text, plan.canonical, flush=True)
            except (ValueError, RuntimeError) as error:
                free_plans.append({"query": text, "error": str(error)})
                print("PLAN ERROR", text, str(error), flush=True)
        report = {
            "protocol": protocol,
            "model": scorer.identity,
            "count": len(rows),
            "raw_correct": sum(r["expected"] == r["raw_yes"] for r in rows),
            "reference_correct": sum(r["expected"] == r["reference_yes"] for r in rows),
            "rows": rows,
            "free_query_plans": free_plans,
        }
        (out / "specialty_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
