"""Build an evidence table and deterministic failure shortlist from completed runs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentage(value):
    return f"{100 * value:.2f}%"


def build(data, evaluation, selection, output):
    items = rows(data)
    digest = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    selected = read(selection)
    if selected["data_sha256"] != digest:
        raise ValueError("Selection/data mismatch")
    evidence = {"data_sha256": digest, "selection": selected, "reports": {}}
    lines = [
        "# 元数据辅助对照结果",
        "",
        "本文件由 `scripts/summarize_results.py` 从实验 JSON 汇总；数值不是手工示例。",
        "自由查询银标衡量元数据符合程度；独立图像判断另标为 AI 视觉审计，均不冒充真人金标准。",
        "",
        "## 冻结数据",
        "",
        f"- 图库：{len(items):,} 件；split：`{dict(Counter(x['split'] for x in items))}`。",
        f"- 来源：`{dict(Counter(x['source'] for x in items))}`；数据 SHA256：`{digest}`。",
        f"- 冻结选择：`{selected['frozen_at']}`。测试集不参与训练、epoch 或重排权重选择。",
        "",
        "## 对象留出配对检索",
        "",
        "每个 test 对象的弱元数据 caption 查询同一 test 图库；完全相同 caption 的图像均算相关。",
        "重复模板使此任务比细粒度视觉描述匹配容易；这里的 Recall@K 是查询命中率。",
        "",
        "| 模型 | 图库/查询 | 独立 caption | R@1 | R@5 | R@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("clip", "siglip", "adapted"):
        report = read(evaluation / f"paired_{name}_test.json")
        if report["data_sha256"] != digest:
            raise ValueError(f"Stale paired result: {name}")
        evidence["reports"][f"paired_{name}_test"] = report
        m = report["metrics"]
        lines.append(
            f"| {name} | {m['gallery']} | {m['unique_captions']} | {percentage(m['r@1'])} | {percentage(m['r@5'])} | {percentage(m['r@10'])} | {m['mrr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 训练与验证选择",
            "",
            "图像塔 LoRA r=8，冻结文本塔；两种 batch 采样各进行实际训练。caption 删除显式年代、朝代、馆名，但标题和材料仍可能带用途/身份语义，属于弱监督。",
            "",
            "| 采样 | 最佳 epoch | dev R@5 | dev MRR | 优于零样本 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for candidate in selected["training_candidates"]:
        m = candidate["best_dev"]
        lines.append(
            f"| {candidate['sampling']} | {candidate['best_epoch']} | {percentage(m['r@5'])} | {m['mrr']:.4f} | {candidate['improved_over_baseline']} |"
        )
    lines.extend(
        [
            "",
            f"选中 adapter：`{selected['adapter_path']}`；是否改善 paired dev：**{selected['adapted_improves_paired_dev']}**。",
            f"视觉属性软重排权重：**{selected['probe_weight']}**（仅使用 dev 双语 silver nDCG@10 选择；不是视觉真实性校准）。",
            "",
            "## 自由查询银标测试",
            "",
            "每个语义意图包含中英两种表述，按语言分别统计。R@20 是检索到的相关图像占全部相关图像的比例；广类相关项很多时该值自然较低。",
            "nDCG 使用 0–3 级相关性和 `2^grade−1` 增益；P/AP/Recall/MRR 以 grade≥2 为相关。AP@10 的分母为全部相关项数。",
            "CI 对语义查询 bootstrap；不能将中英改写视作两倍独立样本。",
            "",
            "| 模型 | 变体 | 语言 | nDCG@10 [95% CI] | P@5 | AP@10 | R@20 | MRR@10 |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("clip_test", "siglip_test", "siglip_adapted_test", "visual_test"):
        report = read(evaluation / name / "metrics.json")
        if report["data_sha256"] != digest:
            raise ValueError(f"Stale free-query result: {name}")
        evidence["reports"][name] = report
        for result in report["summary"]:
            m = result["metrics"]
            n = m["ndcg@10"]
            lines.append(
                f"| {name} | {result['variant']} | {result['language']} | {n['mean']:.4f} [{n['lower']:.4f}, {n['upper']:.4f}] | {m['precision@5']['mean']:.4f} | {m['ap@10']['mean']:.4f} | {m['recall@20']['mean']:.4f} | {m['mrr@10']['mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 成对差值",
            "",
            "候选减基线。每组比较对预先指定的五个指标×两种语言进行 Holm 校正；完整输出保留所有指标与正负结果。下面仅展示主要指标 nDCG@10。",
            "",
            "| 对比 | 语言 | ΔnDCG@10 [95% CI] | p (Holm) |",
            "|---|---|---|---:|",
        ]
    )
    for path in sorted((evaluation / "comparisons").glob("*/comparison.json")):
        report = read(path)
        evidence["reports"][f"comparison_{path.parent.name}"] = report
        for result in report["comparisons"]:
            if result["metric"] != "ndcg@10":
                continue
            d = result["delta"]
            lines.append(
                f"| {path.parent.name} | {result['language']} | {d['mean']:+.4f} [{d['lower']:+.4f}, {d['upper']:+.4f}] | {result['p_value_holm']:.5f} |"
            )
    lines.extend(
        [
            "",
            "## 自动失败短名单",
            "",
            "以下按 hybrid−baseline nDCG@10 最差到最好排序，固定保留前六条；用于进一步看图分析，不自动推断错误原因。",
            "详见 `artifacts/evaluation/failure_shortlist.json`（含两个系统前五名及银标等级）。",
            "",
        ]
    )
    details = rows(evaluation / "siglip_test/per_query.jsonl")
    rankings = {
        (r["qid"], r["language"], r["variant"]): r
        for r in rows(evaluation / "siglip_test/rankings.jsonl")
    }
    grouped = {}
    for record in details:
        grouped.setdefault((record["qid"], record["language"]), {})[record["variant"]] = record
    qrels = {r["qid"]: r["grades"] for r in rows(ROOT / "eval/qrels_test.jsonl")}
    failures = []
    for key, systems in grouped.items():
        first, second = systems["baseline"], systems["hybrid"]
        record = {
            "qid": key[0],
            "language": key[1],
            "delta_ndcg@10": second["ndcg@10"] - first["ndcg@10"],
            "label_source": "metadata_silver",
        }
        for variant in ("baseline", "hybrid"):
            ranked = rankings[(*key, variant)]
            record["query"] = ranked["query"]
            record[variant] = [
                {"item_id": item_id, "grade": qrels[key[0]].get(item_id, 0)}
                for item_id in ranked["ranking"][:5]
            ]
        failures.append(record)
    failures.sort(key=lambda row: (row["delta_ndcg@10"], row["qid"], row["language"]))
    for row in failures[:6]:
        lines.append(
            f"- `{row['qid']}` ({row['language']}), “{row['query']}”：ΔnDCG@10={row['delta_ndcg@10']:+.4f}。"
        )
    (evaluation / "failure_shortlist.json").write_text(
        json.dumps(failures[:12], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 元数据银标与 BM25/slot 共享信息来源，因此高分不能证明更强的视觉理解。",
            "- 器型/材料 probe 从馆藏弱标签训练，概率只用于软重排；年代、文化和身份不能从该分类器视为已识别。",
            "- 当前没有真人标注，不报告真人一致性、双标 κ 或人工 gold 收益。",
            "- 独立 AI 盲图审计、真实 API 延迟与浏览器验收在对应附录中单独报告。",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    output.with_name("experiment_evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/snapshots/collection_v2.jsonl")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "artifacts/evaluation")
    parser.add_argument("--selection", type=Path, default=ROOT / "artifacts/selection.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/METADATA_RESULTS.md")
    args = parser.parse_args()
    build(args.data, args.evaluation, args.selection, args.output)


if __name__ == "__main__":
    main()
