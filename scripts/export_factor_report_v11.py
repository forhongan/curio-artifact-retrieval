"""Standalone, read-only visual report; never writes human review labels."""

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"red_bowl": "红色碗", "blue_white": "青花瓷", "meiping": "梅瓶", "gourd": "葫芦瓶"}
ATOMS = {
    "red": "主体红色",
    "bowl": "碗形",
    "blue_white": "青花外观",
    "meiping": "梅瓶轮廓",
    "gourd": "双腹束腰",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=[11, 12, 13], type=int, default=11)
    version = parser.parse_args().version
    dataset = {11: "boundary_factors_v11", 12: "boundary_monotone_v12", 13: "boundary_surface_v13"}[
        version
    ]
    artifact = {11: "factor_boundary_v11", 12: "monotone_boundary_v12", 13: "surface_boundary_v13"}[
        version
    ]
    base = ROOT / "eval" / dataset
    out = ROOT / "artifacts" / artifact / "report.html"
    cards = []
    sources = [
        ("predictions.jsonl", "新留出" if version == 11 else "已见留出回放"),
        ("stress_predictions.jsonl", "旧失败回归"),
    ]
    if version == 13 and (base / "gallery_audit_labels.jsonl").exists():
        sources.append(("gallery_audit_labels.jsonl", "全库入选抽查（非独立测试）"))
    for file, scope in sources:
        for row in map(json.loads, (base / file).read_text().splitlines()):
            with Image.open(ROOT / row["image_path"]) as original:
                image = ImageOps.contain(original.convert("RGB"), (300, 250))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            correct = row["yes"] == row["decision"]
            details = "".join(
                f"<li>{ATOMS[e['atom']]}：{'通过' if e['yes'] else '未通过'} "
                f"（分数 {e['score']:.3f} / 阈值 {e['threshold']:.3f}）</li>"
                for e in row["evidence"]
            )
            label = (
                "符合"
                if row["grade"] == 3
                else "明确不符"
                if row["grade"] == 0
                else "证据不足，严格不收"
            )
            partition = {
                "accepted_unreviewed_sample": "首次抽查",
                "accepted_previously_reviewed": "已有标签回看",
            }.get(row["split"], row["split"])
            cards.append(
                f'<article data-query="{row["qid"]}" data-error="{str(not correct).lower()}">'
                f'<img src="data:image/jpeg;base64,{encoded}" alt="{html.escape(row["item_id"])}">'
                f'<div class="body"><small>{scope} · {html.escape(partition)} · {html.escape(row["item_id"])}</small>'
                f'<h2>{NAMES[row["qid"]]} · <span class="{"ok" if correct else "bad"}">{"Yes" if row["yes"] else "No"}</span></h2>'
                f"<p>AI 辅助标签：{label}</p><ul>{details}</ul>"
                f'<p class="note">{html.escape(row["notes"])}</p></div></article>'
            )
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>分项边界实验 v11</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f5f1;color:#242c2a;font:15px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:32px 20px}h1{font-size:28px}h2{font-size:19px;margin:6px 0}.intro{max-width:850px}.toolbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;padding:16px 0;position:sticky;top:0;background:#f6f5f1;z-index:1}select{padding:8px;font:inherit}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:18px}article{background:white;border:1px solid #dddcd4;border-radius:10px;overflow:hidden}article img{height:250px;width:100%;object-fit:contain;background:#ecece8}.body{padding:16px}small,.note{color:#606965}.ok{color:#16634e}.bad{color:#b23828}ul{padding-left:20px}.note{font-size:13px}article[hidden]{display:none}
</style><main><h1>分项判别 · 实验结果</h1>
<p class="intro">模型分别判断主体颜色、器形等条件，全部通过才输出 Yes。这是离线实验记录，未替换默认检索算法。标签由 AI 看图并结合馆藏资料复核；分数不是准确率。新留出用于检验，旧失败回归用于诊断，两者不合并宣称独立测试成绩。</p>
<div class="toolbar"><select id="query" aria-label="查询类别"><option value="">全部查询</option><option value="red_bowl">红色碗</option><option value="blue_white">青花瓷</option><option value="meiping">梅瓶</option><option value="gourd">葫芦瓶</option></select><label><input id="errors" type="checkbox">只看与辅助标签不一致的结果</label><span id="count"></span></div><div class="grid">"""
    document += (
        "".join(cards)
        + """</div></main><script>
const query=document.querySelector('#query'), errors=document.querySelector('#errors');
function filter(){let n=0;for(const card of document.querySelectorAll('article')){card.hidden=!!((query.value&&card.dataset.query!==query.value)||(errors.checked&&card.dataset.error!=='true'));if(!card.hidden)n++;}document.querySelector('#count').textContent=n+' 项';}
query.onchange=filter;errors.onchange=filter;filter();</script></html>"""
    )
    document = document.replace("分项边界实验 v11", f"分项边界实验 v{version}")
    if version != 11:
        document = document.replace(
            "新留出用于检验，旧失败回归用于诊断", "本轮使用已经查看过的留出与失败案例作探索回放"
        )
    if version == 13 and (base / "gallery_summary.json").exists():
        summary = json.loads((base / "gallery_summary.json").read_text())["queries"]
        counts = "；".join(f"{NAMES[q]} {r['yes']} 件" for q, r in summary.items())
        document = document.replace(
            '<div class="toolbar">',
            '<p class="intro">全库 4,692 张图片：'
            + counts
            + '。这些是模型入选数量，不是正确数量。粗筛后的 424 个候选经过分项复验；'
            '未进入候选的图片仍可能漏检。下方另有 33 项新入选抽查及 3 项已复核红碗回看，'
            '均未用于本轮调参。灰度图可以支持器形，不能支持颜色。</p><div class="toolbar">',
        )
    out.write_text(document)
    print(out, len(cards), "read-only result cards")


if __name__ == "__main__":
    main()
