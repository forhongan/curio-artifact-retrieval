"""Export a read-only visual comparison, keeping fresh tests and old cases distinct."""

import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/boundary_shape_v14"
OUT = ROOT / "artifacts/shape_boundary_v14"
NAMES = {"meiping": "梅瓶", "gourd": "葫芦瓶", "bowl": "碗形（不限颜色）", "red_bowl": "红色碗"}
SCOPES = {
    "test": "新测试",
    "dev": "校准组",
    "old_test": "旧测试回归",
    "old_failures": "旧失败回归",
    "old_gallery_audit": "旧全库抽查回看",
    "countergate_audit": "葫芦瓶全部候选复验",
}


def readl(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    report = json.loads((BASE / "report.json").read_text())
    rows = [
        dict(r, scope=r["split"])
        for r in readl(BASE / "predictions.jsonl")
        if r["split"] != "train"
    ] + readl(BASE / "regression_predictions.jsonl")
    counter_report = None
    if (BASE / "countergate_audit_report.json").exists():
        counter_report = json.loads((BASE / "countergate_audit_report.json").read_text())
        old = {
            r["item_id"]: r
            for r in json.loads(
                (ROOT / "artifacts/surface_boundary_v13/gallery/gourd.json").read_text()
            )["decisions"]
        }
        new = {
            r["item_id"]: r
            for r in json.loads((OUT / "countergate_gallery.json").read_text())["decisions"]
        }
        for r in readl(BASE / "countergate_audit_labels.jsonl"):
            prior, decision = old[r["item_id"]], new[r["item_id"]]
            e = prior["evidence"][0]
            rows.append(
                {
                    **r,
                    "scope": "countergate_audit",
                    "visual_yes": decision["yes"],
                    "baseline_yes": prior["yes"],
                    "score": e["score"],
                    "threshold": e["threshold"],
                    "catalogue_evidence": None,
                    "contradictions": decision.get("contradictions", []),
                }
            )
    summary = []
    for atom, result in report["per_atom_test"].items():
        b, v = result["v13_baseline"], result["v14_visual_only"]
        summary.append(
            f"<tr><td>{NAMES[atom]}</td><td>{v['n']}</td>"
            f"<td>{b['tp']} / {b['fp']} / {b['fn']}</td>"
            f"<td>{v['tp']} / {v['fp']} / {v['fn']}</td></tr>"
        )
    cards = []
    for r in rows:
        with Image.open(ROOT / r["image_path"]) as opened:
            picture = ImageOps.contain(ImageOps.exif_transpose(opened).convert("RGB"), (360, 300))
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=84)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        wrong = r["visual_yes"] != r["decision"]
        note = html.escape(r["notes"])
        conflict = r["catalogue_evidence"]
        detail = ""
        for name in r.get("contradictions", []):
            reason = {"parallel": "并排双容器", "accessory": "人物/动物雕像中的附属容器"}[name]
            detail += f"<p>图像反例检查：{reason}，因此排除。</p>"
        if conflict:
            detail = (
                "<p class='aux'>资料辅助排除："
                + html.escape(conflict["value"])
                + "。这项辅助判断不计入纯视觉成绩。</p>"
            )
        if r["reason"] == "vertical_topology_rejected":
            detail += "<p>上下双腹结构未获图像支持。</p>"
        if r.get("unchanged_color_passed") is False:
            detail += "<p>颜色条件未通过，组合查询仍为 No。</p>"
        label = (
            "支持"
            if r["grade"] == 3
            else "明确不符"
            if r["grade"] == 0
            else "歧义／证据不足，严格不收"
        )
        before, after = ("Yes" if r[k] else "No" for k in ["baseline_yes", "visual_yes"])
        method = "追加反例复验" if r["scope"] == "countergate_audit" else "本轮纯视觉"
        cards.append(
            f'<article data-scope="{r["scope"]}" data-query="{r["qid"]}" data-error="{str(wrong).lower()}">'
            f'<img src="data:image/jpeg;base64,{encoded}" alt="{html.escape(r["item_id"])}">'
            '<div class="body">'
            f"<small>{SCOPES[r['scope']]} · {html.escape(r['item_id'])}</small>"
            f'<h2>{NAMES[r["qid"]]} · <span class="{"bad" if wrong else "ok"}">{after}</span></h2>'
            f"<p>上一版 {before} → {method} {after}</p>"
            f"<p>器形分数 {r['score']:.3f} / 阈值 {r['threshold']:.3f}</p>"
            f'<p>AI 辅助标签：{label}</p>{detail}<p class="note">{note}</p></div></article>'
        )
    options = "".join(f'<option value="{k}">{v}</option>' for k, v in SCOPES.items())
    queries = "".join(f'<option value="{k}">{v}</option>' for k, v in NAMES.items())
    document = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>器形边界 v14 · 新样本与失败复验</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f5f1;color:#242c2a;font:15px/1.6 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px 20px}h1{font-size:28px}h2{font-size:18px;margin:6px 0}p{margin:8px 0}.intro{max-width:900px}.badge{color:#83551e}.tablewrap{overflow:auto;margin:24px 0}table{border-collapse:collapse;width:100%;max-width:900px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6d8d1;white-space:nowrap}th{background:#eceee7}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px 0;position:sticky;top:0;background:#f6f5f1;z-index:1}select{padding:8px;font:inherit;max-width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:18px}article{background:white;border:1px solid #dddcd4;border-radius:10px;overflow:hidden}article img{height:280px;width:100%;object-fit:contain;background:#ecece8}.body{padding:16px}small,.note{color:#606965}.ok{color:#16634e}.bad{color:#b23828}.aux{background:#fff4dd;padding:10px;border-radius:6px}.note{font-size:13px}article[hidden]{display:none}
</style><main><p class="badge">离线研究 · 未替换默认算法</p><h1>器形边界：新样本与失败复验</h1>
<div class="intro"><p>本轮增加器身轮廓、肩部位置、上下腹与束腰证据，三个视觉分项分别学习和校准。颜色判断沿用上一轮。</p>
<p>133 张新图由助手看图并结合馆藏资料标注，其中 48 项用于新测试。标签不是独立人工金标准。馆藏明确为杯时可另作辅助排除，下表只计纯视觉结果；分数不是准确率。</p>
<p>新葫芦瓶测试没有正例，只能检查误收，不能衡量召回。旧测试、旧失败和旧全库抽查仅用于回归，不与新测试合并。下方不同阶段可能包含同一图片。</p></div>
<div class="tablewrap"><table><thead><tr><th>新测试类别</th><th>样本数</th><th>上一版：命中／误收／漏收</th><th>本轮纯视觉：命中／误收／漏收</th></tr></thead><tbody>"""
    document += "".join(summary) + "</tbody></table></div>"
    if counter_report:
        b, v = counter_report["before"], counter_report["after"]
        document += (
            '<div class="intro"><p><strong>葫芦瓶追加反例试验（旧数据探索）</strong>：保留上一版基础判别，'
            "增加并排容器与雕像配件排除。全部 59 个粗筛候选已由助手逐图复核："
            f"误收 {b['fp']} → {v['fp']}，漏收 {b['fn']} → {v['fn']}；全库模型入选 "
            f"{counter_report['gallery_yes_before']} → {counter_report['gallery_yes_after']}。"
            "这不是新的独立测试，也不衡量粗筛之外的漏检。</p><p>剩余两件歧义为三节葫芦瓶。"
            "当前计划将泛称“葫芦瓶”统一为较窄的 double-gourd，术语范围需要进一步拆分；"
            "本轮保留既有定义以便比较，不把它们直接改标签计入成绩。</p></div>"
        )
    document += f'<div class="toolbar"><select id="scope" aria-label="验证阶段"><option value="">全部阶段</option>{options}</select><select id="query" aria-label="查询类别"><option value="">全部类别</option>{queries}</select><label><input id="errors" type="checkbox">只看判错项</label><span id="count"></span></div><div class="grid">'
    document += (
        "".join(cards)
        + """</div></main><script>
const scope=document.querySelector('#scope'),query=document.querySelector('#query'),errors=document.querySelector('#errors');scope.value='test';
function filter(){let n=0;for(const c of document.querySelectorAll('article')){c.hidden=!!((scope.value&&c.dataset.scope!==scope.value)||(query.value&&c.dataset.query!==query.value)||(errors.checked&&c.dataset.error!=='true'));if(!c.hidden)n++;}document.querySelector('#count').textContent=n+' 项';}
scope.onchange=query.onchange=errors.onchange=filter;filter();</script></html>"""
    )
    (OUT / "report.html").write_text(document)
    print(f"Read-only shape report: {len(rows)} cards; new test selected by default")


if __name__ == "__main__":
    main()
