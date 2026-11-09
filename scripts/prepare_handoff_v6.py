"""Assemble the five research handoffs and the presentation's actual speaker notes."""

from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def write(path, text):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.strip() + "\n", encoding="utf-8")


def main():
    modules = [
        (
            "A_DATA",
            "数据与相关性评测",
            "1–2",
            "55秒",
            "让不同模型在同一批图片、同一语义请求和可追溯判断下比较。",
            "冻结 Met 2,954 + Cleveland 1,738 张图像；保留来源与许可。修订200张图的外观/来源事实，以60张dev选择边界、140张test比较20个中英文意图。器型兼容族、多纹样和未知项替代僵硬的单标签。",
            "4,000条规则衍生判断；排序移除未知项，集合评估单列未知入选率。质量指标范围是140张test核心。",
            "docs/RELEVANCE_V5.md；eval/relevance_v5/；src/curio/evaluate/relevance.py",
            ".venv/Scripts/python.exe scripts/evaluate_relevance_v5.py --help",
        ),
        (
            "B_REPRESENTATION",
            "视觉表示、微调与双语查询",
            "3–4",
            "80秒",
            "研究图像库仅提供像素时，领域适配如何改善文字到图像的匹配。",
            "对照CLIP、SigLIP2、视觉LoRA与组员的双语池化头LoRA；检验真实权重、训练样本和10,200图库。训练使用馆藏文字监督，在线候选使用图像向量；中英文保留完整描述并作有限查询展开。",
            "原始英文nDCG@10：CLIP 0.6233、SigLIP2 0.7219、生产LoRA 0.7954、组员LoRA 0.7511；中文分别0.1408、0.6385、0.7217、0.7104。",
            "docs/RESEARCH_V5_RESULTS.md；docs/TEAMMATE_MODEL_AUDIT.md；src/curio/represent/；scripts/evaluate_teammate_workflow_v5.py",
            ".venv/Scripts/python.exe scripts/evaluate_teammate_workflow_v5.py --help",
        ),
        (
            "C_SET_BOUNDARY",
            "全库yes/no集合边界",
            "5",
            "50秒",
            "将固定Top-K序列扩展为数量随请求变化的入选集合。",
            "比较自适应GMM、dev共享分数阈值、逻辑回归边界、HSV辅助边界。运行时对4,692张图片逐件决策，API返回完整results，preview单独截断，前端每页12张。中文推荐共享阈值，英文推荐颜色辅助；用户可切换。",
            "英文宏F1：0.4686 → 0.5292 / 0.5734 / 0.5831；中文：0.4613 → 0.5257 / 0.5149 / 0.5240。这些是test核心上的集合质量，不是全库召回率。",
            "src/curio/retrieve/calibrated_sets.py；src/curio/evaluate/set_boundary.py；tests/test_relevance_v5.py；POST /api/search/set",
            ".venv/Scripts/python.exe scripts/verify_research_v5_api.py",
        ),
        (
            "D_REFERENCES",
            "专业名称、外部参考图与图图检索",
            "6–7",
            "95秒",
            "把专业器型名转成真实参考照片，再按图像外观比较图库。",
            "博物馆API文本检索、有限同义词发现、来源条件核验、真实下载解码、同物/近重复隔离；比较五种参考聚合，SigLIP2和DINO参与像素匹配。修复只看Vase标题而漏掉正文meiping/gourd的情况：正文器型还必须与器物标题族相容。",
            "六个实际场景全部留档。梅瓶：中文生产路径0/5，英文原始文本4/5，6张真实V&A参考均值5/5（宽肩短颈渐收腹的兼容器型）。这是探索后选出的读图案例。葫芦、青瓷、龙盘、皮囊壶与抱月瓶的结果也保留。",
            "docs/REFERENCE_SCENARIOS_V6.md；src/curio/retrieve/qualified_references.py；src/curio/retrieve/reference_aggregation.py；artifacts/reference_scenarios_v6/",
            ".venv/Scripts/python.exe scripts/explore_reference_scenarios_v6.py",
        ),
        (
            "E_COMPONENTS",
            "视觉辅助、性能与产品集成",
            "8、10–11",
            "110秒",
            "选择能增加检索质量且成本合适的辅助组件，把研究结论落实到可用系统。",
            "消融HSV颜色、HOG形状、LBP纹理、轮廓、GrabCut与DINO；保留有效的软融合，实验算法集中到实验档案。建立普通用户三入口、完整集合分页、来源详情和专业统计；制作离线HTML演示与实测数据导出。",
            "控制图图nDCG@10：生产0.7174、DINO 0.5941、生产+DINO 0.7347、生产+前景颜色0.7306。4,692图LoRA编码102.5秒，缓存余弦排序1.56毫秒；DINO预筛3.99毫秒后还要重排，因此默认直接排序。",
            "docs/CLASSICAL_CV_FEATURES.md；docs/RESEARCH_V5_RESULTS.md；web/src/App.tsx；web/src/ResearchArchive.tsx；web/public/assets/presentation/",
            "pnpm --dir web build\n.venv/Scripts/python.exe -m pytest -q",
        ),
    ]
    table = []
    for key, title, pages, duration, purpose, done, result, evidence, command in modules:
        write(
            f"docs/handoff/{key}.md",
            f"""# {title}

演示模块 {key[0]} · 页面 {pages} · 讲述 {duration}

## 目的

{purpose}

## 已完成

{done}

## 可直接引用的结果

{result}

## 代码与证据

{evidence}

```powershell
{command}
```

整理报告时按“需求 → 方法 → 公平对照 → 结果 → 设计选择”展开；数字以冻结结果为准。公共小型证据见 `docs/research/`，完整实验附件见本轮Release。
""",
        )
        table.append(f"| {key[0]} | [{title}](handoff/{key}.md) | {pages} | {duration} |")
    write(
        "docs/HANDOFF.md",
        """# 五个研究与演讲模块

各模块是可以交给组员整理或继续研究的工作包，对应HTML演示的同名模块，不预设成员归属。

| 模块 | 工作包 | 演示页 | 讲述时间 |
|---|---|---|---|
"""
        + "\n".join(table)
        + """

讲述合计6分30秒；第9页操作展示60秒单独计时，总计7分30秒。讲者稿与演练流程见 [PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)。

阅读顺序：本页 → 对应工作包 → 工作包内证据。当前开发状态见 [CURRENT_STATE.md](CURRENT_STATE.md)，技术稿见 [TECHNICAL_REPORT_DRAFT.md](TECHNICAL_REPORT_DRAFT.md)。
""",
    )
    js = (ROOT / "web/public/assets/presentation/deck.js").read_text(encoding="utf-8")
    notes = re.findall(
        r"\{title:'(.*?)',seconds:(\d+),module:'(.*?)'.*?notes:'(.*?)',zh:'(.*?)'\}", js, re.S
    )
    assert len(notes) == 11, len(notes)
    assert sum(int(n[1]) for n in notes if n[2] != "Video") == 390
    guide = """# 演示流程与讲者稿

打开 `web/public/assets/presentation/index.html`，或运行服务后访问 `/assets/presentation/index.html`。HTML、图片和视频可整体离线复制，无CDN依赖。

方向键/空格翻页，N显示讲者提示，F全屏。点击Start timer开始口头计时；视频页自动暂停口头计时。共11页，讲述390秒，操作展示60秒，总计7分30秒，留30秒切换余量。下文英文稿约按正常课堂语速安排，排练时用页面图表替代逐个读数字。

主线：瓷器（大类）→绿色窄口瓷瓶（组合外观）→梅瓶（专业器型）；中间用集合边界解释“找齐符合条件的图”，最后用经典CV消融和性能解释架构选择。

录制素材由本地真实界面截取并剪辑成60秒步骤展示；屏幕切换不是检索延迟基准。若替换为小组连续录屏，保持第9页video文件接口和60秒时长。
"""
    for i, (title, seconds, module, note, cue) in enumerate(notes, 1):
        guide += f"\n## {i}. {title} · {seconds}秒 · {module}\n\n{note}\n\n讲述提示：{cue}\n"
    guide += """
## 演示前检查

1. 运行 `scripts/run.ps1` 并打开首页，提前完成一次“绿色窄口瓷瓶”和专业模式“梅瓶”查询，准备好模型与API缓存。
2. 验证第9页本地视频可播放；讲者提示默认关闭。
3. 备用：直接打开离线HTML，所有图表和梅瓶对照图片都保存在演示资产中。

## 操作展示的六个镜头

首页与三种入口 → 文字外观结果 → 完整集合与分页 → 博物馆参考照片和梅瓶结果 → 上传图片相似检索 → 专业模式实测图表。
"""
    write("docs/PRESENTATION_GUIDE.md", guide)
    write(
        "docs/REFERENCE_SCENARIOS_V6.md",
        """# 外部参考图场景补测 · 2026-11-09

在看结果前列出六个真实场景，用V&A与MIA文本API搜索，每场最多6张合格参考；比较单张、均值、最大值、多样性和查询加权五种聚合。保留API请求、下载、失败、各路排名和结果池联系表。新增中文生产查询与英文生产展开对照，图库向量全程复用。

| 请求 | 合格参考 | 读图观察 |
|---|---:|---|
| Chinese double gourd vase / 葫芦瓶 | 6 | 文本已匹配良好；多样性聚合引入部分单腹瓶，颜色装饰干扰器型 |
| Chinese meiping vase / 梅瓶 | 6 | 中文生产0/5，英文原始4/5，参考均值5/5；适合展示专业器型的视觉实例价值 |
| Chinese blue and white moon flask / 青花抱月瓶 | 0 | 没有通过来源和照片资格检查的参考，返回明确空结果 |
| Liao dynasty leather flask / 辽代皮囊壶 | 3 | 取得正确来源参考，部分目标与其他壶相近；不能仅凭轮廓确认朝代 |
| Chinese celadon vase / 青瓷瓶 | 6 | 两路均能找到绿色瓷瓶，参考中的欧式金属附件会影响外观匹配 |
| Chinese blue and white dragon dish / 青花龙纹盘 | 4 | 来源文字提到龙，但照片所拍一面不一定呈现龙纹，文本路径更合适 |

梅瓶判据：短颈、小口、宽圆肩、腹部向足渐收；接受实际视觉相近的器型族。5个目标中4个官方记录明确写meiping，另一件CMA122410为读图接受的相容器型。对应IDs与逐项依据见 `docs/research/reference_v6/root_review.json`。这些是主智能体读图和来源辅助的探索案例评分。

关键代码改进：来源标题可能只是“Vase”，器型名在描述正文。现在允许器型正文证据，但标题还必须属于vase/bottle/flask/ewer/vessel等器物族，避免把“画有梅瓶的绘画”当成参考照片。搜索原文优先，再尝试最多两种有限器型同义词；完整查询条件始终参与来源核验。

复现：`scripts/explore_reference_scenarios_v6.py` → `scripts/compare_scenario_controls_v6.py` → `scripts/review_scenarios_v6.py`。原始产物：`artifacts/reference_scenarios_v6/`；公共结果与请求清单：`docs/research/reference_v6/`。普通前端“用专业名称找 → 梅瓶”调用同一qualified流程，默认6参考、uniform_mean。
""",
    )
    dest = ROOT / "docs/research"
    dest.mkdir(exist_ok=True)
    for name in [
        "results.json",
        "thresholds.json",
        "paired_bootstrap.json",
        "team_cache_verification.json",
        "api_acceptance.json",
    ]:
        shutil.copy2(ROOT / "artifacts/relevance_v5" / name, dest / name)
    for name in ["summary.json", "root_review.json"]:
        target = dest / "reference_v6" / name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(ROOT / "artifacts/reference_scenarios_v6" / name, target)
    for folder in (ROOT / "artifacts/reference_scenarios_v6").iterdir():
        if (folder / "comparison.json").exists():
            target = dest / "reference_v6" / folder.name
            target.mkdir(exist_ok=True)
            shutil.copy2(folder / "comparison.json", target / "comparison.json")
    write(
        "docs/research/README.md",
        """# 冻结研究结果

`results.json`、`thresholds.json`和`paired_bootstrap.json`来自relevance v5.1，和专业模式图表共用结果。协议/事实/判断在 `eval/relevance_v5/`，公开完整实验附件保存在本轮Release。

`reference_v6/`保留六个实际API参考图场景及逐项读图记录。`web/public/assets/showcase/attribution.json`给出展示用Met/Cleveland CC0图片的出处。V&A参考照片按来源在线呈现，未复制进公开演示图片包。

本目录是结果快照，修改算法后应新建版本并重新运行实验，不手工改动图表数字。
""",
    )


if __name__ == "__main__":
    main()
