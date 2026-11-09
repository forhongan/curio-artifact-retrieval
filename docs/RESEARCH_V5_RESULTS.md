# 本轮研究结果与复现 · 2026-11-08

本轮完成评测修订、组员检查点及扩展图库对照、辅助视觉消融、集合边界上线、真实外部参考路径和技术稿。没有使用子代理；未更改生产模型默认选择。

## 评测先修什么

200 张图逐件作图像/官方来源核查，分成60 dev、140 test；20个中英文同义意图生成4,000个规则可审计判断。兼容宽器型、多纹样共存；区分主体与附件材质；黑白图不强判颜色。只有必要条件全部满足才是二值正例；证据不足为 unknown，不把粗“清代”判成“不是乾隆”。有明确反证的部分相关项在集合中仍为负。

详见 [协议](RELEVANCE_V5.md)。这些是 AI/来源辅助判断，不是新专家盲测。known-only P@5 会移除未知，不能冒称用户看到的原始前五准确率。全库作决策，但质量指标只覆盖穷举已判核心，另有真实返回图片核查。

## 模型与工作流

相同4,692候选、相同修订test核心、原始文本输入：

| 模型 | EN nDCG@10 | ZH nDCG@10 | EN known P@5 | ZH known P@5 |
|---|---:|---:|---:|---:|
| CLIP | .6233 | .1408 | .550 | .110 |
| SigLIP2 | .7219 | .6385 | .640 | .620 |
| 现有生产 LoRA | .7954 | .7217 | .700 | .670 |
| 组员双语池化头 LoRA | .7511 | .7104 | .650 | .640 |

组员实际训练仍为原train的3,474图+218条细节描述；10,200是图库规模，不是训练样本数。其新权重已实际加载，16张图库图独立编码校验最小余弦约.999997，之后复用缓存。大图库来自Met/MIA/Cleveland/AIC，没有故宫。

还真实运行组员未改的 SearchService，在两个检查点/两个图库上对比 image_only、subquery_and、compositional，并统一探针权重.15。原4,692图库的 image_only 英文/中文 nDCG：现有模型 .8391/.8063，新模型 .7980/.7903。此控制不更改生产探针权重0；组合工作流未稳定胜过其image_only。模型差异的配对区间仍跨0，不能断言普遍优劣。

组员评测的器型材质满足但年代/纹样错误可给grade2，再以≥2计二值正例，不能作为本任务“全部条件满足”的更好真值。发现MIA一条BCE日期被解析成正公元数值。扩展库四查询结果池25张中18张实际可看、7张AIC受403与浏览器验证阻挡，仍为未知；实际存在明代混入元/清、乾隆混入元/宋/玉器等错误。详见 [审计](TEAMMATE_MODEL_AUDIT.md)。

## 辅助组件放在哪里

| 组件 | 实测结论 | 采用位置 |
|---|---|---|
| HSV颜色 | 文本排序及英语集合有小收益 | train-only图像颜色头→支持属性的软加分 |
| 去背景HSV | 英文排序.7954→.8190；配对区间跨0 | 可选辅助，失败回退，不全库强删背景 |
| DINO | 文本属性融合对生产dev选权重0；软邻域有探索收益 | 图图软融合/邻域支持，避免硬排除 |
| 器型/轮廓/HOG | 未证明稳定超过强图文模型 | 研究对照与可用证据，不设硬门槛 |
| LBP纹理 | 不同方法/语言结果混合 | 保留消融，不宣称专门识别年代纹样 |

正确dev参考→独立test图的18意图控制：现有图图nDCG .7174，DINO单独.5941，现有+DINO .7347，现有+前景颜色 .7306。融合权重.25是明确的探索设定，P@5反而下降，因此没有推广为普遍最佳。此控制衡量匹配组件，不代表API能自动找到正确参考。

既有RTX3070整库编码计时（4692图，batch16，含读图预处理/不含加载保存）：CLIP126.4s、SigLIP第一代113.9s、SigLIP2 104.4s、生产LoRA102.5s、DINO FP32 122.0s。它是独立性能实验，质量重测没有重复编码全库。已有缓存时全量余弦排序约1.56ms，DINO半库预筛重排约3.99ms且会漏正例，当前规模不值得默认硬预筛。

## 不定量yes/no边界

全4692图打分，再用dev选择的共享阈值或逻辑回归边界决定每张图是否接受，输出数量不固定。逻辑回归输入为原始余弦、对通用馆藏图的margin、全库标准化分数；颜色版加train-only HSV预测。没有根据test选择阈值。

| 方法 | EN宏F1 | ZH宏F1 | EN微Precision | EN微Recall |
|---|---:|---:|---:|---:|
| 原GMM | .4686 | .4613 | .6641 | .3452 |
| 共享余弦阈值 | .5292 | .5257 | .4426 | .7500 |
| 逻辑回归 | .5734 | .5149 | .6497 | .5079 |
| HSV辅助阈值 | .5831 | .5240 | .6209 | .5198 |

对照说明简单边界已能改善召回/F1，但有精确率取舍，中文并非复杂方法最好。这里的召回只针对140图核心；dev/test共享查询意图，尚未证明未见查询泛化。逻辑输出为支持分，不是匹配正确概率。

接口 `POST /api/search/set` 的method分别为 `dev_cutoff` / `dev_calibrated` / `dev_color`，返回全部 `results`，`top_k` 只限 `preview`。前端文本入口可选择。阈值文件绑定生产索引哈希。实际HTTP与独立单条推理一致；相对离线批量FP16，绿色瓶HSV集合有1张阈值附近变化，台账保留此差异。

## 专业描述→API参考→图图

新增 qualified 模式，支持V&A、MIA、AIC、Cleveland适配器；原文或显式翻译作为API文本查询，有限双语词桥明确披露。先核对来源是否支持识别到的条件（年代按制作时间而非款识），再实际下载解码图片；保留权利/署名，去重并隔离同目标对象。取不到合格照片明确返回空。自动来源支持不等于模型已看图确证。

乾隆粉彩瓶同一英文查询实际获得8张V&A照片；主代理逐张看参考，并查看7种排名合并后的14张目标及官方描述。原文本和展开文本都2/5符合；参考均值、分组均衡、查询辅助参考权重、单参考各3/5；max为2/5。仍会返回铜胎画珐琅或顺治罐；相似外观不能保证材质和历史身份。只证明该案例有用，不能报专业查询总体60%准确率。MIA适配器也已实际取到3张可解码照片；AIC本环境403，不伪称成功。

```powershell
.venv/Scripts/python.exe scripts/search_dynamic_references.py "Qianlong famille rose vase" --mode qualified --aggregation uniform_mean --output artifacts/my_qualified_query
# 或POST /api/search/reference/dynamic，传 mode="qualified"、aggregation="uniform_mean"
```

五种聚合：single_best、uniform_mean、max、diverse、query_aligned。最后一种使用文本辅助参考加权；候选打分均不查目标描述。默认前端使用qualified；API保留direct默认以兼容旧调用。

## 交付与复现

- [英文技术稿](TECHNICAL_REPORT_DRAFT.md)，PDF在 `output/pdf/Curio_Technical_Report_Draft.pdf`。按课程A4/12pt Times New Roman/单倍/1英寸页边排版，无封面目录；当前6页含参考文献，正文约4页多。仍需实际成员贡献和指定peer review。
- 原始查询实验：`scripts/evaluate_relevance_v5.py`；组员实际工作流：`scripts/evaluate_teammate_workflow_v5.py`；图图组件：`scripts/evaluate_image_components_v5.py`。
- 汇总：`scripts/summarize_relevance_v5.py`；专业单例：`scripts/compare_qualified_case_v5.py`。
- 接口核验：`scripts/verify_research_v5_api.py`，保存 `artifacts/relevance_v5/api_acceptance.json`；它只重新编码查询文本，复用图库及参考缓存。
- 文稿生成：`scripts/build_technical_report_v5.py`（需reportlab/Pillow；本机用Codex bundled artifact Python运行）。
- 510项后端测试通过、Ruff通过、前端构建通过；完整来源、权重/下载哈希、冻结判断和逐查询指标保留。

旧状态与旧报告归档 `history/docs_before_relevance_v5/`，初版v5归档 `history/evaluation_v5_initial/`。历史路径保留供引用，新增成果没有重新打包入旧Release，也没有提交或推送。
