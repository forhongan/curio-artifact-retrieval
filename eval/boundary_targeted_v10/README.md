# 针对性样本与全库边界实验 · 2026-11-16

用户提交的 `curio-ai-assisted-067ab168 (1).json` 已校验并归档，与现有 720 项 AI 辅助标签内容一致，无新增更正。导入记录见 [收据](../boundary_assisted_v10/import_receipts/067ab168-6759252e41fe.json)。本轮继续由助手看图复核，没有要求用户逐项标注。

## 已完成

- 新增 150 张图、155 个图文对：红色碗、青花瓷、梅瓶、葫芦瓶，确认 61 个正例、63 个明确负例、28 个颜色不可观察项、3 个语义歧义项。
- 看完 8 张联系表，读取馆藏标题和材质，另放大 4 张原图。每条判断保留说明和来源，记录在 `labels.jsonl`；检索模型的输出不作为这些标签的依据。
- 馆藏关键词只帮助发现候选。红色胎体配绿釉、标题含 Gourd 的单腹扁瓶、只有局部红纹的白碗，都没有直接作为正例。没有将黑白照片里的颜色从馆藏文字补回。
- 保留原始 train/dev/test；排除 v5、上一批随机复核和 v9 专项检查的器物组及图片哈希。模型只读取已有 SigLIP 图像向量；馆藏文字不进入模型输入。
- 中英文四组查询编译成相同计划，实际 155 项 VQA 推断、训练与校准完成。生成四种查询各 4,692 图的完整布尔判断，均为离线实验。

## 算法与实测

第一阶段固定比较原 VQA、每概念视觉 Logistic Regression 分类器、两者概率均值。分类器 C=10、类别平衡，训练仅使用 train 的已知标签；阈值只在 dev 选取，目标 Precision≥90%、Recall≥50%、至少 2 TP。目标不满足时实验阈值为 0.5，并明确记录失败，不部署。

36 项留出测试含 13 个正例。在各方法都使用自己的 dev 阈值后：

| 方法 | 命中 | 误收 | 漏收 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| VQA | 10 | 2 | 3 | 83.3% | 76.9% |
| 视觉分类器 | 12 | 1 | 1 | 92.3% | 92.3% |
| 等权融合 | 9 | 0 | 4 | 100% | 69.2% |

这些小样本数字不能代表全库。随后全库运行发现视觉分类器把镜子、织物、雕塑等陌生类别大量误收：梅瓶判出 3,583 张、葫芦瓶判出 3,833 张。实际查看 64 项命中样本证实失败；单纯训练相近瓶类的分界会在库内其他类别上失效。

第二阶段保留第一阶段全部记录，新增这批诊断中**原始 train** 的已知标签，每概念每器物组只取一条；原 dev/test 不进入训练。增加正例邻域检查：最大正例余弦相似度须达到训练正例留一邻居相似度的最小值减 0.05，范围只由训练数据决定。仍按原规则在 dev 校准。

第二阶段为看过旧测试后的探索性改进，不是新的独立验证。旧测试回放 11 TP / 1 FP / 2 FN，不能据此宣称已经解决全库边界。全库输出青花瓷 226、梅瓶 94、葫芦瓶 59、红色碗 45 个 Yes；这些是预测数量，不是已核实的正确藏品数。

对第二阶段各类命中结果再按固定哈希抽取 16 个未复核器物组，由助手看图检查：

| 查询 | 支持 | 明确不符 | 歧义 |
|---|---:|---:|---:|
| 青花瓷 | 16 | 0 | 0 |
| 梅瓶 | 7 | 5 | 4 |
| 葫芦瓶 | 9 | 7 | 0 |
| 红色碗 | 0 | 14 | 2 |

第二批诊断标签保存在 `support_diagnostic_labels.jsonl`，未继续用来调参或训练。该抽查只涉及已入选结果，不测召回率，也不是独立人工金标准。

## 结论与限制

当前不足以替换默认算法。青花瓷出现可用的改善信号，其他三类全库精度仍不合格；红色碗的 dev 只有 1 个正例、test 没有正例，召回率不可计算。青花瓷 dev 的负例全部是颜色不可观察项，缺少可见颜色的困难负例，校准也不充分。

关键问题是全局图像向量会把颜色、轮廓、类别混在一起。正例相似度检查抑制了大量跨类别误收，但不能代替每项条件的证据。后续需要补充同形异色、同色异形、瓶罐、颈饰/双腹等反例，训练能分别判断主体颜色、器形和局部结构的判别器，并预先冻结新的全库代表性验证样本；不应继续反复调整这 36 项测试的阈值。

所有标签均为 root AI 图片/馆藏辅助判断，非人工金标准。`葫芦瓶` 与 `double gourd` 对三节器物的范围有差异，`met_42358` 保留歧义且未用于训练。这轮仅检验四组查询的计划一致性，不能推论所有中英文自由查询都正确。

线上代码、默认算法、原始 720 项标签均未覆盖；新模型 `release_approved=false`。没有推送 GitHub。

验证：后端 589 passed / 13 skipped；Ruff 全 src/tests/scripts 通过。已额外核验第二阶段每条训练记录均来自原 train、无留出器物组/图片哈希交叉，原 720 项冻结输入哈希保持一致。验收收据为 `verification.json`。

## 复现与文件

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_targeted_boundary_v10.py
# labels.jsonl 是已完成的看图记录；准备脚本不会自动生成标签。
PYTHONPATH=src .venv/bin/python scripts/run_targeted_boundary_v10.py
PYTHONPATH=src .venv/bin/python scripts/audit_targeted_gallery_v10.py
PYTHONPATH=src .venv/bin/python scripts/run_targeted_support_v10.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_targeted_boundary.py tests/test_assisted_review.py
```

- `protocol.json` / `followup_protocol.json`：两阶段固定方法及输入哈希。
- `candidates.jsonl` / `labels.jsonl` / `review_manifest.json`：候选、逐项判断及复核来源。
- `report.json` / `followup_report.json`：模型与阈值成绩，包括失败与缺样项。
- `gallery_diagnostic_labels.jsonl` / `support_diagnostic_labels.jsonl`：两轮全库命中诊断。
- `support_diagnostic_summary.json`：最终不部署的证据。
- `artifacts/targeted_boundary_v10/`：图片联系表、原始 VQA 分数、模型权重、逐图 Yes/No；第二阶段在 `support_followup/`。
