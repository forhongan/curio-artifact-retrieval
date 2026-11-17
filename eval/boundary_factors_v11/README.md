# 分项边界 v11 · 2026-11-17

新增 96 个此前未复核的器物组，在模型评分前冻结 48 dev / 48 test；助手查看 8 张联系表、馆藏标题/材质及 2 张放大图后写入标签。原 source_split 保留，其中部分 source-train 器物被本轮留出，未进入分项判别器拟合。所有标签为 AI 辅助判断，不是独立人工金标准。

从此前 source-train 复核记录建立 139 条训练分项标签：主体红色、碗形、青花外观、梅瓶轮廓、双腹束腰。加入留出组分项标签共 403 条。红色碗要求红色与碗形同时通过，四组中英文输入共用相同分项计划；不支持的自由查询明确报错，不能生成伪造的全库 No。

每个分项读取正向描述与反例描述的本地 Qwen VQA 分数，以及旧前景颜色/口径特征；StandardScaler 与 LogisticRegression(C=1) 仅拟合训练组，阈值只在 dev 选择。黑白照片不能凭馆藏文字恢复颜色，缺失模型输出是错误，不作为 No。

原始新留出测试有 48 项、15 个正例：

| 方法 | 命中 | 误收 | 漏收 |
|---|---:|---:|---:|
| 原 VQA，0.5 | 10 | 7 | 5 |
| 原 VQA，dev 校准 | 9 | 7 | 6 |
| 分项模型 | 9 | 2 | 6 |

分项模型的葫芦瓶为 5 TP / 0 FP / 1 FN；梅瓶为 1 TP / 0 FP / 5 FN；青花瓷为 3 TP / 0 FP / 0 FN。红色碗测试无正例，有 2 个误收，因此不能报告其召回率。

另对 66 项旧全库错误/正例作回归：29 TP / 7 FP / 4 FN。此组是已知诊断案例，不是新的独立测试。黑白红釉碗 met_46998 在本轮打开原图后标为颜色不可观察；只保留可见红碗 met_47312 为正例。

发现重要失败：旧前景规则在 26 张颜色训练图片中仅 3 张有效（2 正例、1 负例）；灵活的线性权重把稀少的颜色特征学错，出现视觉模型明确“不红”而学习头仍高分通过的棕色碗。后续 v12/v13 保留本目录结果，单独记录探索，不回写这些原始成绩。

`report.json`、`predictions.jsonl` 保留原始结果；`stress_report.json`、`stress_predictions.jsonl` 保留回归。`artifacts/factor_boundary_v11/` 包含全部评分、模型与只读图片报告。新版进展见 [v13](../boundary_surface_v13/README.md)。

复现：`scripts/prepare_factor_boundary_v11.py` 只准备候选，不生成标签；`scripts/run_factor_boundary_v11.py` 完成评分/训练/校准；`scripts/verify_factor_boundary_v11.py` 仅做旧案例回归；`scripts/export_factor_report_v11.py --version 11` 导出只读图片报告。
