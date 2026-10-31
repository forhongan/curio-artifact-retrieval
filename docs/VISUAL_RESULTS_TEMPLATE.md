# 视觉主结果模板与生成说明

本文件是汇总结构说明，不包含实验数值。真实主结果由以下命令生成：

```powershell
.\.venv\Scripts\python.exe scripts/summarize_visual_results.py --require-complete
```

默认输出：

- `docs/VISUAL_RESULTS.md`：视觉主实验、全部变体和正负结果。
- `docs/visual_experiment_evidence.json`：原始 JSON、读取文件 SHA256、冻结选择、支持度、系统验收和缺失项。
- `artifacts/evaluation/image_audit/failure_shortlist.json`：固定六个不同语义查询及目标 aliases、最差语言前五 ID/alias/显式等级。

不加 `--require-complete` 时允许生成明确标记的 partial 报告；缺失项不填零。
现有结果哈希、图库、查询数或 AI 标签来源不一致时直接拒绝输出。
可用 `--root` 指向隔离的临时 fixture 根目录验证脚本，不访问真实 test 图。

## 主报告固定结构

1. **协议和来源**：60 图、24 个语义意图、中英分别报告；AI 图像标签与 AI 相关性判断，不是真人 gold。两个语言不是独立的 48 个语义查询。
2. **模型和 dev 选择**：CLIP、SigLIP、弱元数据 caption LoRA、AI 视觉 caption LoRA；production image_only 与单独的正权重 head ablation 分开；保留所有 dev 候选。
3. **配对测试**：四个 `visual_paired_*_test.json` 的 gallery、unique captions、R@1/5/10、MRR；配对 caption 为 AI 盲图描述。
4. **训练及负结果**：random/hard 的 baseline dev、best dev、最佳 epoch、采样、caption 来源、是否真实超过 baseline；false 不隐藏。
5. **自由查询**：四模型 baseline/translated、selected_test、SigLIP 基础与细节 head ablation；五个指标全部显示 95% CI，按语言列出。
6. **probe 来源和支持度**：metadata material/type 与 AI colors/shape/motifs 分开；六组 artifacts（`siglip`、`vision`、`siglip_adapted`、`vision_metadata_adapted`、`siglip_visual_adapted`、`vision_visual_adapted`）按三个真实特征空间报告；test F1/AUC、known/positive support、unknown 排除、选择的 C、实际应用和 fallback/coverage。production 可从三个SigLIP空间各自的base/detail头与四个权重共24种image_only候选选择；base-vs-detail ablation 固定在基础 SigLIP 空间，使用另行冻结的正权重。
7. **成对差值**：`image_audit/comparisons/*/comparison.json` 的所有指标、语言和正负方向，以及 Holm 校正，不只展示显著收益。
8. **失败和系统证据**：固定失败短名单；系统验收若存在则显示并保留原始 JSON，pilot 与当前配置验收分开。

## 失败名单与后续看图

每个 semantic qid 取 selected test 英文和中文 nDCG@10 的最小值，再按
`(minimum_nDCG, qid)` 排序固定选前六个；同一 qid 不因语言重复出现。
最差语言并列时英文先。目标 aliases 是最高 grade 的全部已判匹配，不把
某个相关 ID 冒充查询作者的原目标。缺失 qrel 不能补成零。

脚本把 `error_cause` 留为 null，`manual_analysis_status` 标记为待看图。
汇总不自动猜测“颜色混淆”“纹饰错误”等原因，也不据此改参数。
确需事后检查时可加 `--failure-cards`，从既有盲图 sheet 裁出一个最高等级
目标和前五结果，保留盲编号、名次与等级，供主代理真正看图。
默认不读取任何 test 图片。模型、epoch、C 和重排权重在查看 test 结果前冻结。

元数据银标结果由原 `scripts/summarize_results.py` 生成并作为
`docs/METADATA_RESULTS.md` 附录；不要把其数值填进视觉主表。
