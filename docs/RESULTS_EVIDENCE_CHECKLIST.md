# 最终结果证据清单

本清单用于完成 `docs/RESULTS.md`，不代表下列实验或人工工作已经通过。
最终数字只从冻结实验产物提取；缺失项明确标待完成，不能用零值、规划书中的示例数字或 pilot 结果代替。
正式版本为 `visual-v2-grouped-all-image-candidates`。v1 存在对象/部件跨 split 泄漏，其训练、dev 选择和已有结果均作废，保留在 `archive/v1_invalid_split/`，不能引用为正式结论。

## 主结果入口

- [ ] `docs/RESULTS.md` 明确区分视觉主实验、元数据辅助对照、系统验收与 pilot。
- [ ] 链接机器生成的 `docs/VISUAL_RESULTS.md`、`docs/visual_experiment_evidence.json` 与 `docs/METADATA_RESULTS.md`。
- [ ] 完整保留负结果、不显著比较、未知属性、不可用 AUC、回退与适用边界。
- [ ] 声明 AI 盲图标签/相关性判断和元数据 silver 均不是真人 gold；中英表述不是双倍独立语义样本。

## 数据、监督与冻结

- [ ] 原始来源、逐对象许可、实际可用图数、去重和对象分区证据，引用 `docs/DATA.md`、`data/snapshots/collection_v2.jsonl`、`data/licenses.csv`、`data/processed/audit.json` 与 `grouped_freeze_v2.json`。正式 4,692 项：train 3,474 / dev 568 / test 650。
- [ ] 审计记录 raw/processed/thumb 全量解码和实际 SHA、官方 rights、中国来源、起年 < 1912；162 个年代区间跨 1912 和 3 对同 split 处理图重复作为 warning 如实披露。关联组和有效监督集之间无跨 split 泄漏。
- [ ] `eval/visual_labels/annotations.jsonl`：正式 338 条有效 AI 盲图标签，train 218 / dev 60 / test 60；图像 SHA、split、annotator 与 AI 来源齐全；unknown 不转换成负例。原始 360 条（240 / 60 / 60）是保留的历史事实，22 条原 train 标签因对象族迁入 heldout 而排除，证据在 `excluded_training_labels.json`。
- [ ] dev/test 的 `queries.jsonl`、`qrels.jsonl`、`gallery_ids.json`：各自 60 图、24 个中英语义意图、显式完整 0–3 级相关性。
- [ ] `artifacts/visual_selection.json`：全部 dev 候选、image_only 生产选择、独立正权重 head ablation、两种 adapter 路径和模型文件哈希；配置 SHA 必须对应正式 v2 和有效 338 条标注。
- [ ] 生产候选包含 `siglip`、`siglip_adapted`（元数据 caption LoRA）、`siglip_visual_adapted`（AI 盲图 caption LoRA）三个图像空间；各有基础/细节两组 probe × 权重 `0, 0.05, 0.15, 0.3`，共 24 个 image_only dev 候选。四模型的 baseline/translated 八个对照完整保留，overall best 单独报告，不能替代生产选择规则。
- [ ] `artifacts/visual_test_freeze.json`：冻结选择文件 SHA、配置及测试输入 SHA；与当前选择和所有测试报告贯通。
- [ ] `artifacts/visual_experiment_status.json` 与 `logs/visual_experiments/`：命令阶段、耗时、退出状态；任何早期工程路径曾计算测试指标的事实如实披露，不声称其未发生。

## 训练与检索质量

| 必需证据 | 产物与检查 |
|---|---|
| 两种视觉 caption LoRA | `artifacts/training/visual_lora_{random,hard}/run.json`、`history.jsonl`、`best_adapter/`；使用 v2 的 218 张有效 train 图，报告 baseline/best dev、最佳 epoch、训练图数、caption 来源、是否真实改善。hard 是冻结 teacher 图像近邻；v1 训练产物不能续用。 |
| 四模型 caption 配对测试 | `artifacts/evaluation/visual_paired_{clip,siglip,siglip_adapted,siglip_visual_adapted}_test.json`；R@1/5/10、MRR、图库数、unique captions；标签为 AI 图像 caption。 |
| 四模型自由查询对照 | `artifacts/evaluation/image_audit/{model}_baseline_translated_test/{metrics.json,per_query.jsonl,rankings.jsonl}`；英文和中文分别报告 baseline、translated。 |
| 冻结生产选择 | `image_audit/selected_test/` 三类报告；模型索引、probe 文件、权重与 dev 冻结选择严格一致。 |
| head 消融 | `image_audit/siglip_siglip_test/` 与 `siglip_vision_test/`；都使用基础 SigLIP 空间及独立冻结的同一正权重。 |
| 全部检索指标 | nDCG@10、P@5、AP@10、Recall@20、MRR@10，各语言给 95% bootstrap CI；grade≥2 为二值相关，nDCG 增益为 `2^grade−1`。 |
| 实际图像评分依赖 | image_only 的 metadata/BM25 通道为空；报告 probe 实际应用数、covered/unknown、降级和有效变体。 |

## 分类头与统计比较

- [ ] 六组 `artifacts/probes/{siglip,vision,siglip_adapted,vision_metadata_adapted,siglip_visual_adapted,vision_visual_adapted}/` 各自保留 `report.json`、`ontology.json`、所有训练 NPZ 与独立 `test_report.json`；每组只能匹配所属 v2 图像空间，不能跨 adapter 混用。
- [ ] dev 使用 `--defer-test`；test 使用 `--evaluate-test`，只做冻结 NPZ 预测，不再 fit 或选 C，训练报告 SHA 不变。
- [ ] 基础器型/材质为元数据弱监督；颜色/器形/纹饰为 AI 盲图监督，各自说明训练支持度和标签来源。
- [ ] AI 属性逐类报告 test known、positives、F1、ROC AUC；单类支持时 AUC 留空；基础头报告 accuracy、balanced accuracy、macro F1、eligible/excluded。
- [ ] `image_audit/comparisons/{clip_vs_siglip,metadata_lora,visual_lora,translated_vs_selected,base_vs_detail_heads}/comparison.json` 五组齐全。
- [ ] 成对比较保留五指标×两语言的全部正负差值、CI、p 和每组 Holm 校正；不只挑显著收益。

## 失败、系统与可复现性

- [ ] `image_audit/failure_shortlist.json` 固定选择六个不同 semantic qid；按 selected test 的中英最小 nDCG 排序，不人为挑成功案例。
- [ ] 目标 aliases 为最高 grade 的全部匹配图，前五结果有显式 qrel 等级。图像错误归因必须在冻结后实际看图并记录，自动汇总不能代写归因。
- [ ] 当前生产配置的 `artifacts/system_acceptance.json` 或明确命名的新文件：实际 API、图像解码、严格空筛选、输入验证、无本地路径、有限且不重复结果。
- [ ] 系统验收 health 的 index version、probe weight/tasks 与冻结生产选择一致；记录运行硬件、模型冷/热状态、请求数、workers、P50/P95 和吞吐。
- [ ] 验收 `runtime_config` 的 data/index manifest/probe 文件 SHA 和权重匹配冻结生产选择；`request_records` 包含每条 query、variant、request ID、wall_ms 及并发请求。运行前后配置及进程一致；汇总 scope 为 `current_frozen_production`，旧报告或来源不匹配不能证明当前配置通过。
- [ ] `scripts/system_check.py` 的当前并发字段是 `concurrent_image_only`；pilot 结果单独列出，不能证明当前配置通过。
- [ ] README 命令、依赖锁、测试输出、应用启动和真实浏览器流程证据可重现。
- [ ] AI 使用范围、验证方式、实际组员贡献如实记录；真人标注一致性、用户研究和组员分工未发生时不能补写为已完成。

## 汇总命令

在全部 test 阶段完成后执行：

```powershell
.\.venv\Scripts\python.exe scripts/summarize_results.py
.\.venv\Scripts\python.exe scripts/summarize_visual_results.py --require-complete
```

需要事后失败图卡时再显式加 `--failure-cards`；该开关会读取测试盲图。
`--require-complete` 证明预期实验 JSON 齐备及来源检查通过，并不自动证明系统验收、人工失败归因或最终课程交付已经完成。
