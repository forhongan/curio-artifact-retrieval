# Curio 实验与验收入口

更新日期：2026-10-30。v2 数据、模型选择、正式视觉/元数据评测和系统验收已完成；最终数字以本页链接的冻结实验产物为准。

## 证据分层

1. [主要视觉结果](VISUAL_RESULTS.md)：AI 盲图监督、独立 dev/test 小图库自由查询、图像属性头和视觉 caption 训练。它检验图像检索，但不是真人人工金标准。
2. [元数据辅助对照](METADATA_RESULTS.md)：馆藏字段派生 caption 和 silver qrels；只衡量字段符合程度，不能替代视觉结论。
3. [数据质量与许可](DATA.md)：正式图库 4,692 件、对象分组、文件校验、来源与许可。
4. [开发与验收](DEVELOPMENT.md)：真实训练、单元测试、API、浏览器及性能验收记录。

## 冻结数据

`data/snapshots/collection_v2.jsonl`：train 3,474 / dev 568 / test 650；SHA256 `4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68`。

主实验的盲图监督是 train 218 / dev 60 / test 60。自由检索 dev/test 各 24 个意图、每个意图具有中英文表述，相关性穷举相应 60 图。主实验的分数不能声称覆盖全部 4,692 图；最终系统仍检索完整图库。

## 已完成证据

- 视觉 dev 选择、冻结配置与 test 指标见 [VISUAL_RESULTS.md](VISUAL_RESULTS.md)，保留训练无收益和零样本对照。
- 分类与细节头的独立 test 报告见 [PROBES.md](PROBES.md) 及视觉结果附录；超参数只由 dev 选择。
- 元数据辅助对照见 [METADATA_RESULTS.md](METADATA_RESULTS.md)，不替代图像结论。
- 固定最差查询的真实看图分析见 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)；分析不回改测试标签或参数。
- 最终配置下 200 次顺序和 200 次四并发 API 请求、输入校验、空 facet、路径安全、浏览器桌面/移动端验收和构建记录见 `artifacts/system_acceptance.json`、[DEVELOPMENT.md](DEVELOPMENT.md) 与项目日志。

最终课程报告、PPT、真人复核和组员贡献分配属于下一阶段。
