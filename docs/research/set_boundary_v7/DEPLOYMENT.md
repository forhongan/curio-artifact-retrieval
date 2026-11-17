# v7 条件边界历史部署说明

当前默认已在 2026-11-16 改为 [严格逐项筛选](../strict_boundary_v8/README.md)。以下记录 v7 当时的部署；现在复现需显式传入 `method=condition_boundary`，`verify_condition_api.py` 已使用此参数。

2026-11-15，用户已明确授权替换默认算法。本地项目 `curio-set-boundary` 的集合检索现使用 `condition_boundary`，工作分支为 `codex/set-boundary-experiments`。尚未提交或推送 GitHub，未部署到远端服务器。

## 使用

在当前机器项目根目录运行 `bash scripts/run_local.sh`，然后打开 http://127.0.0.1:8000，选择“筛选匹配集合”。检索设置中默认显示“条件匹配（默认）”。已有 Windows 环境仍可使用项目原有启动流程，并配置 `CURIO_ENCODER=siglip`、`CURIO_INDEX=artifacts/siglip_adapted`、`CURIO_PROBE_WEIGHT=0`。

```json
{"query":"绿色瓶或花瓶","top_k":24}
```

向 `POST /api/search/set` 发送以上请求，省略 method 即使用新默认。显式传入 `dev_cutoff`、`dev_color`、`dev_calibrated` 可以比较旧方法。已有 `CURIO_SET_PROBES` 环境变量会继续选择专用探针路径；使用本默认时不要设置该变量。

API 返回全部接受项，`top_k` 只截断 preview。新方法按支持分数排序，`score` 和 `support_score` 是未校准的逻辑回归支持分数；`cosine` 保留原始相似度。它们不是“正确率”。

## 已接入的行为

- 将自由文本映射到有限条件词表；40 个已有中英文查询的条件计划与离线实验一致。
- 无法解析的剩余文本、否定、复杂或关系、多器型及未支持专业器型使用完整原句的三特征模型回退。API 的 `effective_model` 和 `fallback_reasons` 明确报告，页面显示提示。
- 年代、工艺、窄口等未支持条件显式列入 missing。不会把窄口偷偷等同于细长颈，也不会声称视觉已核实年代。
- 运行时校验图库指纹、图库大小、文本模型 ID/revision、条件词表、特征顺序、模型维度和文本缓存哈希。不匹配时返回服务不可用。
- 文本评分缓存有容量上限和锁；候选标题、材质、年代等元数据不参与条件分数。
- 修复已有索引清单中 Windows 反斜杠 adapter 路径在 macOS 上的加载问题。

## 在线与离线数值一致性

初次验收中，英文“盘或碟”有一张图片 `met_49579` 的入选不同：离线支持分数约 0.214742，重新编码的在线分数约 0.214570，阈值为 0.214689。原因是发布包文本向量与当前设备重新编码结果有细小差别，经标准化分类器放大后影响临界样本。

修复采用正常的文本向量复用：随代码附带冻结开发查询及条件提示词的文本向量，绑定其哈希；这些缓存不包含图片标签、候选图分数或答案集合。对新查询、新组合提示词仍运行匹配编码器。阈值没有为测试结果调整。

`condition_boundary_v7.json` 和 `condition_text_v7.npz` 放在 Python 包目录，避免被项目全局 `data/` 忽略规则遗漏。可运行 `python scripts/export_condition_boundary.py` 从已验证研究资源重新导出。

## 验收证据

| 检查 | 结果 |
|---|---|
| 真实 HTTP、默认 method、40 个中英文查询 | 全部通过，入选 ID 集合与离线逐项相同 |
| 全库处理与 preview 分离 | 4,692 候选；preview=1 时 results 仍完整 |
| 后端测试 | 530 passed，13 skipped；另有标注保护测试见后续验收记录 |
| Ruff | src、tests、scripts 通过 |
| 前端 | TypeScript/Vite 构建通过 |
| 浏览器 | 新默认、760 项结果的第二页、历史条件提示、否定回退、旧方法切换均通过 |
| 手机宽度 390px | 无横向溢出，无页面 JavaScript 异常 |

真实 HTTP 证据在 `runtime_verification.json`，浏览器检查在 `ui_verification.json`。截图位于 `artifacts/frontend_v7/`。

原浏览器控制接口两次连接超时，实际页面验收改用临时 Playwright 加系统 Chrome 完成。临时测试依赖没有加入产品依赖。

## 适用范围

新模型在已有意图上宏 F1 为 EN 0.6604、ZH 0.5536；中文留出意图结果仍不稳定。切换默认是用户批准的工程选择，不改变研究结论，也不等于准确划清任意查询的全库边界。

下一阶段材料与已观察到的中文错误见 [FOLLOWUP.md](FOLLOWUP.md)。
