# 当前状态 · 2026-11-08 · relevance v5.1

以用户本轮要求为开发范围：修订评测、测试组员模型和大图库、视觉辅助消融、改进全库集合边界、外部 API 参考图到图图检索、完成课程技术稿。本轮没有调用子代理。

## 入口

- [本轮结论与复现](RESEARCH_V5_RESULTS.md)：先读此文件。
- [评测定义与边界](RELEVANCE_V5.md)：200 张核查、60 dev / 140 test、20 意图，4,000 条规则衍生判断；不是专家金标准。
- [组员模型审计](TEAMMATE_MODEL_AUDIT.md)：真实权重、训练来源、10,200 图库和评测缺陷。
- [英文技术稿](TECHNICAL_REPORT_DRAFT.md)；排版 PDF：`output/pdf/Curio_Technical_Report_Draft.pdf`。已逐页检查，需小组补充实际成员贡献及指定 peer review。

## 可用算法与默认选择

文本排序继续使用 `artifacts/siglip_adapted` 的 SigLIP2 视觉 LoRA、`image_only`、属性探针权重 0。图库 4,692；候选分数依赖像素向量，不读取候选馆藏文本进行匹配。模型和相关预处理不变时重用缓存，不逐次重编码图库。

`POST /api/search/set` 新增 `dev_cutoff`、`dev_calibrated`、`dev_color`。它们给全库作 yes/no 决策，`results` 是完整接受集合，`top_k` 只限制 `preview`。原 `adaptive_evidence` 保留为对照。新阈值只在 60 dev 图上选择；不能把逻辑回归输出叫正确率，也未证明未见查询泛化。

`POST /api/search/reference/dynamic` 支持 `mode=qualified`：V&A / MIA / AIC / Cleveland API 文本搜索，核查已识别的来源条件、下载解码、图像大小、同物与重复隔离，再将实际参考照片匹配本地像素图库。没有合格参考明确返回空；保存来源支持与获取失败。五种聚合为 single_best / uniform_mean / max / diverse / query_aligned。前端默认新 API 合格参考路径；API 默认 direct 兼容旧调用，命令行也可显式选择 qualified。AIC 受 403 限制；没有声称已接通故宫 API。

HSV 可作选择性的文本和集合辅助。DINO 与前景颜色在控制的图图试验中有小幅排序增益，但 P@5 未同步改善。形状和纹理未证实稳定收益；去背景保留失败回退与可选使用。不默认硬预排除，因为既有实测会丢失相关图且缓存排序本身很快。

## 验证与产物

- 后端完整测试：510 passed，10 warnings，见 `artifacts/relevance_v5/pytest_final.log`。
- Ruff 全 src/tests/scripts 通过；前端 TypeScript/Vite 构建通过。最后展示来源字段修正后，3项动态API回归再次通过；浏览器实际显示8张参考、来源署名与SigLIP2+DINO融合，截图 `artifacts/relevance_v5/browser_acceptance.png`。
- `artifacts/relevance_v5/api_acceptance.json`：三个集合方法均与独立单查询推理重放一致；8 张合格参考、4,692 图排序和 8 条图片路由通过。验收是 17 次缓存命中、0 次新联网请求；前期采集另有实际联网记录。
- 单条/批量 FP16 文本编码余弦约 .99999988；HSV 集合有 1 张阈值附近图的判定不同，已记录，没有修改冻结标签或阈值来消除差异。
- `artifacts/relevance_v5/results.json`、`scores.npz`、`thresholds.json`、`paired_bootstrap.json`：修订核心上的原始查询对照。
- `artifacts/relevance_v5/team_workflow/`：组员原搜索服务的隔离运行，两个模型 × 两个图库 × 三条工作流 × 中英文。
- `artifacts/relevance_v5/image_components/`：正确 dev 参考到独立 test 的图图消融。
- `artifacts/qualified_reference_v5/qianlong/`：8 张真实 V&A 参考及14张目标图读图/来源核查；同一查询文本2/5，参考均值3/5，仅为单案例。

本机 8000 服务由本轮启动（当前 PID 947540，日志 `artifacts/relevance_v5/server_final.log`），环境为生产 SigLIP2-LoRA/冻结图库、离线底模。`scripts/run.ps1` 是日常运行入口。未 commit/push/deploy，新增研究和 PDF 未重新打包进旧 Release。

## 已知限制与历史

检索质量穷举真值仅覆盖 140 张 test 核心；不能报告全库召回率。20 个同意图的 dev 校准不等于新查询评测。AI 与来源辅助判断、联系表分辨率、历史 test 重用、专家日期不可由外观确证均已披露。10,200 图库只做了可判子集对照与四查询结果池观察，未给全部新增对象自动造负例。

旧状态和旧稿保存在 `history/docs_before_relevance_v5/`，初版 v5 判断保存在 `history/evaluation_v5_initial/`。它们用于溯源，不能覆盖本文件；尤其旧的“只做API核验/已暂停”不代表本轮已授权开发范围。保持旧产物路径以免破坏研究引用。
