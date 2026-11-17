# 全库 yes/no 边界实验 v7

v7 于 2026-11-15 接入默认检索；2026-11-16 默认已进一步改为严格逐项筛选，见 [当前严格版](../strict_boundary_v8/README.md)。显式 `method=condition_boundary` 可回放本目录实验；v7 冻结研究指标保持不变。
原始 v5 标注和历史结果保留；v7 当时的部署验收见 `DEPLOYMENT.md`。

## 方法

1. 对原始分数做精确 PR 扫描。相同分数必须一起入选，允许空集合，未知标签不当负例。
2. 用查询本身的结构化条件构造条件描述，以及每次只替换一个条件的对照描述。
3. 使用固定 SigLIP2 文本塔和 production LoRA 图像向量计算证据，不读取候选藏品的标题、年代或材质。
4. 对比三种固定模型：原始三特征逻辑回归；九特征条件模型；将 grade=1 部分匹配负例权重设为 2 的条件模型。
5. 只在 60 张 dev 图上训练。三折按器物组交叉拟合得到 out-of-fold 分数选阈值，再在全部 dev 上重拟合；140 张 test 只用于计分。
6. 另做 leave-one-intent-out：每次从边界训练和选阈值中排除一个完整意图，在该意图的 test 图片上计分。中英文共享意图分组。

九个特征是整体相似度、通用提示词差值、全库标准化相似度、条件分数均值/最小值、对照差值均值/最小值、支持条件比例、条件数。困难负例实验是固定加权，不是新增人工标注或全模型微调。

## 复现

先按 `docs/RESOURCES.md` 恢复 v2 的 artifacts 和 models-siglip，再按 `docs/SHOWCASE_RELEASE.md` 恢复 v6 research/runtime。先 v2 后 v6，避免旧资源覆盖新的冻结文件。

最小必需输入：

- 仓库内 `eval/relevance_v5/{freeze,queries,items,qrels}`；
- `artifacts/relevance_v5/{scores.npz,thresholds.json,text_vectors.npz}`；
- `artifacts/siglip/item_ids.json`（冻结评分矩阵的行顺序）；
- `artifacts/siglip_adapted/{item_ids.json,image_embeddings.npy}`；
- `models/huggingface/hub` 中固定 revision 的 SigLIP2 底模与处理器。

在项目根目录、项目支持的 Python 3.11–3.12 环境中运行：

```bash
python scripts/run_set_boundary_v7.py --diagnostics-only --output artifacts/set_boundary_v7_diagnostics
python scripts/run_set_boundary_v7.py
python -m pytest tests/test_condition_boundary.py tests/test_relevance_v5.py tests/test_set_boundary.py -q
ruff check src/curio/retrieve/condition_boundary.py src/curio/evaluate/boundary_diagnostics.py scripts/run_set_boundary_v7.py tests/test_condition_boundary.py
```

`--root` 可指定另一个完整资源目录。默认离线加载本地模型；不必重新编码图库。
脚本检查冻结标注哈希、图像/查询行顺序、器物组拆分、文本塔一致性，以及图像向量重算分数与冻结矩阵的一致性。
文本缓存绑定提示词、图像向量和模型 revision；相应输入变化后会生成新缓存。

## 输出

默认输出目录是 `artifacts/set_boundary_v7/`：

| 文件 | 用途 |
|---|---|
| `RESULTS.md` / `results.json` | 全部对照、逐查询指标、视觉/专业查询分层结果与意图聚类 bootstrap |
| `diagnostics.json` | 精确 PR 曲线、AP 和事后最优阈值，仅供诊断 |
| `pr_curves_en/zh.png` | 中英文逐查询 PR 曲线 |
| `score_overlap_en/zh.png` | 正负例相似度分布 |
| `feature_manifest.json` | 所有提示词、未支持条件、特征顺序、哈希与一致性检查 |
| `models.json` | 可移植的标准化参数、逻辑回归系数、阈值及折分 |
| `scores.npz` / `selected_ids.json` | 新模型对全库每个候选的分数及完整入选集合 |
| `annotation_queue.csv` | 每个中英文查询最接近新阈值的 10 个未标注候选，共 400 行 |

`support_scores` 可以重放模型，但调用方必须检查 `models.json` 绑定的特征 manifest、图像索引和特征顺序。运行时已接入自由文本 API；有限规则解析不了的表达会显式回退到同版本整体分数模型。

## 解释限制

- Test oracle 使用测试答案寻找每个查询的最好阈值，只表示这批分数的可分性上限，不能用于正式预测。
- 已有查询的 `conditions` 是人工定义的结构化查询输入；运行时规则解析已对这 40 个表达验证一致；任意新表达的解析质量仍未获得独立标注评估。
- 缺失的历史/工艺条件会记录为 missing；这不是自动拒答机制。模型仍输出二元支持判定，不能称为“已确认年代”。
- 颜色、纹饰可能共存，对照描述不一定互斥；只把差值作为可学习特征，不当成硬规则。
- 交叉拟合阈值不等于独立概率校准，也不提供 precision/recall 的统计保证。重拟合后分数可能有变化。
- LOIO 只检验边界模型向留出意图迁移。编码器、词汇表和既往模型选择并未对该意图完全隔离。
- 核心标注小、含 AI 辅助和历史测试暴露。bootstrap 按 20 个意图聚类，中英文先配对平均，不能把 40 个表达当独立查询；区间不涵盖标注错误或多重比较。
- 没有全库完整真值，因此召回率只适用于 140 张 test 图。全库入选数量只是输出规模，不是准确性证据。

## 后续人工标注

`annotation_queue.csv` 的 `human_grade` 留空。逐图检查所有查询条件：3=全部满足；1=只满足部分且存在明确矛盾；0=不相关；-1=证据不足。保留 notes，并让第二位标注者复核分歧。

该队列偏向边界样本，适合新增训练数据，不可作为无偏测试集或用于估计全库召回率。下一轮正式评测需要预先固定的新查询意图，并按器物组隔离新的独立测试样本；正例稀少时可优先对少量查询完整标注全库。不能标完队列后重复调参，把既有 test 当成全新证据。

预先固定的实验设置见 `PROTOCOL.md`。新一轮假设需要新版本协议，不回改本轮结果。
