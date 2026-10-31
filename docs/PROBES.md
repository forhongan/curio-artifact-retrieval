# 视觉属性线性探针

`curio.represent.probes` 在冻结图像嵌入上训练轻量 Logistic Regression，用于实验性估计器型和材质属性。标签来自清洗后的馆藏字段，只是弱监督标签；输出在产品中必须称为“弱监督视觉分类器预测”，不能称为图像事实或人工确认。

正式数据是 `data/snapshots/collection_v2.jsonl` 的 4,692 项：train 3,474 / dev 568 / test 650。对象整体、部件、重复图像及官方关联记录按连通组隔离。v1 的对象/部件跨 split 泄漏使其训练、选择及结果作废；历史产物保留在 `archive/v1_invalid_split/`，不用于正式 v2 结论或继续训练。

```powershell
python -m curio.represent.probes --data data/snapshots/collection_v2.jsonl --index artifacts/siglip --output artifacts/probes/siglip --defer-test
```

训练只接受 `train/dev/test` 三个显式 split，并复用 `validate_object_splits` 检查重复图像和跨 split 泄漏。索引必须包含 `item_ids.json`、`image_embeddings.npy`、`manifest.json`；程序核对覆盖范围、矩阵维度，以及数据和向量的 SHA-256。训练类别要求非空 canonical label 且 train 支持数至少 10。基础 `material` 与 `object_type` 是多分类任务；基础脚本不从馆藏缺失字段制造颜色/纹饰负类。

候选 `C` 仅在 dev 集按 macro-F1 选择（`0.1, 1, 10`）。正式dev使用 `--defer-test`，`report.json` 中test为空；选择冻结后通过 `--evaluate-test` 只预测保存的NPZ，另写 `test_report.json`，不重新拟合或改写训练报告。每个任务输出 `material.npz` / `object_type.npz`，只包含 `weights`、`bias`、`classes`，推理使用 `allow_pickle=False`；报告记录 support、类别、accuracy、balanced accuracy、macro-F1、混淆矩阵、provenance、输入索引哈希以及 `artifacts_sha256`（权重和本体文件哈希）。`LinearProbes` 加载时会校验这些文件，替换或损坏权重会拒绝加载；`ontology.json` 保存类别本体。

## 独立视觉细节监督

`curio.represent.detail_probes` 使用218 train / 60 dev / 60 test盲图AI标注，在颜色、器形、纹饰各属性上训练独立sigmoid分类器。unknown逐属性排除，不用背景颜色充当器物颜色；黑白照片颜色全为unknown。训练至少5正/5负，dev必须有两类，未达到支持数的属性不训练。每任务C按dev宏F1选取，固定0.5阈值，test逐属性给出known support、positives、F1和ROC AUC。该脚本同样采用defer/evaluate分阶段协议。

正式 `eval/visual_labels/annotations.jsonl` 共 338 条有效标签。原始盲图池及 AI 标注共 360 条（240 train / 60 dev / 60 test）；v2 分组后 22 张原 train 图属于 heldout 对象族，其原标注保留，合并时按真实 split 排除，记录在 `eval/visual_labels/excluded_training_labels.json`。它们不能继续作为训练监督，也不自动扩充 60 图 dev/test gallery。AI 标签不是真人 gold。

细节目录复制匹配空间的基础头。基础 SigLIP、元数据 caption LoRA 和 AI 盲图 caption LoRA 是三个不同图像空间；每个适配空间都必须重新拟合基础头和细节头，不能混用零样本或其他 adapter 的权重。正式共六组目录：

| 图像索引 | 基础器型/材质头 | 基础头 + AI 颜色/器形/纹饰细节头 |
|---|---|---|
| `artifacts/siglip` | `artifacts/probes/siglip` | `artifacts/probes/vision` |
| `artifacts/siglip_adapted` | `artifacts/probes/siglip_adapted` | `artifacts/probes/vision_metadata_adapted` |
| `artifacts/siglip_visual_adapted` | `artifacts/probes/siglip_visual_adapted` | `artifacts/probes/vision_visual_adapted` |

`scripts/run_visual_experiments.py` 使用协议 `visual-v2-grouped-all-image-candidates`，仅据 dev 比较三个 SigLIP 空间 × 两组 probe × 权重 `0, 0.05, 0.15, 0.3`，共 24 个 image_only 生产候选。元数据 caption 只影响训练监督；`siglip_adapted` 的 image_only 推理仍只使用图文嵌入和匹配空间的图像探针。CLIP 与四模型 baseline/translated 结果保留为对照，overall best 对照与生产选择分开报告。六组头在 dev 阶段均 defer test；最终冻结后才分别执行只预测的 `--evaluate-test`。

单标签 `object_type=lid` 只支持独立盖子正向意图，不能判断另一器物带盖或无盖。image_only 对盖子部件条件报告 `unknown` / `component_presence_not_trained`，不计探针分数或 lid 证据；不能把 `1-P(lid)` 解释为无盖。

```python
from curio.represent.probes import LinearProbes
probes = LinearProbes("artifacts/probes/siglip")
pred = probes.predict(embeddings)
# pred["material"]["probabilities"]: [N, num_classes]
```

正式 v2 训练与评测的完成状态以当前实验状态文件和冻结产物为准；`VISUAL_RESULTS.md` 只能汇总来源校验通过的 v2 证据。仓库回归测试使用合成特征夹具，不依赖GPU或真实图像。
