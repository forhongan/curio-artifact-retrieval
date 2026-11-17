# 单调证据实验 v12 · 未采用全局替换

基于 v11 的失败，移除覆盖不足的前景特征，学习正向 VQA logit 与负向反例 logit 的非负权重，保证更强反对证据不能提高通过分数。颜色还要求正向描述得到 Yes 倾向、反例描述得到 No 倾向。训练数据和分组不变，未使用测试标签拟合。

这是查看 v11 测试后的探索回放：48 项原测试变为 5 TP / 0 FP / 10 FN；66 项旧错误回归为 24 TP / 5 FP / 9 FN。虽然消除了部分误收，召回下降明显，未作为全局替换。v13 仅借用其红色语义证据约束，其他器形分项保留 v11。

方法、输入哈希见 `protocol.json`；逐项结果见 `predictions.jsonl`、`stress_predictions.jsonl`；模型在 `artifacts/monotone_boundary_v12/heads.json`。所有 `release_approved=false`，没有默认算法变更。
