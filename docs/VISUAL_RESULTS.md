# 视觉主实验结果

由 `scripts/summarize_visual_results.py` 读取真实 JSON 生成；当前 v2 所需 JSON 已齐备，数值均来自冻结实验产物。

**AI 图像标注与 AI 视觉相关性审计，不是真人 gold；不报告真人一致性或独立人工收益。**

状态：`complete`。协议声明显式受限图库 **60 图**、**24 个语义查询**；每份已到报告均核对数量。中英分别报告，两种语言不是两倍独立样本。

nDCG 使用 0–3 级和 `2^grade−1`；P/AP/Recall/MRR 以 grade≥2 为相关。AP@10 分母为全部相关项。CI 在各语言内 bootstrap 语义查询，不能推断对完整馆藏或真人查询的泛化。

## 来源与冻结选择

- CLIP / SigLIP 为零样本图文基线；`siglip_adapted` 是馆藏弱元数据 caption LoRA 对照；`siglip_visual_adapted` 使用披露的 AI 盲图 caption。
- 基础 material/type heads 来自馆藏元数据弱标签；vision 在对应特征空间追加 AI 盲图 colors/shape/motifs sigmoid heads。SigLIP 零样本与视觉 LoRA 各有匹配的独立 probe artifacts，不能跨空间混用。推理输入仍为图像向量，不是查询时的元数据分数。
- `image_only` 不使用 BM25、metadata slot 或 metadata conflict 评分；显式 facet 可以按元数据筛选候选。
- 视觉主路径和正权重 head ablation 都在 dev 冻结；test 结果不反过来选模型、采样、epoch、C、查询或重排权重。

冻结时间：`2026-10-30T06:28:00.117704+00:00`；数据 SHA256：`4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68`；视觉标注 SHA256：`cb2d78cf16f08e9aea66ceff89da981dfc6e5a6ff0b556a6e7d77ae8b13a73f3`。

选中 production：`{"model": "siglip_adapted", "variant": "image_only", "probes": "artifacts/probes/siglip_adapted", "weight": 0.0, "dev_ndcg@10": 0.9175135501477796}`。
全比较系统 dev 最佳（与 production 区分）：`{"model": "siglip_adapted", "variant": "translated", "probes": null, "weight": 0.0, "dev_ndcg@10": 0.9175135501477796}`。
独立 head ablation 正权重：`0.05`；选择规则：Production: highest bilingual visual-dev nDCG@10 within the explicit image_only family; stable ties prefer earlier candidate and lower weight. Overall best across all comparison models is reported separately. Restricted AI-labeled dev gallery, not human gold.。

| dev 候选 | 变体 | probes | 权重 | 双语 dev nDCG@10 |
|---|---|---|---:|---:|
| clip | baseline | None | 0.00 | 0.4987 |
| clip | translated | None | 0.00 | 0.6079 |
| siglip | baseline | None | 0.00 | 0.8604 |
| siglip | translated | None | 0.00 | 0.8605 |
| siglip_adapted | baseline | None | 0.00 | 0.9130 |
| siglip_adapted | translated | None | 0.00 | 0.9175 |
| siglip_visual_adapted | baseline | None | 0.00 | 0.8814 |
| siglip_visual_adapted | translated | None | 0.00 | 0.8870 |
| siglip | image_only | artifacts/probes/siglip | 0.00 | 0.8605 |
| siglip | image_only | artifacts/probes/siglip | 0.05 | 0.8595 |
| siglip | image_only | artifacts/probes/siglip | 0.15 | 0.8552 |
| siglip | image_only | artifacts/probes/siglip | 0.30 | 0.8213 |
| siglip | image_only | artifacts/probes/vision | 0.00 | 0.8605 |
| siglip | image_only | artifacts/probes/vision | 0.05 | 0.8594 |
| siglip | image_only | artifacts/probes/vision | 0.15 | 0.8504 |
| siglip | image_only | artifacts/probes/vision | 0.30 | 0.8339 |
| siglip_adapted | image_only | artifacts/probes/siglip_adapted | 0.00 | 0.9175 |
| siglip_adapted | image_only | artifacts/probes/siglip_adapted | 0.05 | 0.9086 |
| siglip_adapted | image_only | artifacts/probes/siglip_adapted | 0.15 | 0.8801 |
| siglip_adapted | image_only | artifacts/probes/siglip_adapted | 0.30 | 0.8415 |
| siglip_adapted | image_only | artifacts/probes/vision_metadata_adapted | 0.00 | 0.9175 |
| siglip_adapted | image_only | artifacts/probes/vision_metadata_adapted | 0.05 | 0.9084 |
| siglip_adapted | image_only | artifacts/probes/vision_metadata_adapted | 0.15 | 0.8978 |
| siglip_adapted | image_only | artifacts/probes/vision_metadata_adapted | 0.30 | 0.8800 |
| siglip_visual_adapted | image_only | artifacts/probes/siglip_visual_adapted | 0.00 | 0.8870 |
| siglip_visual_adapted | image_only | artifacts/probes/siglip_visual_adapted | 0.05 | 0.8723 |
| siglip_visual_adapted | image_only | artifacts/probes/siglip_visual_adapted | 0.15 | 0.8510 |
| siglip_visual_adapted | image_only | artifacts/probes/siglip_visual_adapted | 0.30 | 0.8303 |
| siglip_visual_adapted | image_only | artifacts/probes/vision_visual_adapted | 0.00 | 0.8870 |
| siglip_visual_adapted | image_only | artifacts/probes/vision_visual_adapted | 0.05 | 0.8716 |
| siglip_visual_adapted | image_only | artifacts/probes/vision_visual_adapted | 0.15 | 0.8478 |
| siglip_visual_adapted | image_only | artifacts/probes/vision_visual_adapted | 0.30 | 0.8305 |

## AI caption 配对测试

配对标签是 AI 盲图 caption；完全相同 caption 多正例。R@K 为查询命中率。此任务与下方自由查询相关性审计不同。

| 模型 | 图库 | 独立 caption | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| clip | 60 | 60 | 0.7333 | 1.0000 | 1.0000 | 0.8547 |
| siglip | 60 | 60 | 0.9333 | 1.0000 | 1.0000 | 0.9667 |
| siglip_adapted | 60 | 60 | 0.9500 | 1.0000 | 1.0000 | 0.9750 |
| siglip_visual_adapted | 60 | 60 | 0.9667 | 1.0000 | 1.0000 | 0.9833 |

## LoRA 训练与负结果

保留 random/hard 两条真实训练记录；`improved_over_baseline=false` 不被删除。下表只描述 paired dev，不能将 dev 改善冒充自由查询 test 改善。

| 采样 | caption 来源 | 训练/dev 图数 | 最佳 epoch | baseline dev R@5 | best dev R@5 | ΔR@5 | best MRR | 优于 baseline |
|---|---|---:|---:|---:|---:|---:|---:|---|
| random | ai_image_caption | 218/60 | 4 | 0.9833 | 1.0000 | +0.0167 | 0.9611 | True |

`random` caption policy：AI image-only blind-card captions; no museum metadata or retriever rankings supplied to annotators。

| hard | ai_image_caption | 218/60 | 6 | 0.9833 | 1.0000 | +0.0167 | 0.9639 | True |

`hard` caption policy：AI image-only blind-card captions; no museum metadata or retriever rankings supplied to annotators。


## 全变体自由查询测试

每个指标均显示 mean [95% CI]；展示全部模型、变体与语言，不只展示获胜结果。

| run | 变体 | 语言 | nDCG@10 [CI] | P@5 [CI] | AP@10 [CI] | R@20 [CI] | MRR@10 [CI] |
|---|---|---|---|---|---|---|---|
| clip_baseline_translated_test | baseline | en | 0.7926 [0.7013, 0.8812] | 0.3667 [0.2750, 0.4583] | 0.7419 [0.6157, 0.8693] | 0.9931 [0.9792, 1.0000] | 0.8250 [0.6979, 0.9459] |
| clip_baseline_translated_test | baseline | zh | 0.1559 [0.0792, 0.2558] | 0.0583 [0.0250, 0.0917] | 0.1116 [0.0238, 0.2358] | 0.3601 [0.2215, 0.5087] | 0.1569 [0.0597, 0.2848] |
| clip_baseline_translated_test | translated | en | 0.8051 [0.7131, 0.8892] | 0.3667 [0.2750, 0.4583] | 0.7677 [0.6457, 0.8828] | 0.9931 [0.9792, 1.0000] | 0.8514 [0.7298, 0.9625] |
| clip_baseline_translated_test | translated | zh | 0.5030 [0.3540, 0.6509] | 0.2250 [0.1333, 0.3250] | 0.4452 [0.2877, 0.6061] | 0.7431 [0.5694, 0.9097] | 0.5250 [0.3504, 0.6969] |
| siglip_baseline_translated_test | baseline | en | 0.9076 [0.8670, 0.9424] | 0.4583 [0.3500, 0.5667] | 0.8753 [0.7932, 0.9515] | 1.0000 [1.0000, 1.0000] | 0.9306 [0.8472, 1.0000] |
| siglip_baseline_translated_test | baseline | zh | 0.7930 [0.7156, 0.8676] | 0.3833 [0.3000, 0.4750] | 0.7091 [0.5921, 0.8264] | 0.9618 [0.9132, 1.0000] | 0.8174 [0.6937, 0.9306] |
| siglip_baseline_translated_test | translated | en | 0.8968 [0.8594, 0.9301] | 0.4500 [0.3500, 0.5500] | 0.8633 [0.7896, 0.9358] | 1.0000 [1.0000, 1.0000] | 0.9306 [0.8472, 1.0000] |
| siglip_baseline_translated_test | translated | zh | 0.8091 [0.7232, 0.8852] | 0.3917 [0.2917, 0.4917] | 0.7685 [0.6456, 0.8839] | 0.9861 [0.9583, 1.0000] | 0.8789 [0.7560, 0.9792] |
| siglip_adapted_baseline_translated_test | baseline | en | 0.9152 [0.8789, 0.9490] | 0.4417 [0.3417, 0.5417] | 0.8866 [0.8146, 0.9548] | 1.0000 [1.0000, 1.0000] | 0.9583 [0.8958, 1.0000] |
| siglip_adapted_baseline_translated_test | baseline | zh | 0.8488 [0.7867, 0.9086] | 0.4083 [0.3167, 0.5083] | 0.8099 [0.7125, 0.9038] | 0.9931 [0.9792, 1.0000] | 0.8972 [0.7917, 0.9792] |
| siglip_adapted_baseline_translated_test | translated | en | 0.9151 [0.8792, 0.9494] | 0.4583 [0.3500, 0.5585] | 0.8990 [0.8254, 0.9656] | 1.0000 [1.0000, 1.0000] | 0.9583 [0.8958, 1.0000] |
| siglip_adapted_baseline_translated_test | translated | zh | 0.8078 [0.7208, 0.8892] | 0.3917 [0.2917, 0.5000] | 0.7772 [0.6588, 0.8943] | 0.9931 [0.9792, 1.0000] | 0.8556 [0.7333, 0.9653] |
| siglip_visual_adapted_baseline_translated_test | baseline | en | 0.9166 [0.8838, 0.9461] | 0.4417 [0.3417, 0.5417] | 0.8966 [0.8243, 0.9625] | 1.0000 [1.0000, 1.0000] | 0.9583 [0.8958, 1.0000] |
| siglip_visual_adapted_baseline_translated_test | baseline | zh | 0.8251 [0.7525, 0.8928] | 0.3917 [0.3000, 0.4833] | 0.7720 [0.6659, 0.8801] | 0.9618 [0.9132, 1.0000] | 0.8403 [0.7153, 0.9444] |
| siglip_visual_adapted_baseline_translated_test | translated | en | 0.9183 [0.8871, 0.9474] | 0.4417 [0.3417, 0.5417] | 0.8963 [0.8266, 0.9608] | 1.0000 [1.0000, 1.0000] | 0.9583 [0.8958, 1.0000] |
| siglip_visual_adapted_baseline_translated_test | translated | zh | 0.8290 [0.7459, 0.9004] | 0.4000 [0.2917, 0.5083] | 0.7878 [0.6700, 0.9000] | 1.0000 [1.0000, 1.0000] | 0.8833 [0.7667, 0.9792] |
| selected_test | image_only | en | 0.9151 [0.8792, 0.9494] | 0.4583 [0.3500, 0.5585] | 0.8990 [0.8254, 0.9656] | 1.0000 [1.0000, 1.0000] | 0.9583 [0.8958, 1.0000] |
| selected_test | image_only | zh | 0.8078 [0.7208, 0.8892] | 0.3917 [0.2917, 0.5000] | 0.7772 [0.6588, 0.8943] | 0.9931 [0.9792, 1.0000] | 0.8556 [0.7333, 0.9653] |
| siglip_siglip_test | image_only | en | 0.9009 [0.8615, 0.9361] | 0.4500 [0.3500, 0.5500] | 0.8654 [0.7904, 0.9374] | 1.0000 [1.0000, 1.0000] | 0.9306 [0.8472, 1.0000] |
| siglip_siglip_test | image_only | zh | 0.7760 [0.6820, 0.8636] | 0.3917 [0.2917, 0.5000] | 0.7144 [0.5970, 0.8344] | 0.9861 [0.9583, 1.0000] | 0.8303 [0.7057, 0.9435] |
| siglip_vision_test | image_only | en | 0.8994 [0.8608, 0.9350] | 0.4500 [0.3500, 0.5500] | 0.8717 [0.8010, 0.9391] | 1.0000 [1.0000, 1.0000] | 0.9306 [0.8472, 1.0000] |
| siglip_vision_test | image_only | zh | 0.7898 [0.6977, 0.8731] | 0.4000 [0.3000, 0.5000] | 0.7407 [0.6203, 0.8590] | 0.9861 [0.9583, 1.0000] | 0.8580 [0.7366, 0.9643] |

## 实际 probe 使用与降级

计数按 query-language 观测分别列出；covered 不是图像属性真值。weight=0 不算实际 probe 应用；未知模型类不当作属性不存在。

| run | 变体/语言 | 实际 probe 查询 | 降级查询 | effective variants | P50/P95 ms |
|---|---|---:|---:|---|---:|
| clip_baseline_translated_test | baseline/en | 0 | 0 | {'baseline': 24} | 40.10/54.73 |
| clip_baseline_translated_test | baseline/zh | 0 | 0 | {'baseline': 24} | 34.70/52.43 |
| clip_baseline_translated_test | translated/en | 0 | 0 | {'translated': 24} | 38.96/48.74 |
| clip_baseline_translated_test | translated/zh | 0 | 0 | {'translated': 24} | 36.83/110.55 |
| siglip_baseline_translated_test | baseline/en | 0 | 0 | {'baseline': 24} | 36.50/50.31 |
| siglip_baseline_translated_test | baseline/zh | 0 | 0 | {'baseline': 24} | 31.69/41.67 |
| siglip_baseline_translated_test | translated/en | 0 | 0 | {'translated': 24} | 39.19/57.20 |
| siglip_baseline_translated_test | translated/zh | 0 | 0 | {'translated': 24} | 41.71/48.77 |
| siglip_adapted_baseline_translated_test | baseline/en | 0 | 0 | {'baseline': 24} | 36.93/53.61 |
| siglip_adapted_baseline_translated_test | baseline/zh | 0 | 0 | {'baseline': 24} | 29.75/58.80 |
| siglip_adapted_baseline_translated_test | translated/en | 0 | 0 | {'translated': 24} | 46.78/58.94 |
| siglip_adapted_baseline_translated_test | translated/zh | 0 | 0 | {'translated': 24} | 37.56/49.23 |
| siglip_visual_adapted_baseline_translated_test | baseline/en | 0 | 0 | {'baseline': 24} | 36.17/62.47 |
| siglip_visual_adapted_baseline_translated_test | baseline/zh | 0 | 0 | {'baseline': 24} | 32.42/52.03 |
| siglip_visual_adapted_baseline_translated_test | translated/en | 0 | 0 | {'translated': 24} | 45.96/54.95 |
| siglip_visual_adapted_baseline_translated_test | translated/zh | 0 | 0 | {'translated': 24} | 35.16/50.92 |
| selected_test | image_only/en | 0 | 0 | {'image_only': 13, 'dense_rewritten': 11} | 18.24/30.78 |
| selected_test | image_only/zh | 0 | 0 | {'image_only': 17, 'dense_rewritten': 7} | 18.90/24.46 |
| siglip_siglip_test | image_only/en | 13 | 0 | {'image_only': 13, 'dense_rewritten': 11} | 20.43/29.02 |
| siglip_siglip_test | image_only/zh | 17 | 0 | {'image_only': 17, 'dense_rewritten': 7} | 18.52/24.59 |
| siglip_vision_test | image_only/en | 18 | 0 | {'image_only': 18, 'dense_rewritten': 6} | 19.59/32.09 |
| siglip_vision_test | image_only/zh | 19 | 0 | {'image_only': 19, 'dense_rewritten': 5} | 22.56/24.54 |

| run | 变体/语言 | positive covered/unknown | negation covered/unknown |
|---|---|---:|---:|
| clip_baseline_translated_test | baseline/en | 0/0 | 0/0 |
| clip_baseline_translated_test | baseline/zh | 0/0 | 0/0 |
| clip_baseline_translated_test | translated/en | 0/0 | 0/0 |
| clip_baseline_translated_test | translated/zh | 0/0 | 0/0 |
| siglip_baseline_translated_test | baseline/en | 0/0 | 0/0 |
| siglip_baseline_translated_test | baseline/zh | 0/0 | 0/0 |
| siglip_baseline_translated_test | translated/en | 0/0 | 0/0 |
| siglip_baseline_translated_test | translated/zh | 0/0 | 0/0 |
| siglip_adapted_baseline_translated_test | baseline/en | 0/0 | 0/0 |
| siglip_adapted_baseline_translated_test | baseline/zh | 0/0 | 0/0 |
| siglip_adapted_baseline_translated_test | translated/en | 0/0 | 0/0 |
| siglip_adapted_baseline_translated_test | translated/zh | 0/0 | 0/0 |
| siglip_visual_adapted_baseline_translated_test | baseline/en | 0/0 | 0/0 |
| siglip_visual_adapted_baseline_translated_test | baseline/zh | 0/0 | 0/0 |
| siglip_visual_adapted_baseline_translated_test | translated/en | 0/0 | 0/0 |
| siglip_visual_adapted_baseline_translated_test | translated/zh | 0/0 | 0/0 |
| selected_test | image_only/en | 15/24 | 0/0 |
| selected_test | image_only/zh | 19/23 | 0/0 |
| siglip_siglip_test | image_only/en | 15/24 | 0/0 |
| siglip_siglip_test | image_only/zh | 19/23 | 0/0 |
| siglip_vision_test | image_only/en | 37/2 | 0/0 |
| siglip_vision_test | image_only/zh | 41/1 | 0/0 |

## Head 标签来源与支持度

材料/器型与 AI 颜色/器形/纹饰的标签来源分开。AI detail test 的 support 是该属性已知标签图数，unknown 被排除；无双类支持时 AUC 为空，不补成零。

| probe artifacts | 特征维数 | index version | AI 标注 train/dev/test 图数 | 标注 SHA256 |
|---|---:|---|---|---|
| siglip | 768 | f94c6e2d4245f54c | 不适用/未报告 | 不适用/未报告 |
| vision | 768 | f94c6e2d4245f54c | {'train': 218, 'dev': 60, 'test': 60} | cb2d78cf16f08e9aea66ceff89da981dfc6e5a6ff0b556a6e7d77ae8b13a73f3 |
| siglip_adapted | 768 | d7fb9f1d47dafbd1 | 不适用/未报告 | 不适用/未报告 |
| vision_metadata_adapted | 768 | d7fb9f1d47dafbd1 | {'train': 218, 'dev': 60, 'test': 60} | cb2d78cf16f08e9aea66ceff89da981dfc6e5a6ff0b556a6e7d77ae8b13a73f3 |
| siglip_visual_adapted | 768 | b2d890f346302a4e | 不适用/未报告 | 不适用/未报告 |
| vision_visual_adapted | 768 | b2d890f346302a4e | {'train': 218, 'dev': 60, 'test': 60} | cb2d78cf16f08e9aea66ceff89da981dfc6e5a6ff0b556a6e7d77ae8b13a73f3 |

| probes | 任务 | 状态/标签来源 | 类别 | selected C | dev macro F1 |
|---|---|---|---|---:|---:|
| siglip | material | trained/museum_metadata_weak | bronze, ceramic, glass, jade, lacquer, porcelain, silver, stone, textile, wood | 10.00 | 0.7548 |
| siglip | object_type | trained/museum_metadata_weak | bowl, censer, cup, figure, jar, mirror, ornament, pendant, plate, pot, ritual vessel, teapot, textile, vase | 10.00 | 0.6927 |
| vision | material | trained/museum_metadata_weak | bronze, ceramic, glass, jade, lacquer, porcelain, silver, stone, textile, wood | 10.00 | 0.7548 |
| vision | object_type | trained/museum_metadata_weak | bowl, censer, cup, figure, jar, mirror, ornament, pendant, plate, pot, ritual vessel, teapot, textile, vase | 10.00 | 0.6927 |
| vision | colors | trained/ai_image_only | black, blue, brown, green, pink, red, white, yellow | 10.00 | 0.7358 |
| vision | shape | trained/ai_image_only | footed, handles, narrow_mouth, round_body, wide_mouth | 10.00 | 0.8216 |
| vision | motifs | trained/ai_image_only | bird, cloud, dragon, floral, geometric, landscape | 1.00 | 0.5343 |
| siglip_adapted | material | trained/museum_metadata_weak | bronze, ceramic, glass, jade, lacquer, porcelain, silver, stone, textile, wood | 0.10 | 0.7812 |
| siglip_adapted | object_type | trained/museum_metadata_weak | bowl, censer, cup, figure, jar, mirror, ornament, pendant, plate, pot, ritual vessel, teapot, textile, vase | 10.00 | 0.6697 |
| vision_metadata_adapted | material | trained/museum_metadata_weak | bronze, ceramic, glass, jade, lacquer, porcelain, silver, stone, textile, wood | 0.10 | 0.7812 |
| vision_metadata_adapted | object_type | trained/museum_metadata_weak | bowl, censer, cup, figure, jar, mirror, ornament, pendant, plate, pot, ritual vessel, teapot, textile, vase | 10.00 | 0.6697 |
| vision_metadata_adapted | colors | trained/ai_image_only | black, blue, brown, green, pink, red, white, yellow | 10.00 | 0.7047 |
| vision_metadata_adapted | shape | trained/ai_image_only | footed, handles, narrow_mouth, round_body, wide_mouth | 10.00 | 0.8360 |
| vision_metadata_adapted | motifs | trained/ai_image_only | bird, cloud, dragon, floral, geometric, landscape | 10.00 | 0.5450 |
| siglip_visual_adapted | material | trained/museum_metadata_weak | bronze, ceramic, glass, jade, lacquer, porcelain, silver, stone, textile, wood | 10.00 | 0.7596 |
| siglip_visual_adapted | object_type | trained/museum_metadata_weak | bowl, censer, cup, figure, jar, mirror, ornament, pendant, plate, pot, ritual vessel, teapot, textile, vase | 10.00 | 0.7136 |
| vision_visual_adapted | material | trained/museum_metadata_weak | bronze, ceramic, glass, jade, lacquer, porcelain, silver, stone, textile, wood | 10.00 | 0.7596 |
| vision_visual_adapted | object_type | trained/museum_metadata_weak | bowl, censer, cup, figure, jar, mirror, ornament, pendant, plate, pot, ritual vessel, teapot, textile, vase | 10.00 | 0.7136 |
| vision_visual_adapted | colors | trained/ai_image_only | black, blue, brown, green, pink, red, white, yellow | 10.00 | 0.7709 |
| vision_visual_adapted | shape | trained/ai_image_only | footed, handles, narrow_mouth, round_body, wide_mouth | 10.00 | 0.8356 |
| vision_visual_adapted | motifs | trained/ai_image_only | bird, cloud, dragon, floral, geometric, landscape | 0.10 | 0.5062 |

| probes | 元数据弱标签任务 | train/dev support | test eligible/excluded | test support | test accuracy | test balanced accuracy | test macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| siglip | material | 3273/530 | 567/83 | 567 | 0.8660 | 0.7732 | 0.7388 |
| siglip | object_type | 2371/356 | 398/252 | 398 | 0.8065 | 0.7439 | 0.6672 |
| vision | material | 3273/530 | 567/83 | 567 | 0.8660 | 0.7732 | 0.7388 |
| vision | object_type | 2371/356 | 398/252 | 398 | 0.8065 | 0.7439 | 0.6672 |
| siglip_adapted | material | 3273/530 | 567/83 | 567 | 0.8536 | 0.8012 | 0.7446 |
| siglip_adapted | object_type | 2371/356 | 398/252 | 398 | 0.8392 | 0.7503 | 0.7050 |
| vision_metadata_adapted | material | 3273/530 | 567/83 | 567 | 0.8536 | 0.8012 | 0.7446 |
| vision_metadata_adapted | object_type | 2371/356 | 398/252 | 398 | 0.8392 | 0.7503 | 0.7050 |
| siglip_visual_adapted | material | 3273/530 | 567/83 | 567 | 0.8660 | 0.7787 | 0.7565 |
| siglip_visual_adapted | object_type | 2371/356 | 398/252 | 398 | 0.8090 | 0.7608 | 0.6728 |
| vision_visual_adapted | material | 3273/530 | 567/83 | 567 | 0.8660 | 0.7787 | 0.7565 |
| vision_visual_adapted | object_type | 2371/356 | 398/252 | 398 | 0.8090 | 0.7608 | 0.6728 |

| probes | AI 任务/类别 | train 0/1 | test known | test positives | test F1 | test ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| vision | colors/black | 135/46 | 49 | 11 | 0.7143 | 0.9234 |
| vision | colors/blue | 129/51 | 49 | 10 | 0.8000 | 0.9564 |
| vision | colors/brown | 117/64 | 49 | 25 | 0.7755 | 0.8533 |
| vision | colors/green | 106/75 | 49 | 22 | 0.8372 | 0.9545 |
| vision | colors/pink | 171/10 | 49 | 5 | 0.3636 | 0.8864 |
| vision | colors/red | 140/41 | 49 | 15 | 0.9091 | 0.9863 |
| vision | colors/white | 95/86 | 49 | 23 | 0.8696 | 0.9515 |
| vision | colors/yellow | 128/53 | 49 | 17 | 0.5294 | 0.7886 |
| vision | shape/footed | 175/43 | 60 | 6 | 0.4706 | 0.8735 |
| vision | shape/handles | 202/16 | 60 | 8 | 0.5882 | 0.9231 |
| vision | shape/narrow_mouth | 127/61 | 55 | 17 | 0.9412 | 0.9768 |
| vision | shape/round_body | 140/77 | 60 | 20 | 0.8780 | 0.9613 |
| vision | shape/wide_mouth | 127/64 | 59 | 21 | 0.8718 | 0.9687 |
| vision | motifs/bird | 119/7 | 51 | 1 | 0.2500 | 0.9400 |
| vision | motifs/cloud | 115/10 | 43 | 1 | 0.4000 | 1.0000 |
| vision | motifs/dragon | 116/8 | 50 | 2 | 0.6667 | 1.0000 |
| vision | motifs/floral | 75/70 | 50 | 16 | 0.7500 | 0.9412 |
| vision | motifs/geometric | 80/78 | 56 | 25 | 0.8302 | 0.8632 |
| vision | motifs/landscape | 136/11 | 58 | 6 | 0.6667 | 0.9423 |
| vision_metadata_adapted | colors/black | 135/46 | 49 | 11 | 0.7692 | 0.9498 |
| vision_metadata_adapted | colors/blue | 129/51 | 49 | 10 | 0.7273 | 0.9564 |
| vision_metadata_adapted | colors/brown | 117/64 | 49 | 25 | 0.8235 | 0.8917 |
| vision_metadata_adapted | colors/green | 106/75 | 49 | 22 | 0.7727 | 0.9024 |
| vision_metadata_adapted | colors/pink | 171/10 | 49 | 5 | 0.5000 | 0.8909 |
| vision_metadata_adapted | colors/red | 140/41 | 49 | 15 | 0.9091 | 0.9961 |
| vision_metadata_adapted | colors/white | 95/86 | 49 | 23 | 0.8444 | 0.9448 |
| vision_metadata_adapted | colors/yellow | 128/53 | 49 | 17 | 0.6667 | 0.8511 |
| vision_metadata_adapted | shape/footed | 175/43 | 60 | 6 | 0.4706 | 0.8549 |
| vision_metadata_adapted | shape/handles | 202/16 | 60 | 8 | 0.7059 | 0.9808 |
| vision_metadata_adapted | shape/narrow_mouth | 127/61 | 55 | 17 | 0.9412 | 0.9876 |
| vision_metadata_adapted | shape/round_body | 140/77 | 60 | 20 | 0.8837 | 0.9525 |
| vision_metadata_adapted | shape/wide_mouth | 127/64 | 59 | 21 | 0.8000 | 0.9561 |
| vision_metadata_adapted | motifs/bird | 119/7 | 51 | 1 | 0.3333 | 1.0000 |
| vision_metadata_adapted | motifs/cloud | 115/10 | 43 | 1 | 1.0000 | 1.0000 |
| vision_metadata_adapted | motifs/dragon | 116/8 | 50 | 2 | 1.0000 | 1.0000 |
| vision_metadata_adapted | motifs/floral | 75/70 | 50 | 16 | 0.9091 | 0.9871 |
| vision_metadata_adapted | motifs/geometric | 80/78 | 56 | 25 | 0.7925 | 0.8684 |
| vision_metadata_adapted | motifs/landscape | 136/11 | 58 | 6 | 0.6154 | 0.9744 |
| vision_visual_adapted | colors/black | 135/46 | 49 | 11 | 0.7407 | 0.9378 |
| vision_visual_adapted | colors/blue | 129/51 | 49 | 10 | 0.9000 | 0.9872 |
| vision_visual_adapted | colors/brown | 117/64 | 49 | 25 | 0.7500 | 0.8983 |
| vision_visual_adapted | colors/green | 106/75 | 49 | 22 | 0.8636 | 0.9428 |
| vision_visual_adapted | colors/pink | 171/10 | 49 | 5 | 0.2222 | 0.8818 |
| vision_visual_adapted | colors/red | 140/41 | 49 | 15 | 0.8824 | 0.9902 |
| vision_visual_adapted | colors/white | 95/86 | 49 | 23 | 0.8889 | 0.9565 |
| vision_visual_adapted | colors/yellow | 128/53 | 49 | 17 | 0.6000 | 0.7868 |
| vision_visual_adapted | shape/footed | 175/43 | 60 | 6 | 0.5000 | 0.8642 |
| vision_visual_adapted | shape/handles | 202/16 | 60 | 8 | 0.6667 | 0.9615 |
| vision_visual_adapted | shape/narrow_mouth | 127/61 | 55 | 17 | 0.9412 | 0.9783 |
| vision_visual_adapted | shape/round_body | 140/77 | 60 | 20 | 0.8636 | 0.9575 |
| vision_visual_adapted | shape/wide_mouth | 127/64 | 59 | 21 | 0.8205 | 0.9649 |
| vision_visual_adapted | motifs/bird | 119/7 | 51 | 1 | 0.2222 | 0.9600 |
| vision_visual_adapted | motifs/cloud | 115/10 | 43 | 1 | 0.4000 | 1.0000 |
| vision_visual_adapted | motifs/dragon | 116/8 | 50 | 2 | 0.6667 | 1.0000 |
| vision_visual_adapted | motifs/floral | 75/70 | 50 | 16 | 0.7273 | 0.9393 |
| vision_visual_adapted | motifs/geometric | 80/78 | 56 | 25 | 0.7843 | 0.8387 |
| vision_visual_adapted | motifs/landscape | 136/11 | 58 | 6 | 0.7273 | 0.9679 |

## 成对比较：保留正负差值

候选减基线；每组 JSON 的预声明指标×语言共同组成 Holm 家族。CI、p 值与所有指标均保留；负值或不显著结果不会隐藏。

| 对比 | 语言 | 指标 | 查询数 | baseline | candidate | Δ [95% CI] | p | p Holm |
|---|---|---|---:|---:|---:|---|---:|---:|
| clip_vs_siglip | en | ndcg@10 | 24 | 0.7926 | 0.9076 | +0.1151 [+0.0307, +0.2029] | 0.01520 | 0.07599 |
| clip_vs_siglip | en | precision@5 | 24 | 0.3667 | 0.4583 | +0.0917 [+0.0250, +0.1750] | 0.03170 | 0.09509 |
| clip_vs_siglip | en | ap@10 | 24 | 0.7419 | 0.8753 | +0.1334 [+0.0311, +0.2401] | 0.02240 | 0.08959 |
| clip_vs_siglip | en | recall@20 | 24 | 0.9931 | 1.0000 | +0.0069 [+0.0000, +0.0208] | 1.00000 | 1.00000 |
| clip_vs_siglip | en | mrr@10 | 24 | 0.8250 | 0.9306 | +0.1056 [-0.0139, +0.2396] | 0.12639 | 0.25277 |
| clip_vs_siglip | zh | ndcg@10 | 24 | 0.1559 | 0.7930 | +0.6371 [+0.5209, +0.7350] | 0.00010 | 0.00100 |
| clip_vs_siglip | zh | precision@5 | 24 | 0.0583 | 0.3833 | +0.3250 [+0.2333, +0.4167] | 0.00010 | 0.00100 |
| clip_vs_siglip | zh | ap@10 | 24 | 0.1116 | 0.7091 | +0.5975 [+0.4292, +0.7401] | 0.00010 | 0.00100 |
| clip_vs_siglip | zh | recall@20 | 24 | 0.3601 | 0.9618 | +0.6017 [+0.4545, +0.7389] | 0.00010 | 0.00100 |
| clip_vs_siglip | zh | mrr@10 | 24 | 0.1569 | 0.8174 | +0.6604 [+0.4840, +0.8125] | 0.00010 | 0.00100 |
| metadata_lora | en | ndcg@10 | 24 | 0.9076 | 0.9152 | +0.0076 [-0.0148, +0.0292] | 0.52495 | 1.00000 |
| metadata_lora | en | precision@5 | 24 | 0.4583 | 0.4417 | -0.0167 [-0.0500, +0.0167] | 0.49785 | 1.00000 |
| metadata_lora | en | ap@10 | 24 | 0.8753 | 0.8866 | +0.0113 [-0.0141, +0.0384] | 0.43196 | 1.00000 |
| metadata_lora | en | recall@20 | 24 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| metadata_lora | en | mrr@10 | 24 | 0.9306 | 0.9583 | +0.0278 [+0.0000, +0.0764] | 0.49955 | 1.00000 |
| metadata_lora | zh | ndcg@10 | 24 | 0.7930 | 0.8488 | +0.0557 [+0.0022, +0.1124] | 0.07369 | 0.66323 |
| metadata_lora | zh | precision@5 | 24 | 0.3833 | 0.4083 | +0.0250 [+0.0000, +0.0583] | 0.25437 | 1.00000 |
| metadata_lora | zh | ap@10 | 24 | 0.7091 | 0.8099 | +0.1009 [+0.0275, +0.1866] | 0.02180 | 0.21798 |
| metadata_lora | zh | recall@20 | 24 | 0.9618 | 0.9931 | +0.0312 [+0.0000, +0.0833] | 0.49515 | 1.00000 |
| metadata_lora | zh | mrr@10 | 24 | 0.8174 | 0.8972 | +0.0799 [-0.0139, +0.1772] | 0.13059 | 1.00000 |
| visual_lora | en | ndcg@10 | 24 | 0.9076 | 0.9166 | +0.0090 [-0.0112, +0.0305] | 0.45845 | 1.00000 |
| visual_lora | en | precision@5 | 24 | 0.4583 | 0.4417 | -0.0167 [-0.0417, +0.0000] | 0.49165 | 1.00000 |
| visual_lora | en | ap@10 | 24 | 0.8753 | 0.8966 | +0.0213 [-0.0075, +0.0514] | 0.19768 | 1.00000 |
| visual_lora | en | recall@20 | 24 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| visual_lora | en | mrr@10 | 24 | 0.9306 | 0.9583 | +0.0278 [+0.0000, +0.0764] | 0.49955 | 1.00000 |
| visual_lora | zh | ndcg@10 | 24 | 0.7930 | 0.8251 | +0.0321 [-0.0112, +0.0786] | 0.20288 | 1.00000 |
| visual_lora | zh | precision@5 | 24 | 0.3833 | 0.3917 | +0.0083 [-0.0333, +0.0583] | 0.74653 | 1.00000 |
| visual_lora | zh | ap@10 | 24 | 0.7091 | 0.7720 | +0.0629 [-0.0054, +0.1408] | 0.12119 | 1.00000 |
| visual_lora | zh | recall@20 | 24 | 0.9618 | 0.9618 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| visual_lora | zh | mrr@10 | 24 | 0.8174 | 0.8403 | +0.0229 [-0.0639, +0.1146] | 0.68443 | 1.00000 |
| translated_vs_selected | en | ndcg@10 | 24 | 0.8968 | 0.9151 | +0.0183 [-0.0004, +0.0381] | 0.07549 | 0.67943 |
| translated_vs_selected | en | precision@5 | 24 | 0.4500 | 0.4583 | +0.0083 [-0.0167, +0.0333] | 1.00000 | 1.00000 |
| translated_vs_selected | en | ap@10 | 24 | 0.8633 | 0.8990 | +0.0357 [+0.0066, +0.0680] | 0.04850 | 0.48495 |
| translated_vs_selected | en | recall@20 | 24 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| translated_vs_selected | en | mrr@10 | 24 | 0.9306 | 0.9583 | +0.0278 [+0.0000, +0.0764] | 0.49955 | 1.00000 |
| translated_vs_selected | zh | ndcg@10 | 24 | 0.8091 | 0.8078 | -0.0013 [-0.0418, +0.0399] | 0.95850 | 1.00000 |
| translated_vs_selected | zh | precision@5 | 24 | 0.3917 | 0.3917 | -0.0000 [-0.0250, +0.0250] | 1.00000 | 1.00000 |
| translated_vs_selected | zh | ap@10 | 24 | 0.7685 | 0.7772 | +0.0086 [-0.0542, +0.0712] | 0.79942 | 1.00000 |
| translated_vs_selected | zh | recall@20 | 24 | 0.9861 | 0.9931 | +0.0069 [-0.0208, +0.0417] | 1.00000 | 1.00000 |
| translated_vs_selected | zh | mrr@10 | 24 | 0.8789 | 0.8556 | -0.0233 [-0.1091, +0.0486] | 0.75102 | 1.00000 |
| base_vs_detail_heads | en | ndcg@10 | 24 | 0.9009 | 0.8994 | -0.0016 [-0.0176, +0.0120] | 0.86571 | 1.00000 |
| base_vs_detail_heads | en | precision@5 | 24 | 0.4500 | 0.4500 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| base_vs_detail_heads | en | ap@10 | 24 | 0.8654 | 0.8717 | +0.0063 [+0.0000, +0.0167] | 0.50015 | 1.00000 |
| base_vs_detail_heads | en | recall@20 | 24 | 1.0000 | 1.0000 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| base_vs_detail_heads | en | mrr@10 | 24 | 0.9306 | 0.9306 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| base_vs_detail_heads | zh | ndcg@10 | 24 | 0.7760 | 0.7898 | +0.0139 [-0.0062, +0.0492] | 0.86661 | 1.00000 |
| base_vs_detail_heads | zh | precision@5 | 24 | 0.3917 | 0.4000 | +0.0083 [+0.0000, +0.0250] | 1.00000 | 1.00000 |
| base_vs_detail_heads | zh | ap@10 | 24 | 0.7144 | 0.7407 | +0.0263 [-0.0070, +0.0857] | 0.75272 | 1.00000 |
| base_vs_detail_heads | zh | recall@20 | 24 | 0.9861 | 0.9861 | +0.0000 [+0.0000, +0.0000] | 1.00000 | 1.00000 |
| base_vs_detail_heads | zh | mrr@10 | 24 | 0.8303 | 0.8580 | +0.0278 [+0.0000, +0.0833] | 1.00000 | 1.00000 |

## 固定失败短名单

按 selected test 每个语义 qid 在中英两种表述中的最小 nDCG@10 升序，qid 稳定打破并列，固定保留六个不同 qid。先后规则不使用人工挑选。
目标 alias 是 qrels 最高等级的全部匹配图，不是假定查询源图。下表展示最差语言的前五；JSON 保留 ID、alias 和显式等级。冻结盲图的人工复核归因见 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。

| qid | 最差语言 | min nDCG@10 | 目标 aliases | top5 alias:grade |
|---|---|---:|---|---|
| vt024 | zh | 0.2070 | V058 | V005:0, V036:1, V018:0, V043:1, V045:0 |
| vt014 | zh | 0.4275 | V051 | V031:0, V010:0, V051:3, V059:2, V014:0 |
| vt021 | zh | 0.4949 | V039 | V016:1, V037:1, V021:0, V025:1, V039:3 |
| vt006 | zh | 0.5104 | V041, V042, V044 | V053:0, V057:0, V042:3, V059:0, V010:0 |
| vt020 | zh | 0.6240 | V043 | V038:2, V005:0, V018:0, V043:3, V013:0 |
| vt007 | zh | 0.6843 | V003, V016, V025, V037 | V053:0, V025:3, V004:2, V021:0, V016:3 |

联系表由 `--failure-cards` 生成；本轮随后人工查看六张联系表并记录了工程归因，不改变参数。

## 系统验收证据

- `artifacts/system_acceptance.json`：scope=`current_frozen_production`；health=`ok`。
  warm sequential：`{"image_only": {"requests": 50, "p50_ms": 56.11839999619406, "p95_ms": 87.70423500172909, "max_ms": 99.30220000387635}, "baseline": {"requests": 50, "p50_ms": 64.32580000546295, "p95_ms": 215.41040499359943, "max_ms": 244.30749999010004}, "hybrid": {"requests": 50, "p50_ms": 80.52395000413526, "p95_ms": 214.80259499730886, "max_ms": 235.41770000883844}, "visual": {"requests": 50, "p50_ms": 81.01840000017546, "p95_ms": 224.04614499901072, "max_ms": 259.36289998935536}}`；并发：`{"requests": 200, "p50_ms": 190.15539999236353, "p95_ms": 233.5269750088628, "max_ms": 286.26969999459106, "workers": 4, "seconds": 9.695850300005986, "requests_per_second": 20.62738117974826}`。

pilot 验收仅证明其当时图库和配置；不能替代当前冻结图库或 production image_only 的系统验收。原始 checks、健康状态与时延均保留在 evidence JSON。

## 解释边界与下一阶段

- 所有预期 JSON 已齐备；数值仍受小规模 AI 视觉审计的边界约束。
- 60 图受限图库的结果不能当作全馆藏检索精度。AI caption paired 与 AI 自由查询共享 AI 判断偏差。
- 黑白历史图无法恢复真实本体颜色；细粒度纹饰和难例需另行看图，未知不能当确定负例。
- 详情 heads、LoRA 和查询改写可能无收益或退化；完整训练和成对比较保留这些负结果。
- 元数据银标仅为附录：`scripts/summarize_results.py` / `docs/METADATA_RESULTS.md`，不替代视觉主结果。
- 所有读取 JSON 的 SHA256、完整原始 reports、选项与失败记录见 `docs/visual_experiment_evidence.json`。
