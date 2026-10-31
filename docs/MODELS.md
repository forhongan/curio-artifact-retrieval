# 模型与训练协议

## 预训练编码器

| 名称 | 官方模型 | 固定 revision | 表示维度 |
|---|---|---|---|
| CLIP 基线 | openai/clip-vit-base-patch32 | 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268 | 512 |
| 多语主模型候选 | google/siglip2-base-patch16-224 | 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2 | 768 |

来源为各模型作者的 Hugging Face model card 及 Transformers 官方实现。模型代码不启用 `trust_remote_code`。SigLIP2 模型卡标 Apache 2.0；CLIP 权重来源与原项目许可参见 openai/CLIP。缓存保存在 Hugging Face 默认目录，索引 manifest 同时记录 model_id 和 commit revision。服务离线运行可通过启动器设置 offline 环境变量。

- https://huggingface.co/google/siglip2-base-patch16-224
- https://huggingface.co/openai/clip-vit-base-patch32
- https://github.com/openai/CLIP
- https://huggingface.co/docs/transformers/model_doc/siglip2
- https://huggingface.co/docs/peft/package_reference/lora

图像使用各模型原生 image processor。CLIP 使用最长 77 token；SigLIP2 按其训练方式 padding=max_length、最长 64 token。输出均转 float32 并 L2 归一化，检索使用精确内积。图像保存时保留长宽比，原生模型预处理的裁剪/缩放与图库缩略图处理分离。

## 图像塔领域 LoRA

冻结原模型，在 vision attention 的 q_proj/v_proj 添加 LoRA，rank=8、alpha=16、dropout=.05，共 294,912 个可训练参数。文本塔固定，其 caption 向量缓存；对称多正例对比损失将完全相同 caption 的多个图像视为正例，而不把同样描述的文物当负例。温度 .07；AdamW，lr=1e-4，weight_decay=.01，cosine schedule、5% warmup，梯度裁剪1.0，CUDA fp16 autocast 和 GradScaler。

为了减小窄领域训练破坏预训练视觉空间的风险，加入 .5 × (1-cosine(student_image, frozen_teacher_image)) 的特征锚定损失。只按 dev 对象的 text-to-image R@5，再按 MRR 选 epoch。即使微调没有提高，也保留训练日志和负结果；是否部署不能靠训练 loss 下降决定。

train/dev/test 在对象层固定。训练和留出评估会拒绝重复 ID、跨分区相同 image_path/SHA256/duplicate_group，以及跨分区已识别的近重复关系。训练不使用 test。主表和模型的哈希写入 run.json，适配器单独保存，重载时计算权重哈希。

## 弱监督 caption 的边界

caption 用馆藏标题、器名、材质及显式纹饰线索组成，不加入 period/culture/馆名/creditLine。清洗删除年代数字、英文序数世纪、朝代、常见年号及中文对应词。不使用策展长描述造视觉标签，也不把背景颜色当器物颜色。规则清洗仍可能留下非视觉艺术史词汇或错误元数据，这项残余限制应随实验记录披露，不能称为人写视觉 caption。

## 视觉属性探针

在冻结图像嵌入上训练器型/材质 LogisticRegression。参数 C 只在 dev 选择；test 仅在参数冻结后评估。探针标签来自元数据，是弱监督。多标签纹饰里“未写龙”不代表“没有龙”，因此基础头不把馆藏缺失纹饰当负例，也不用元数据或背景提取颜色标签。

检索可选 visual 变体用探针概率软重排，显示“弱监督视觉分类预测”。该概率不等于人工确认，不支持仅凭预测宣称看见年代、数量或空间关系。探针必须和所用索引的图像特征哈希严格一致。

## 试运行记录

正式视觉细节监督另见下一节；本节只记录工程试运行，不可当正式结果。

2026-10-30 已完成 CUDA 图文编码、小批次 LoRA 反向传播、真实 612 件快照上的训练/适配器保存及重载。该试运行仅用于发现工程错误，不列入最终方法对比。正式数据版本、训练 epoch 和测试结果以 experiments 产物及最终 RESULTS.md 为准。

## AI 盲图细节监督

原始盲图池依据来源与稳定SHA抽样240 train、60 dev、60 test。后续官方同物/部件关系审计发现原分区仍有家族泄漏，v1实验全部作废。正式v2按关系连通组重分区，保留独立dev/test盲图库为组锚，并将22张迁入留出家族的原训练图排除；实际监督为218 train、60 dev、60 test。所有360张原始图卡与标签保留，canonical标注338条，排除台账另存。

AI标注者只看随机编号图卡，不读取馆藏标题、材料、描述或检索排名。这是AI弱监督，不是人工gold。黑白历史照片的本体颜色保留unknown；细纹饰无法判清时逐属性null，不把模糊当负例。详见 VISUAL_LABELING.md。

`detail_probes` 对满足至少5正/5负训练例、且dev含两类的属性训练独立LogisticRegression。每任务C∈{.1,1,10}按dev宏F1选择，固定0.5阈值；输出sigmoid而非把多属性强行softmax。test报告逐属性support、F1和AUC。基础器型/材质头与细节头来源分开说明。

视觉caption LoRA使用218个train对象和60个dev对象，各训练随机/hard采样12epochs，其他优化参数保持一致。另将视觉适配器的全库图像向量重建并重新训练匹配该空间的基础/细节头。冻结SigLIP、元数据caption LoRA与视觉caption LoRA各自的base/detail头以及权重0/.05/.15/.3只在视觉dev选择；生产使用image_only家族中双语nDCG@10最高者。所有比较模型的最优者另列，避免暗示生产候选就是全局最优。

24条视觉dev意图和24条视觉test意图均由AI直接看对应60图后编写中英表述，并逐查询穷举0–3级相关性；不是由caption词重合、原始目标ID或检索排名自动生成。两种语言不是两倍独立语义样本。检索指标只代表这两个受限图库上的AI判断，不能直接宣称4,692图全库的真人效果。
