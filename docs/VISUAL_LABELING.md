# AI 盲图标注协议

用途：独立视觉细节分类头和视觉 caption 训练。**AI 生成，不是人工 gold**。标注者只读指定编号卡片，不读 catalog/blind_map、馆藏标题、年代、材质或检索排名。看到的像素与自己的不确定性都应如实记录。不能把受限分辨率下未看清的纹饰猜成标签。

## 每张图像的 JSONL

```json
{"alias":"T001","caption_en":"A pale green vessel with a rounded body and a narrow opening.","caption_zh":"浅绿色器物，圆腹，开口较窄。","object_type":"vase","colors":{"blue":0,"white":0,"green":1,"red":0,"black":0,"yellow":0,"brown":0,"pink":0},"shape":{"narrow_mouth":1,"wide_mouth":0,"handles":0,"footed":0,"round_body":1},"motifs":{"dragon":null,"phoenix":null,"lotus":null,"floral":null,"cloud":null,"bird":null,"fish":null,"tiger":null,"geometric":null,"landscape":null},"confidence":"medium","uncertainty":"Small decoration is not resolved."}
```

示例只是 schema，不可当实际标注。原始编号由工具生成，标注者不能按编号推断类别。每个 split 的 alias 独立：T=train、D=dev、V=test。输出以 UTF-8 保存；每次看完一张 sheet 即保存该批标注，避免丢失工作。

- `caption_en` / `caption_zh`：只描述可见外观，简短自然语言。不要加朝代、用途、人物身份、器物专名或未经确认的材质。可写“光滑白色碗状器”，不能仅凭白色断言“乾隆瓷碗”。
- `object_type`：可见器形，使用 bowl/plate/vase/jar/cup/teapot/ewer/pot/figure/mirror/box/textile/ritual vessel/pendant/stele/ornament/cong/bi/lid/other；不清楚用 other 并说明。
- 颜色是器物本体的显著颜色，排除摄影背景、阴影和小高光。8 个键均填 0/1/null；白/蓝双色纹样可同时为 1。金黄色视觉颜色用 yellow，不推断真实金材质。
- `narrow_mouth`：可见容器开口大约不超过最大腹宽一半，或明显细颈；看不到开口也不确定颈宽时 null。
- `wide_mouth`：碗盘类敞开或开口接近主体最大宽度。不开口的雕像等为 0。
- `handles`：看得见的柄、耳或把手。无法看到背面并不证明背面没有；标签仅指当前图中可见。
- `footed`：明显独立足或较高承台，普通窄圈足不算。
- `round_body`：球形、卵圆形器腹；不是所有圆盘、圆镜都算。
- 纹饰键是可见 dragon/phoenix/lotus/floral/cloud/bird/fish/tiger/geometric/landscape。明显看到为 1；当前展示没有看到为 0；分辨率/遮挡导致无法判断则 null。花卉不能仅凭“可能是莲”擅加 lotus；不清楚的兽纹不可猜 dragon。

0 表示“当前像素未呈现该属性”，不表示藏品真实没有该属性。unknown 在训练中逐属性排除，不自动填成负例。最终模型输出是对可见外观的 AI 弱监督预测。

## 合并与实验隔离

主代理在盲图标注完成后，使用隐藏映射补充 item_id、split、image_sha256、label_source=ai_image_only、annotator。加载器核对图像哈希与 split。训练只读 train，C 与 epoch 只按 dev 选择；test 只做最终评价。所有卡片和原始标注保留，模型记录完整标注文件 SHA256。

自由查询相关性另行标注：必须逐查询看全图库，不能从原始目标 ID、检索名次或元数据词重合构造相关性。小样本 AI 审计应明确图库大小、查询数和作者身份，不报告真人一致性。
