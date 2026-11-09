# 专业名称、外部参考图与图图检索

演示模块 D · 页面 6–7 · 讲述 95秒

## 目的

把专业器型名转成真实参考照片，再按图像外观比较图库。

## 已完成

博物馆API文本检索、有限同义词发现、来源条件核验、真实下载解码、同物/近重复隔离；比较五种参考聚合，SigLIP2和DINO参与像素匹配。修复只看Vase标题而漏掉正文meiping/gourd的情况：正文器型还必须与器物标题族相容。

## 可直接引用的结果

六个实际场景全部留档。梅瓶：中文生产路径0/5，英文原始文本4/5，6张真实V&A参考均值5/5（宽肩短颈渐收腹的兼容器型）。这是探索后选出的读图案例。葫芦、青瓷、龙盘、皮囊壶与抱月瓶的结果也保留。

## 代码与证据

docs/REFERENCE_SCENARIOS_V6.md；src/curio/retrieve/qualified_references.py；src/curio/retrieve/reference_aggregation.py；artifacts/reference_scenarios_v6/

```powershell
.venv/Scripts/python.exe scripts/explore_reference_scenarios_v6.py
```

整理报告时按“需求 → 方法 → 公平对照 → 结果 → 设计选择”展开；数字以冻结结果为准。公共小型证据见 `docs/research/`，完整实验附件见本轮Release。
