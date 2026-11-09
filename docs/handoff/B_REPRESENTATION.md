# 视觉表示、微调与双语查询

演示模块 B · 页面 3–4 · 讲述 80秒

## 目的

研究图像库仅提供像素时，领域适配如何改善文字到图像的匹配。

## 已完成

对照CLIP、SigLIP2、视觉LoRA与组员的双语池化头LoRA；检验真实权重、训练样本和10,200图库。训练使用馆藏文字监督，在线候选使用图像向量；中英文保留完整描述并作有限查询展开。

## 可直接引用的结果

原始英文nDCG@10：CLIP 0.6233、SigLIP2 0.7219、生产LoRA 0.7954、组员LoRA 0.7511；中文分别0.1408、0.6385、0.7217、0.7104。

## 代码与证据

docs/RESEARCH_V5_RESULTS.md；docs/TEAMMATE_MODEL_AUDIT.md；src/curio/represent/；scripts/evaluate_teammate_workflow_v5.py

```powershell
.venv/Scripts/python.exe scripts/evaluate_teammate_workflow_v5.py --help
```

整理报告时按“需求 → 方法 → 公平对照 → 结果 → 设计选择”展开；数字以冻结结果为准。公共小型证据见 `docs/research/`，完整实验附件见本轮Release。
