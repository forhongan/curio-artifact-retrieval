# 全库yes/no集合边界

演示模块 C · 页面 5 · 讲述 50秒

## 目的

将固定Top-K序列扩展为数量随请求变化的入选集合。

## 已完成

比较自适应GMM、dev共享分数阈值、逻辑回归边界、HSV辅助边界。运行时对4,692张图片逐件决策，API返回完整results，preview单独截断，前端每页12张。中文推荐共享阈值，英文推荐颜色辅助；用户可切换。

## 可直接引用的结果

英文宏F1：0.4686 → 0.5292 / 0.5734 / 0.5831；中文：0.4613 → 0.5257 / 0.5149 / 0.5240。这些是test核心上的集合质量，不是全库召回率。

## 代码与证据

src/curio/retrieve/calibrated_sets.py；src/curio/evaluate/set_boundary.py；tests/test_relevance_v5.py；POST /api/search/set

```powershell
.venv/Scripts/python.exe scripts/verify_research_v5_api.py
```

整理报告时按“需求 → 方法 → 公平对照 → 结果 → 设计选择”展开；数字以冻结结果为准。公共小型证据见 `docs/research/`，完整实验附件见本轮Release。
