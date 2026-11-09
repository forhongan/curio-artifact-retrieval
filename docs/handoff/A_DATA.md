# 数据与相关性评测

演示模块 A · 页面 1–2 · 讲述 55秒

## 目的

让不同模型在同一批图片、同一语义请求和可追溯判断下比较。

## 已完成

冻结 Met 2,954 + Cleveland 1,738 张图像；保留来源与许可。修订200张图的外观/来源事实，以60张dev选择边界、140张test比较20个中英文意图。器型兼容族、多纹样和未知项替代僵硬的单标签。

## 可直接引用的结果

4,000条规则衍生判断；排序移除未知项，集合评估单列未知入选率。质量指标范围是140张test核心。

## 代码与证据

docs/RELEVANCE_V5.md；eval/relevance_v5/；src/curio/evaluate/relevance.py

```powershell
.venv/Scripts/python.exe scripts/evaluate_relevance_v5.py --help
```

整理报告时按“需求 → 方法 → 公平对照 → 结果 → 设计选择”展开；数字以冻结结果为准。公共小型证据见 `docs/research/`，完整实验附件见本轮Release。
