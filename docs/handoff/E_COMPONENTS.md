# 视觉辅助、性能与产品集成

演示模块 E · 页面 8、10–11 · 讲述 110秒

## 目的

选择能增加检索质量且成本合适的辅助组件，把研究结论落实到可用系统。

## 已完成

消融HSV颜色、HOG形状、LBP纹理、轮廓、GrabCut与DINO；保留有效的软融合，实验算法集中到实验档案。建立普通用户三入口、完整集合分页、来源详情和专业统计；制作离线HTML演示与实测数据导出。

## 可直接引用的结果

控制图图nDCG@10：生产0.7174、DINO 0.5941、生产+DINO 0.7347、生产+前景颜色0.7306。4,692图LoRA编码102.5秒，缓存余弦排序1.56毫秒；DINO预筛3.99毫秒后还要重排，因此默认直接排序。

## 代码与证据

docs/CLASSICAL_CV_FEATURES.md；docs/RESEARCH_V5_RESULTS.md；web/src/App.tsx；web/src/ResearchArchive.tsx；web/public/assets/presentation/

```powershell
pnpm --dir web build
.venv/Scripts/python.exe -m pytest -q
```

整理报告时按“需求 → 方法 → 公平对照 → 结果 → 设计选择”展开；数字以冻结结果为准。公共小型证据见 `docs/research/`，完整实验附件见本轮Release。
