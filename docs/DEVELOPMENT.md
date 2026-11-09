# 开发入口

当前功能和验收见 [CURRENT_STATE](CURRENT_STATE.md)；工作包见 [HANDOFF](HANDOFF.md)；演示见 [PRESENTATION_GUIDE](PRESENTATION_GUIDE.md)。运行命令见根README。

前端：`web/src/App.tsx`（普通/专业模式）、`ResearchArchive.tsx`（实验档案）、`museum.css`（当前样式）。API入口：`src/curio/serve/app.py`。

统计图由 `scripts/build_showcase_data.py` 从冻结结果生成；演示数据由 `scripts/build_presentation_v6.py`生成；交接包和短讲稿由 `scripts/prepare_handoff_v6.py`生成。讲稿与演示JS的notes同步。

修改后运行Ruff、相关pytest与TypeScript构建。图像、权重和完整实验缓存放Release，小型结果和代码进入Git。历史阶段文档在 `history/research_before_showcase_v6/`。
