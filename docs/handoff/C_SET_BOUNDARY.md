# 全库 Yes/No 集合边界 · v15

负责将查询变成全库逐图布尔判定，提供完整集合、逐条件依据及中文/英文一致的查询计划。前端接入从 [Windows 与 API 交接说明](../BOUNDARY_FRONTEND_HANDOFF.md) 开始。

## 当前方法

`catalogue_visual` 将器类/材质的馆藏证据与照片验证结合。颜色需照片可观察；窄口增加口部可见性、反例验证及口缘/器身比例门槛。任一必要条件不成立或证据不足，最终均为 No，同时保留原因。输出不是固定 Top-K。

核心文件为 `src/curio/retrieve/catalogue_query.py`、`catalogue_visual.py`、`joint_sets.py` 与 `joint_vlm.py`。实时入口为 `POST /api/search/set/jobs`，请求必须传 `method: "catalogue_visual"`。前端只需复制 `integration/boundary-client.ts` 并接入自己的页面；本分支不包含页面改动。

## 交付与验证范围

- Windows 轻量结果包：4,692 张展示图、详情和 8 组已验证中英文查询结果，不运行新模型推理。
- 实时模型：目前依赖 Apple Silicon MLX；Windows 前端可经代理或 SSH 转发连接该后端。
- 8 组全库结果每组均有 4,692 个布尔决定，中英文返回相同集合。计数、接口和限制见接入说明。
- 研究复核记录见 `eval/boundary_catalogue_v15/README.md`；AI/馆藏辅助标签并非独立人工金标准，不能由此宣称任意查询准确率。
- 已知局限：严格筛选会漏掉部分特殊葫芦瓶；35% 的窄口几何阈值尚未独立人工校准。

早期 GMM/共享阈值/颜色辅助方法及 v5 的 F1 属于历史对照，不代表 v15 的新评测结果。历史实验保留在 `docs/research/`，前端应使用上述异步 jobs 接口。
