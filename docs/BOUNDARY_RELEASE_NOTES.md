全库 Yes/No 集合边界 v15，供前端接入。

[前端接入说明](https://github.com/zhizhizhi-victor/curio-boundary-handoff/blob/codex/boundary-v15-handoff/docs/BOUNDARY_FRONTEND_HANDOFF.md) · [交接源码](https://github.com/zhizhizhi-victor/curio-boundary-handoff/tree/codex/boundary-v15-handoff)

- Windows：下载 curio-boundary-v15-windows.zip，安装 Python 3.12，解压后双击 run-boundary-windows.cmd；API http://127.0.0.1:8015/docs。
- 包含 4,692 张展示图片、馆藏详情、8 组双语查询的完整二元结果和 TypeScript 调用示例。
- Windows 包为已验证结果回放，不执行新推理；不支持的查询返回 422。实时新查询需要 Apple Silicon MLX 后端。
- curio-boundary-v15-runtime.zip 是实时算法的额外冻结资源；不含原图库或 MLX 模型。
- 前端页面和样式未改动。完整对接说明见 docs/BOUNDARY_FRONTEND_HANDOFF.md。
- 交接分支 651 tests passed、13 skipped；ZIP 恢复后的轻量 API 和 Chrome 客户端验证通过，未在真实 Windows 主机执行启动脚本。
- 特殊器形仍可能漏检，35% 口缘比例是未独立校准的严格工程阈值；复核标签不是独立人工金标准。
