# v15 前端交接验收 · 2026-11-17

基线：`codex/research-showcase-v6` 的 `59af1e6a58fe203807ef85b73feae687b53c8eaa`。本分支只交付集合边界后端、研究记录、接口客户端和运行资料；与基线相比 `web/` 无差异。原研究工作区的未提交前端改动未纳入交付。

## 已验证

- 完整 Python 测试：651 passed、13 skipped；Ruff 检查通过，TypeScript 客户端严格类型检查通过。
- 从 Windows ZIP 解压到独立目录，用新建 Python 3.12 虚拟环境只安装 `requirements-boundary-demo.txt` 启动 API；环境没有 Torch、MLX 或原项目安装。依赖记录见 `boundary_v15_handoff_environment.json`。
- HTTP 验收覆盖 8 组双语意图、37,536 个布尔决定、结果/决定对应、子集关系、详情、图片、CORS、未知查询 422，以及已完成任务取消的无操作行为。见 `boundary_v15_handoff_http.json`。可用 `python scripts/verify_boundary_handoff.py` 重跑。
- Chrome 从独立 `127.0.0.1:3000` 测试页导入实际交付的 TypeScript 客户端，8 组中英文调用及图片显示通过、无页面异常。实时 queued/running/complete 与取消流程使用模拟响应验证；这部分不表示 Windows 已运行 MLX 推理。见 `boundary_v15_handoff_browser.json`。
- 所有 4,692 张 JPEG 均解码校验，最大边不超过 384px；ZIP CRC、源码字节一致性和附件 SHA-256 已校验。见 `boundary_v15_handoff_archives.json`。
- 原冻结视觉 worker SHA-256 仍为 `4005f48b0c6c46d87716f644841e5989966c3cbc15119696b9368badb1bac8ab`。

## 尚未验证及限制

验收主机是 macOS arm64，未在真实 Windows 主机执行 PowerShell/CMD 启动脚本。回放包仅覆盖列出的 8 组意图，支持该解析器识别的等义表达，不能推理新的任意请求。完整实时服务仍依赖 Apple Silicon MLX、原图库与原 SigLIP 资源。

这是交付和接口验收，不是独立准确率评测。AI/馆藏辅助复核不等于独立人工金标准；特殊葫芦瓶漏检及窄口门槛的校准局限保留，详见 `eval/boundary_catalogue_v15/README.md`。
