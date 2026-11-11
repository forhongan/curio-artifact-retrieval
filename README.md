# 拾遗 Curio

中国古代文物视觉检索系统与课程研究项目。用中文或英文描述器物、通过专业名称寻找外部参考照片，或上传图片，在4,692张开放馆藏图像中检索。

## 运行

```powershell
uv sync --extra dev
pnpm --dir web install --frozen-lockfile
python scripts/download_resources.py
pnpm --dir web build
.\scripts\run.ps1
```

打开 `http://127.0.0.1:8000/`。资源恢复见 [RESOURCES](docs/RESOURCES.md)；追加研究附件见 [SHOWCASE_RELEASE](docs/SHOWCASE_RELEASE.md)。

已部署的笔记本可双击 **Start Curio.cmd** 启动，双击 **Open Presentation.cmd** 打开离线演示；见 [笔记本使用说明](docs/LAPTOP_DEMO.md)。

- **用文字找**：SigLIP2视觉LoRA，中英文查询展开，支持相似排序和完整匹配集合。
- **用专业名称找**：博物馆API取得真实照片，再用SigLIP2与DINO比较图库。已核查的演示案例：梅瓶。
- **用图片找**：上传JPG/PNG/WebP，以领域图像模型寻找相似器物。
- **专业模式**：模型、集合边界、视觉辅助与编码成本的实测图表。
- **实验档案**：集中保存经典CV、历史对照和复现工具。

## 演示与交接

[HTML演示](web/public/assets/presentation/index.html)可离线打开；服务入口 `/assets/presentation/index.html`。11页围绕“大类→组合外观→专业器型”展开，讲述6分30秒+操作展示60秒=7分30秒。支持键盘翻页、全屏、计时和讲者提示。

- [演示流程与短讲稿](docs/PRESENTATION_GUIDE.md)
- [五个研究交接模块](docs/HANDOFF.md)
- [英文技术稿](docs/TECHNICAL_REPORT_DRAFT.md)
- [当前状态](docs/CURRENT_STATE.md)

## 研究

| 研究问题 | 实现与证据 |
|---|---|
| 仅提供图库像素时的图文表示 | [CLIP/SigLIP2/LoRA同协议比较](docs/RESEARCH_V5_RESULTS.md) |
| 真实器型与不完整证据 | [v5.1相关性协议](docs/RELEVANCE_V5.md) |
| 数量可变的完整集合 | [集合边界模块](docs/handoff/C_SET_BOUNDARY.md) |
| 专业名称转视觉实例 | [六个外部参考图场景](docs/REFERENCE_SCENARIOS_V6.md) |
| 经典CV、DINO与性能 | [辅助组件模块](docs/handoff/E_COMPONENTS.md) |
| 更大图库与组员模型 | [模型与评测审计](docs/TEAMMATE_MODEL_AUDIT.md) |

冻结结果在 `docs/research/`，事实和判断在 `eval/relevance_v5/`。质量对照覆盖200张核查图中的60 dev / 140 test与20个双语意图；运行图库4,692张。训练监督、来源核查和显示使用馆藏记录，在线图库候选分数来自图像特征。

## 开发

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/ruff.exe check src tests scripts
pnpm --dir web build
```

`src/curio/`包含采集、表示、索引、检索、评测和API；`web/`包含普通前端、专业模式和演示；`scripts/`包含实验入口；`docs/handoff/`是当前工作包；`history/`保存旧阶段文档和评测。模型和预处理不变时复用图库向量。

详细说明：[架构](docs/ARCHITECTURE.md)、[API](docs/API.md)、[数据](docs/DATA.md)、[AI过程](docs/ai_log.md)。
