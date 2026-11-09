# 本轮交付与追加资源

版本：`curio-showcase-v6-20261109`。下载位置：[本仓库Release](https://github.com/forhongan/curio-artifact-retrieval/releases/tag/curio-showcase-v6-20261109)。

| 附件 | 用途 |
|---|---|
| `curio-v6-runtime.zip` | 新集合边界、冻结阈值、HSV头/全库预测和索引ID；先恢复v2基础资源，再解压到项目根目录 |
| `curio-v6-research.zip` | v5.1评分矩阵、模型与视觉辅助结果、背景消融、六个API场景请求与排名；解压到项目根目录 |
| `curio-v6-presentation.zip` | 独立HTML、实测数据、CC0藏品示例和60秒界面剪辑；解压后打开`presentation/index.html` |
| `Curio_Technical_Report_Draft.pdf` | 已排版的英文课程技术稿 |
| `resources_manifest.json` / `SHA256SUMS.txt` | 附件大小、SHA256和运行时逐文件清单 |

仓库内的固定清单是 `docs/showcase_manifest.json`。例如校验已下载的运行包：

```powershell
Get-FileHash .\curio-v6-runtime.zip -Algorithm SHA256
Expand-Archive -LiteralPath .\curio-v6-runtime.zip -DestinationPath .
```

基础图库、索引、LoRA与SigLIP2/DINO底模继续使用 [v2资源恢复流程](RESOURCES.md)。追加包无需重新下载相同图库或重编码向量。源码更新后运行 `pnpm --dir web build`，然后 `scripts/run.ps1`。

演示讲述390秒，操作剪辑60秒。视频是实际界面截图的步骤剪辑，切换速度不作为算法延迟。讲稿、五个交接模块和指标来源都随源码提供。

发布范围包括代码、公开馆藏研究、评测、实验参数与输出；本地课程原始要求、沟通接入说明、环境凭据和组员完整克隆不在发布包中。
