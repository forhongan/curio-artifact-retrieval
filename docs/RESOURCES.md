# 数据、模型与实验记录下载

资源版本：`curio-resources-v2-20261104`。资源放在[本仓库 GitHub Release](https://github.com/forhongan/curio-artifact-retrieval/releases/tag/curio-resources-v2-20261104)的 Assets 中，与源码一起公开共享。`git clone` 只下载源码，Release附件需另行下载。所有压缩包都是可独立解压的ZIP，不是需要拼接的分卷。

## 推荐安装

在克隆得到的项目根目录运行（Python 3.11–3.12）：

```powershell
python scripts/download_resources.py --list
python scripts/download_resources.py
uv sync --extra dev
pnpm --dir web install --frozen-lockfile
pnpm --dir web build
.\scripts\run.ps1 -CheckOnly
.\scripts\run.ps1
```

默认下载约 **2.81 GiB**：4,793张图库处理图与缩略图、45张实际使用的参考照片、来源与数据快照、全部当前索引/训练适配器/评测产物，以及固定revision的SigLIP2和DINOv2底模。依赖环境仍通过uv/pnpm安装，未把Windows虚拟环境或node_modules打包。启动器在没有自定义 `HF_HUB_CACHE` 时自动使用 `models/huggingface/hub`。

完整归档约 **11.11 GiB**（25个Release附件，其中22个ZIP），包含下表所有组。下载缓存与解压文件会同时占用磁盘，建议预留至少25 GiB：

```powershell
python scripts/download_resources.py --all
# 仅补充原图或CLIP：
python scripts/download_resources.py --groups originals models-clip
# 按固定清单检查已恢复文件，任何修改或缺失都会报错：
python scripts/download_resources.py --all --verify-only
```

下载支持.partial续传；若服务器不支持Range，会重新下载该附件。ZIP恢复时校验路径、文件大小和SHA256，已有同哈希文件直接保留；发现不同内容会停止，只有显式加 `--overwrite` 才替换。独立的 `--root` 和 `--archives-dir` 可用于另一份检出或本地备份。

## 包内内容

| 组 | 内容 | 默认下载 |
|---|---|---|
| `records` | 全部数据快照、来源API记录、许可CSV、采集与审计记录、训练/构建/测试日志、模型卡和许可 | 是 |
| `gallery` | 冻结4,692图 + 扩展101图的实际处理图和缩略图，共9,586文件 | 是 |
| `references` | bank43 + supplemental2 的45张实际参考照片 | 是 |
| `artifacts` | CLIP/SigLIP/DINO向量索引、4组LoRA及训练历史、teacher缓存、分类头、遮罩与特征、查询结果、评测与验收JSON | 是 |
| `models-siglip` | `google/siglip2-base-patch16-224`，revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` | 是 |
| `models-dino` | `facebook/dinov2-small`，revision `ed25f3a31f01632728cabb09d1542f84ab7b0056` | 是 |
| `models-clip` | `openai/clip-vit-base-patch32`，revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`，原始PyTorch权重 | 否 |
| `originals` | 4,793张通过许可台账的图库原图，保留实际采集位置和字节 | 否 |
| `history` | `archive/` 中已作废v1及parser修复前的历史实验，保留追溯 | 否 |

**history不是正式结果，不能用其中模型和数字替代v2。** 有些历史文件指向早期数据版本，本包并不补造丢弃的候选图库。所有现有正式v2产物字节保持原样，解压恢复不会重写冻结数据、拆分、配置或报告哈希。

素材不是SQL服务器数据库：目前使用JSONL/CSV/Parquet保存藏品与标注记录，NumPy向量索引、NPZ分类头及safetensors适配器保存模型资源，均按原目录恢复。

## 模型缓存与再训练

模型快照复制固定revision的实际文件，不包含用户的Hugging Face token或其他模型缓存。图像/文字编码和参考图检索可使用这些本地文件离线运行。命令行再训练时也应指向同一缓存：

```powershell
$env:HF_HUB_CACHE = (Resolve-Path models/huggingface/hub).Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
```

Linux/macOS可设置 `export HF_HUB_CACHE="$PWD/models/huggingface/hub"`。CLIP对照需额外下载 `models-clip`。模型、GPU依赖环境与源码分开版本管理，跨设备仍需兼容的PyTorch/CUDA环境。

## 校验与可追溯性

- 仓库中的 [resources_manifest.json](resources_manifest.json)：固定每个附件的大小、文件数、用途和SHA256。
- Release中的 `curio-files.jsonl`：逐文件的相对路径、大小、SHA256及所属附件。
- Release中的 `resources_manifest.json` 与 `SHA256SUMS.txt`：方便手工核对下载；以检出源码中的清单为信任起点。
- `python scripts/package_resources.py` 可从原本地数据和固定模型缓存重新打包并重新审计。`scripts/publish_resource_release.py` 先上传draft、验证GitHub返回的附件SHA256，再按 `--publish` 发布；凭据由本机Git Credential Manager提供，不写入包。

图库及参考照片只重新分发已接受、标为CC0的文件；未接受的候选图片不上传，候选API记录及失败原因保留。逐件来源、credit和rights证据仍在 `data/licenses.csv`、原API响应与参考manifest中。SigLIP2和DINOv2遵循Apache-2.0，CLIP遵循其MIT许可；官方模型卡、许可文本及获取URL/hash在 `models/licenses/`。它们不因发布到本仓库而改换许可。

不上传：凭据、授权页面截图、私人会话快照、课程原始Word文件、活跃服务PID、虚拟环境和node_modules。测试/性能日志中记录的本地路径、硬件及时间戳作为复现信息保留。
