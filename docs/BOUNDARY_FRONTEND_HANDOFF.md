# 全库 Yes/No 边界交接给前端 · v15

负责范围：集合边界算法、完整异步判定接口、双语查询计划、逐条件证据及测试。本交接分支不修改 `web/` 中已有页面、样式或组件。

## 前端同学在 Windows 上先跑通

安装 **Python 3.12（64 位）**。下载 Release 附件 `curio-boundary-v15-windows.zip`，解压到普通文件夹，双击 `run-boundary-windows.cmd`。首次启动会创建 `.venv-boundary` 并从 PyPI 安装轻量依赖，保持命令窗口开启。

- 接口文档：<http://127.0.0.1:8015/docs>
- 健康检查：<http://127.0.0.1:8015/api/health>
- 查询范围：<http://127.0.0.1:8015/api/boundary/capabilities>
- 改端口：PowerShell 执行 `powershell -ExecutionPolicy Bypass -File scripts/run_boundary_windows.ps1 -Port 8016`。
- 自动验收：保持服务运行，在另一个终端执行 `.venv-boundary\Scripts\python.exe scripts\verify_boundary_handoff.py`，应输出 `"passed": true`。

**Windows 包是已验证结果的真实回放，不在 Windows 上执行新的视觉模型推理。** 它含 4,692 张展示缩略图、全部馆藏详情和下列 8 组意图的逐图 Yes/No。接口返回 `execution_mode: "snapshot_replay"`、`live_inference: false`。不支持的查询明确返回 HTTP 422，不会假装扫描并给出全 No。缩略图仅用于网页显示，原决定使用原图计算。

| 中文 / English | Yes | No |
|---|---:|---:|
| 青花瓷 / blue and white porcelain | 295 | 4397 |
| 梅瓶 / meiping vase | 90 | 4602 |
| 葫芦瓶 / gourd-shaped vase | 39 | 4653 |
| 双腹葫芦瓶 / double-gourd vase | 36 | 4656 |
| 三节葫芦瓶 / triple-gourd vase | 1 | 4691 |
| 红色碗 / red bowl | 3 | 4689 |
| 绿色瓷瓶 / green porcelain vase | 29 | 4663 |
| 绿色窄口瓷瓶 / green narrow-mouth porcelain vase | 10 | 4682 |

已有源代码 checkout 的同学也可把这个 zip 解压到 checkout 根目录。包内路径与分支源码一致，展示数据位于 `data/boundary_v15/`。无需安装整个项目的 Torch/CUDA 依赖，也无需下载 5 GB 的 MLX 模型。

## 接到自己的网页

复制 `integration/boundary-client.ts`，该文件不包含页面和样式。假设自己的 Vite 前端运行在 `http://localhost:5173`：

```ts
import { boundaryClient } from './boundary-client';
const boundary = boundaryClient('http://127.0.0.1:8015');
const controller = new AbortController();
const job = await boundary.search('绿色窄口瓷瓶', progress => {
  // 展示 progress.processed / progress.total；未完成时不要展示为完整集合。
}, controller.signal);
const items = job.results ?? []; // 完整集合，前端自行分页，不截断成固定 Top-K。
const firstImage = items[0] && boundary.imageUrl(items[0].image_url);
// 用 <img src={firstImage}> 渲染；相对图片 URL 必须加后端地址。
// 取消：controller.abort();
// 点击藏品：await boundary.detail(item.item_id)
// “本次判定依据”使用结果 item.condition_evidence，详情接口只返回馆藏资料。
```

组件渲染 `item.title`、`image_url`、`condition_evidence` 即可。`source: "catalogue"` 表示馆藏字段证据，`source: "image"` 表示图像证据。`value` 包含原字段内容或口部比例说明。模型分数不等于正确率，不要显示成“100% 准确”。

API 使用方式与实时后端一致：

1. `POST /api/search/set/jobs`，JSON `{"query":"梅瓶","method":"catalogue_visual"}`，返回 HTTP 202 和 `job_id`。
2. 若 `complete` 为 false，每秒 GET `/api/search/set/jobs/{job_id}`。
3. `queued/running` 显示进度；`failed` 显示 `error`；`cancelled` 停止等待。只有 `complete: true` 才能展示完整 `results`。
4. `POST /api/search/set/jobs/{job_id}/cancel` 取消未完成任务。回放结果已完成，取消是无操作。
5. 需要完整逐图判定时 GET `/api/search/set/jobs/{job_id}?decisions=true`。默认响应不传 4,692 条判定以减少流量。
6. `GET /api/items/{item_id}` 取馆藏详情，`GET /api/images/{item_id}?thumb=true` 取图片。

不要使用旧同步 `/api/search/set` 调用 v15。v15 通过上述 jobs 接口接入，必须显式传 `method: "catalogue_visual"`；不要硬编码 job ID 前缀。

Windows 快速请求示例：

```powershell
$body = @{query="梅瓶";method="catalogue_visual"} | ConvertTo-Json
$r = Invoke-RestMethod http://127.0.0.1:8015/api/search/set/jobs -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
$r.matched_count  # 90
$r.results[0]
```

跨域默认允许 localhost/127.0.0.1 的 5173、3000 端口。其他端口在启动前设置：

```powershell
$env:CURIO_ALLOWED_ORIGINS="http://localhost:5174,http://127.0.0.1:5174"
```

生产网页推荐把 `/api/` 代理到后端，使用 `boundaryClient('')`。不要从 HTTPS 网页直接请求 HTTP localhost，也不要把模型依赖放进浏览器。

## 新查询与实时算法

算法源文件：`catalogue_query.py`、`catalogue_visual.py`、`joint_sets.py`、`joint_vlm.py`，位于 `src/curio/retrieve/`。主服务接入位于 `src/curio/serve/app.py`。复核与局限位于 `eval/boundary_catalogue_v15/README.md`。

实时视觉模型当前为 **Apple Silicon MLX**，尚未提供 Windows/CUDA 等价实现。Windows 前端要处理结果包以外的新查询，需要连接运行该后端的 Mac 服务；仅下载结果包不会获得实时推理能力。新查询仍受解析器支持范围和模型能力限制。把 MLX 换成另一种模型需要重新验证分数与阈值，不能沿用本轮准确性结论。

在已有完整项目资源的 Apple Silicon Mac 上，checkout 本分支，解压 `curio-boundary-v15-runtime.zip` 到仓库根目录，然后：

```bash
# 已安装原项目环境、原始图像与 SigLIP 索引时：
bash scripts/setup_joint_vlm.sh
bash scripts/run_local.sh
# 本机实时 API 默认 http://127.0.0.1:8000
```

缺少原图/原索引时，用仓库现有资源恢复器 `python scripts/download_resources.py --list` 查看原 v2 Release 分组，再按项目原说明恢复。新 runtime zip 包含 v15 需要的冻结颜色模型、专业器类支持索引、4692 图元数据和模型版本记录；不包含原图、SigLIP 权重或 MLX 5 GB 模型。MLX 模型由安装脚本按固定版本下载。不要把 Windows 的 384px 缩略图替换进实时图库。

远程接入可通过同源反向代理，或 SSH 本地转发：`ssh -L 8015:127.0.0.1:8000 user@mac-host`。后端默认只监听本机；不要未经访问控制直接公开到互联网。

## 判定规则与边界

器类、材质允许馆藏记录提供证据，但必须检查照片主体。颜色只能由照片支持，黑白/不可观察照片不能证明颜色。窄口需可见开口，并通过口缘/器身对称轮廓比例 ≤35% 的严格工程门槛。No 包括明确不符和证据不足；不会再把所有缺乏证据的结果堆进待核验集合。

8 组意图各覆盖 4,692 图，中英文语义计划一致。交接分支 651 项测试通过、13 项跳过；独立 TypeScript 客户端在 Chrome 中的双语查询、图片、详情、跨域及进度/取消检查通过。特殊葫芦瓶仍会漏收，35% 门槛尚未独立人工校准。Windows 发布包的 API 已从 ZIP 恢复并在干净的轻量 Python 环境验收，启动脚本未在真实 Windows 机器上执行；请按文档做第一次 Windows 启动验收。

本包不修改前端作者的页面，不包含原作者本地前端改动。Release 校验文件为 `SHA256SUMS.txt`；Git 中的 `docs/BOUNDARY_RELEASE_MANIFEST.json` 固定附件哈希。不要混用不同版本的源码、资源和结果快照。
