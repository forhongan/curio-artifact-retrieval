# 笔记本演示

安装完成后，在项目根目录双击 **Start Curio.cmd**。它会启动本机服务并打开浏览器；再次双击可重新打开页面。服务地址为 `http://127.0.0.1:8000/`，关闭浏览器后仍可重新访问，重启电脑后重新双击启动。

双击 **Open Presentation.cmd** 可直接打开离线演示文稿，包含图表、讲者提示和操作视频。讲述顺序见 [PRESENTATION_GUIDE](PRESENTATION_GUIDE.md)。

## 演示顺序

1. 用文字找：输入“瓷器”，再尝试“绿色窄口瓷瓶”或“青花龙纹盘”。
2. 切换“筛选匹配集合”，查看系统判断符合条件的完整集合。
3. 用图片找：上传 `web/public/assets/showcase/cma_140168.jpg`，查看相似器物。
4. 用专业名称找：输入“梅瓶”，通过博物馆参考图检索相似器型。
5. 打开专业模式查看研究对比，或打开项目演示。

图库、模型、文字检索、上传图片检索、研究图表和演示文稿均保存在本机。梅瓶案例已保存外部 API 响应与参考照片，可离线复用；获取新的外部参考照片需要联网。页面上的博物馆原始链接也需要联网。

## 运行配置

笔记本使用 CPU PyTorch，复用 4,692 张图库的既有向量，无需重新编码图库。文字与上传图片使用 SigLIP2 视觉 LoRA；专业参考图路线使用原始 SigLIP2 与 DINO。首次查询需要载入模型，后续查询复用模型。

`logs/laptop/server.log` 保存服务日志，`logs/laptop/server.json` 记录本次服务进程，`logs/laptop/acceptance.json` 保存实际查询验收结果。`logs/laptop/runtime.json` 可指定 `model_cache` 路径。Windows 的 SentencePiece 读取模型文件时需要 ASCII 缓存路径；安装时可将缓存放在同盘英文路径，并在原位置创建目录联接。

源码通过 Git 更新，图像、模型、索引与前端构建文件通过资源包恢复。CPU 环境应先从 PyTorch 官方 CPU 索引安装 `torch==2.9.1` 与 `torchvision==0.24.1`，再通过虚拟环境执行 `pip install -e ".[dev]"`。依赖版本记录在本机 `logs/laptop/requirements-cpu.txt`。项目默认的 `uv sync` 使用 CUDA 源，CPU 笔记本请沿用已经安装的虚拟环境。
