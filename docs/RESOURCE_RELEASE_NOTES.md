完整资源包：对应Curio正式v2及后续参考图研究扩展。

- 4,692张冻结馆藏 + 101张扩展图片，包含处理图、缩略图和4,793张批准的CC0原图。
- 45张实际使用的外部参考照片、来源API记录、许可、数据快照与采集审计。
- 固定revision的SigLIP2、CLIP和DINOv2底模，4组领域LoRA、分类头、向量索引、遮罩与特征缓存。
- 实际训练history/run、评测排名/标签/指标、系统验收和构建测试日志。
- 单独标明的history包包含作废v1实验，仅用于追溯。

资源约11.11 GiB，默认运行子集约2.81 GiB。每个ZIP可独立解压到项目根目录；推荐使用下载脚本自动选择、续传、校验并恢复路径：

```powershell
python scripts/download_resources.py       # 默认运行资源
python scripts/download_resources.py --all # 全部资源
```

请先克隆仓库，阅读 `docs/RESOURCES.md`，再安装uv/pnpm依赖并启动。源码ZIP/tar.gz由GitHub自动生成，不包含下面的资源附件。模型与照片保留各自原许可；附件不含凭据、授权截图、私人会话或课程原始Word文件。每个附件有SHA256，`curio-files.jsonl`提供逐文件清单。
