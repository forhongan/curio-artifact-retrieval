# 系统结构

项目采用 Python 单仓库后端和 React 前端，离线计算产物与在线检索分离。

## 离线流程

1. `scripts/data_ingest.py` 请求博物馆官方接口，逐条验证开放许可，保存原始 JSON/图像，再生成 processed 主表、检索图和缩略图。
2. `curio.represent.encoders` 统一 CLIP/SigLIP2 编码与归一化。
3. `curio.index.build` 生成按 item_id 对齐的矩阵、ID 列表、数据及模型哈希。当前规模下精确搜索足够，不依赖向量数据库。
4. `curio.represent.train` 分别使用馆藏派生 caption 和 AI 盲图 caption，执行图像塔 LoRA；随机/冻结teacher近邻 hard batch 两种采样，按 dev 选择 epoch。
5. `curio.represent.probes` 从图像向量及馆藏弱标签训练器型/材质 softmax 头；`detail_probes` 使用 AI 盲图标注训练颜色、器形、纹饰的独立 sigmoid 头。unknown 逐属性排除。
6. `curio.evaluate` 独立运行像素视觉审计和元数据 silver 对照，保存每查询排名、指标和成对差值；真正人工标注工具作为后续组员复核入口。

## 在线流程

`query` 保守抽取已知器型、材质、颜色、纹饰、年代和否定条件。未支持的数量/空间约束会写入 warnings。baseline 直接编码原文；translated 是词典扩展，不是假称调用了翻译模型。

主分支 `image_only` 使用重写文本与图像向量相似度，加上图像分类头对已覆盖属性的概率。其评分不读取馆藏描述、BM25、年代或元数据槽位；正条件用 P，已覆盖负条件用 1−P。稠密排名的并列项共享平均名次。测试以更换馆藏文字后排序不变来验证该隔离。没有图像索引时返回 unavailable，不拿文字结果冒充视觉检索；缺失分类头则明确报告 dense_rewritten 回退。

`hybrid` 融合图像向量、BM25 和结构化字段；旧 `visual` 在其基础上加分类概率，二者均属于元数据辅助对照。原始相似度、排名分数和分类概率分别展示，不能互相当成置信度。

研究端点 `/api/search/level3` 采用显式两阶段流程：先用 `image_only` 做
视觉召回，再只对器型、材质和年代使用馆藏字段重排，并在每个结果返回
`level3_evidence`。它不会把年代、真实性或内部结构当作图像事实，也不改变
默认生产路由。

用户显式 facet 严格执行；同字段多选 OR、不同字段 AND，年代区间按重叠匹配。空结果不会自动丢掉用户筛选。否定条件不进入正向 hybrid 召回，只依据已知冲突降权，缺失标签不视为视觉否定事实。

`serve` 只从已知 item_id 提供图片，验证真实路径位于 processed 目录内，避免任意路径读取；API 详情隐藏本机文件路径。索引缺失时默认image_only返回unavailable和空结果；只有用户显式选择的文字诊断/辅助变体才可使用metadata_only。前端区分元数据、嵌入相似度和视觉分类预测，原始排序分数不显示为概率。

## 数据与接口契约

- 训练/服务冻结主表：`data/snapshots/collection_v2.jsonl`；采集工作表为 `data/processed/items.jsonl`。item_id 带来源前缀避免多馆 ID 冲突。
- 索引：`artifacts/{encoder}/image_embeddings.npy`、`item_ids.json`、`manifest.json`。
- 模型适配：`artifacts/training/{run}/best_adapter/`。
- 探针：`artifacts/probes/{encoder}/`，NPZ 数值权重与 JSON 类名，不用 pickle 执行对象。
- 视觉标注：`eval/visual_{train,dev,test}` 原始盲图判断，经哈希映射合并为 `eval/visual_labels/annotations.jsonl`。
- 视觉评测：`eval/visual_{dev,test}/{queries,qrels}.jsonl`，每split为完整判读的独立60图图库；与全库规模明确区分。元数据对照保留 `eval/queries_{split}.jsonl` 和 `qrels_{split}.jsonl`。
- 每次评测保存 `metrics.json`、`per_query.jsonl`、`rankings.jsonl`。`visual_selection.json` 冻结生产选择和正权重属性消融，test启动时锁定测试文件哈希。
- 浏览器：`web/dist` 由 FastAPI 同源服务，开发期 Vite 代理 `/api`。

详细端点见 API.md，启动与复现命令见 README。原课程 Word 文件和源数据评估保留原样；开发文档不会覆盖其规划承诺或把示例指标改成实测。
