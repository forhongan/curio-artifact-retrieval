# 数据来源、预处理与正式分区

更新：2026-10-30。实际图库 **4,692 件**，Met 2,954、Cleveland Museum of Art 1,738；5,000是采集目标，未达到的部分不计入实际数量。正式主表为 `data/snapshots/collection_v2.jsonl`，SHA256：

```text
4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68
```

## 来源与权利

采集器逐条要求 Met `isPublicDomain == true` 或 Cleveland `share_license_status == CC0`，保存官方JSON、原图URL、馆藏页面、署名和权利证据。`data/licenses.csv`有4,692条唯一item ID，与正式图库一一对应；不是只根据网站主页推断许可。原始JSON与图像保存在 `data/raw/`，处理图和缩略图在 `data/processed/`。

Met公开接口：`https://collectionapi.metmuseum.org/public/collection/v1`；Cleveland公开接口：`https://openaccess-api.clevelandart.org/api/artworks/`。候选按中国地域/文化字段筛选，最终再次核对官方记录。Met旧搜索端点受限时，实际通过Chrome访问官方新搜索取得候选，原响应保存为 `data/raw/met_browser_*.json`；下载原图仍保留官方来源，不依赖第三方图片站。

年代规则是有数值起年且 `year_from < 1912`。**162条年代区间跨越1912**，因此“起年在1912以前”不代表整个年代区间都属于古代。起年条件和中国来源字段的自动审计不等于艺术史鉴定。实际数据包含器物、雕塑、织物、绘画/书法等，部分历史照片为黑白、场景或多物件；不把全库描述为干净的单器物照。

## 图像处理与重复

原图短边至少256、长边至少512像素；处理图最长1024，缩略图最长320，不放大原图。保留长宽比、EXIF方向校正和RGB转换；模型另用各自官方processor，不在全库强制切掉器物或纹饰。CLIP/SigLIP2输入为224px，细小纹饰、摄影背景和多物件构成真实局限。

记录原图 `sha256`、处理图 `image_sha256` 和缩略图 `thumb_sha256`。历史字段 `phash` 实际是16×16 aHash，只按完全相同哈希分组；没有宣称完成pHash/Hamming距离聚类。采集时去除原始字节完全重复；三个同split处理图字节重复对仍保留并分组：`cma_122445/cma_122444`、`met_48659/met_47864`、`met_52042/met_48692`。它们原图不同，审计作为warning记录，不能宣称所有近重复均已消除。

## v1作废与v2关系分组

v1最初按ID分区并补重复边；全量官方accession、parts、related_works审计仍发现77个关联组跨split，涉及363项。该版本训练、选择和指标全部作废并归档 `archive/v1_invalid_split/`，正式v2不继承其adapter。

`scripts/freeze_grouped_catalog.py`把原图/处理图哈希、路径、已有重复组、aHash、官方accession字母部件、实际存在的父accession以及全部Cleveland related_works连接为 **4,234个连通家族**。为防止对象泄漏，整个家族只落入一个split。保留互不相交的盲dev/test图库作锚；其他组按原test > dev > train优先，任何原留出家族不进入train。

| split | 图像/记录数 |
|---|---:|
| train | 3,474 |
| dev | 568 |
| test | 650 |

正式冻结策略、迁移项和来源hash见 `data/processed/grouped_freeze_v2.json`；关系证据见 `data/processed/object_relations_v1.json`。文件名v1指其审计输入，边在v2冻结时保留。先前未使用的候选分区另存 `collection_v2_candidate_rejected.jsonl`，没有用于正式训练。

## 视觉监督与评测范围

原始盲图池240/60/60均由AI真实看图标注。关系分组后22张原train图随家族移入留出分区，从监督中排除；canonical为 **218 train / 60 dev / 60 test**，原始360条与排除台账分别保留。dev/test各24条中英意图，对各自60图穷举0–3相关性。三者不存在对象家族交叉。

这是AI图像弱监督与AI相关性审计，**没有真人gold或真人双标一致性**。60图检索指标不能声称代表4,692图全库真人检索精度；完整图库用于最终在线检索，另有元数据silver辅助对照。详见 `VISUAL_LABELING.md`、`EVALUATION_PROTOCOL.md`。

## 实际审计与复现

`data/processed/audit.json` 对正式v2实际解码全部 **14,076个文件**（原图/处理图/缩略图），核对哈希、尺寸、许可、中国来源、起年、官方关联和split策略。结果 **0 errors、165 warnings**：162条跨1912区间，加3对同split处理图重复。审计独立重构关系策略，跨split家族为0。不是只抽查存在性。

已有数据无需重下。采集器有缓存、退避、限速和失败记录；工作表重建不会覆盖冻结快照：

```powershell
.\.venv\Scripts\python.exe scripts/data_ingest.py --target 5000 --met-candidates 9000 --cleveland 2500 --workers 8
.\.venv\Scripts\python.exe scripts/rebuild_from_raw.py
.\.venv\Scripts\python.exe scripts/data_audit.py --data data/snapshots/collection_v2.jsonl
```

`data_ingest.py`的按ID初始split只是采集工作表，不可直接作为新的正式训练分区。新增数据应重新审计官方关系、冻结新版本、重建索引与模型，禁止在实验期间覆盖当前v2图像或主表。
