# Curio UI 说明

## 页面入口

- `/`：中文主界面的跨语言文本到图像检索工作台。
- `/annotate.html`：独立人工相关性标注页，由标注服务 `127.0.0.1:8001` 同源提供。

## 检索工作台

页面只通过 `/api` 调用检索后端：搜索请求发送 `POST /api/search`，启动时读取 `GET /api/facets`，点击结果后读取 `GET /api/items/{id}`。Vite 开发服务器把 `/api` 代理到 `localhost:8000`；生产构建输出到 `web/dist`，由 FastAPI 托管。

界面支持中文和英文自然语言查询、器型/材质/年代/来源筛选、Top-K、七种检索变体。“图像模型检索”默认使用图像向量与视觉属性概率；“元数据辅助”保留词法与槽位融合对照，并支持“对比基线”双列状态。每个结果显示排序、模型分数、证据摘要和匹配槽位，打开右侧抽屉可看完整对象信息与来源链接。标题、年代、材质等字段统一标为“馆藏元数据”；证据中的图像相似度由后端 evidence 的 `kind=similarity` 决定，前端不自行补造结论。

加载、错误、降级和空结果均有独立状态。未连接后端时不会填充示例馆藏或虚构指标，示例只作为可点击的查询文本。

## 人工标注页

标注页由独立服务 `127.0.0.1:8001` 提供并使用同源 `/api`：

- `GET /api/task?annotator=...` 返回该标注者随机未标候选（跳过时附带 `skip_task_id`）：`task_id/qid/text/image_url/remaining/judged/total`。
- `POST /api/judgment` 追加 `{task_id, annotator, grade: 0..3, reason}`。
- `GET /api/progress` 用于服务端统计（页面当前仅显示 task 返回的进度字段）。

页面隐藏馆藏元数据、系统分数和自动相关性建议；标注者必须显式填写姓名或编号。键盘 `0/1/2/3` 可选等级，`Enter` 提交；“跳过这张”不会写入判断。等级依据图像可见证据，四级规则写在页面底部。

## 本地运行

```powershell
cd web
pnpm install
pnpm dev
# 生产构建
pnpm build
```

构建结果位于 `web/dist`。字体使用系统字体栈，页面不依赖远程字体服务。
