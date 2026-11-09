# 前端工作流

首页 `/#`：文字、专业名称、上传图片三个入口，按任务展开设置。每页12张，卡片打开来源详情；文字可选择相似排序或完整可变数量集合。

专业模式 `/#research`：中英文模型、集合Precision/Recall/F1、辅助消融和编码成本。图表来源和SHA256在折叠说明中。

实验档案 `/#experiments`：历史完整工具，顶部返回专业模式。

演示 `/assets/presentation/index.html`：本地HTML、图片、60秒实际界面剪辑，方向键/空格翻页、N讲者提示、F全屏。

文字：`POST /api/search`、`variant=image_only`；集合：`POST /api/search/set`；图片：`POST /api/search/image?method=siglip`；专业名称：`POST /api/search/reference/dynamic`、`mode=qualified`、`reference_limit=6`、`aggregation=uniform_mean`。
