# 当前状态 · 2026-11-09 · showcase v6

本轮完成普通前端重设计、专业实测图表、11页HTML演示、60秒实际界面剪辑、六个外部参考图场景与五个研究交接包。没有调用子代理。

## 交付入口

首页 `http://127.0.0.1:8000/`；专业模式 `/#research`；实验档案 `/#experiments`；演示 `/assets/presentation/index.html`。

- [五个研究模块](HANDOFF.md)、[短讲稿与时间分配](PRESENTATION_GUIDE.md)：讲述390秒+视频60秒=7分30秒。
- [冻结研究结论](RESEARCH_V5_RESULTS.md)、[评测协议](RELEVANCE_V5.md)、[组员审计](TEAMMATE_MODEL_AUDIT.md)、[技术稿](TECHNICAL_REPORT_DRAFT.md)。
- [六个新参考图场景](REFERENCE_SCENARIOS_V6.md)、[本轮发布附件](SHOWCASE_RELEASE.md)。

## 当前算法

文字和上传图片：`artifacts/siglip_adapted`的SigLIP2视觉LoRA；文字image_only，探针权重0，中英文有限查询展开。全库4,692图，候选评分使用像素衍生特征。

完整集合：中文推荐dev_cutoff，英文推荐dev_color；可切换dev_calibrated。全部接受对象保存在results并完整分页，preview单独截断。边界在60张dev选择，质量计分在140张test核心。

外部参考：qualified模式，实际文本API→来源与照片核验→去重和同物隔离→SigLIP2+DINO图像匹配。默认6参考、uniform_mean；该路径用原始SigLIP2与DINO图库，文字/上传使用领域LoRA图库。

## 新补测

预列六个场景：葫芦瓶、梅瓶、抱月瓶、辽代皮囊壶、青瓷瓶、青花龙纹盘。实际查看参考/结果图与官方记录。梅瓶：中文生产0/5、英文原始4/5、6参考均值5/5；这是探索案例，完整结果留在docs/research/reference_v6。

修复Vase标题遗漏正文meiping/gourd专业器型；正文器型还须标题器物族支持。原文搜索优先，最多追加两种有限发现查询。新增防止画中器物误收的回归测试。

## 验收

完整后端511 passed（10 warnings），Ruff全src/tests/scripts通过，TypeScript/Vite构建通过。浏览器实际检查文字、完整集合分页、梅瓶6参考、上传、来源详情、专业中英文统计和11页演示。截图/日志：artifacts/frontend_v6/、artifacts/frontend_v6_pytest.log。

图表数据：web/public/assets/research-v6.json；公开小型结果：docs/research/；大资源使用Release。模型和预处理未变时复用缓存，不逐次重编码图库。

## 工作区

先读本页与HANDOFF。历史文档在history/research_before_showcase_v6/；早期v5状态/评测在history/docs_before_relevance_v5/与history/evaluation_v5_initial/。本地组员仓库与权重research/teammate_review/不进入Git。

生产启动scripts/run.ps1；环境CURIO_ENCODER=siglip、CURIO_INDEX=artifacts/siglip_adapted、CURIO_DATA=data/snapshots/collection_v2.jsonl、CURIO_PROBE_WEIGHT=0。
