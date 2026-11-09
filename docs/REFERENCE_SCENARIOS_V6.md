# 外部参考图场景补测 · 2026-11-09

在看结果前列出六个真实场景，用V&A与MIA文本API搜索，每场最多6张合格参考；比较单张、均值、最大值、多样性和查询加权五种聚合。保留API请求、下载、失败、各路排名和结果池联系表。新增中文生产查询与英文生产展开对照，图库向量全程复用。

| 请求 | 合格参考 | 读图观察 |
|---|---:|---|
| Chinese double gourd vase / 葫芦瓶 | 6 | 文本已匹配良好；多样性聚合引入部分单腹瓶，颜色装饰干扰器型 |
| Chinese meiping vase / 梅瓶 | 6 | 中文生产0/5，英文原始4/5，参考均值5/5；适合展示专业器型的视觉实例价值 |
| Chinese blue and white moon flask / 青花抱月瓶 | 0 | 没有通过来源和照片资格检查的参考，返回明确空结果 |
| Liao dynasty leather flask / 辽代皮囊壶 | 3 | 取得正确来源参考，部分目标与其他壶相近；不能仅凭轮廓确认朝代 |
| Chinese celadon vase / 青瓷瓶 | 6 | 两路均能找到绿色瓷瓶，参考中的欧式金属附件会影响外观匹配 |
| Chinese blue and white dragon dish / 青花龙纹盘 | 4 | 来源文字提到龙，但照片所拍一面不一定呈现龙纹，文本路径更合适 |

梅瓶判据：短颈、小口、宽圆肩、腹部向足渐收；接受实际视觉相近的器型族。5个目标中4个官方记录明确写meiping，另一件CMA122410为读图接受的相容器型。对应IDs与逐项依据见 `docs/research/reference_v6/root_review.json`。这些是主智能体读图和来源辅助的探索案例评分。

关键代码改进：来源标题可能只是“Vase”，器型名在描述正文。现在允许器型正文证据，但标题还必须属于vase/bottle/flask/ewer/vessel等器物族，避免把“画有梅瓶的绘画”当成参考照片。搜索原文优先，再尝试最多两种有限器型同义词；完整查询条件始终参与来源核验。

复现：`scripts/explore_reference_scenarios_v6.py` → `scripts/compare_scenario_controls_v6.py` → `scripts/review_scenarios_v6.py`。原始产物：`artifacts/reference_scenarios_v6/`；公共结果与请求清单：`docs/research/reference_v6/`。普通前端“用专业名称找 → 梅瓶”调用同一qualified流程，默认6参考、uniform_mean。
