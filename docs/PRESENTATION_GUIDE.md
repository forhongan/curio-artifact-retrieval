# 演示流程与讲者稿

打开 `web/public/assets/presentation/index.html`，或运行服务后访问 `/assets/presentation/index.html`。HTML、图片和视频可整体离线复制，无CDN依赖。

方向键/空格翻页，N显示讲者提示，F全屏。点击Start timer开始口头计时；视频页自动暂停口头计时。共11页，讲述390秒，操作展示60秒，总计7分30秒，留30秒切换余量。下文英文稿约按正常课堂语速安排，排练时用页面图表替代逐个读数字。

主线：瓷器（大类）→绿色窄口瓷瓶（组合外观）→梅瓶（专业器型）；中间用集合边界解释“找齐符合条件的图”，最后用经典CV消融和性能解释架构选择。

录制素材由本地真实界面截取并剪辑成60秒步骤展示；屏幕切换不是检索延迟基准。若替换为小组连续录屏，保持第9页video文件接口和60秒时长。

## 1. The problem in three queries · 20秒 · A

A visitor may ask for porcelain, a green narrow-necked vase, or simply a meiping. These questions grow from broad categories to visible attributes and specialist vocabulary. Curio connects all three to the same museum image collection.

讲述提示：快速引入三个问题，不介绍界面细节。让听众先理解难度递进。

## 2. Data and a meaningful definition of relevance · 35秒 · A

Our collection has 4,692 open-access images from two museums. We revised relevance around the whole request. A Chinese bottle query should not fail just because one museum calls the object a vase. Several motifs can coexist, and incomplete evidence stays unknown. Sixty reviewed development images select boundaries; 140 test images support the quality comparisons. This gives every method the same, explicit evaluation task.

讲述提示：重点是查询条件和合理兼容关系。一次交代评测范围，后面不反复补免责声明。

## 3. Level 1: learn the domain · 45秒 · B

The first level asks whether a visual encoder separates broad artifact categories. We compare CLIP, SigLIP2, our visual LoRA, and the teammate’s bilingual pooling-head adapter under the same protocol. The domain LoRA has the highest point estimate here. Its image tower learns from museum captions during training. At retrieval time, the candidate contributes its image vector. This is the representation baseline that every later component must justify improving.

讲述提示：解释训练监督与在线候选打分的区别。图表保留数值，口头只讲模型比较和领域适配的作用。

## 4. Level 2: compose appearance · 35秒 · B

At the second level, the description becomes a conjunction: form, color, and details. We keep the complete text embedding and add a small color score predicted from image pixels. The weight comes from development experiments. This uses labels to train an attribute predictor, rather than comparing the query with a stored tag list. It also lets us test whether a classical visual descriptor contributes beyond the large encoder.

讲述提示：颜色模型只是辅助分量，不把系统讲成标签检索。右图是说明用图库示例，不伪装为固定查询的结果。

## 5. Return a set, not an arbitrary quota · 50秒 · C

A ranked list always produces a fifth result, even if the fifth image is irrelevant. We therefore score every image and make an explicit membership decision. The simplest control uses one development-selected score cutoff. A logistic boundary combines cosine, generic-object margin, and standardized score. Color assistance raises English macro F1 from 0.469 to 0.583. Precision and recall are both shown in professional mode, so the operating choice is visible. The interface returns the complete accepted set and paginates it, instead of treating the first few matches as the entire answer.

讲述提示：此页是创造性工作之一。强调不定量全库集合，完整结果与显示分页分开。

## 6. Level 3: translate a name into visual evidence · 45秒 · D

A specialist word can hide several visual properties. Rather than expect the encoder to know every museum term, we ask museum APIs for reference photographs. We check source conditions and verify actual downloadable image bytes. The same-object and near-duplicate checks separate references from gallery targets. SigLIP2 and DINO then compare reference and gallery pixels. Multiple references can represent different colors and decorations of the same form. A broad object type such as “Vase” is combined with the specialist form mentioned in the museum description.

讲述提示：不讲成调用LLM。目前使用真实API文本搜索和有限词汇转写，说明取得正确图片的过程。

## 7. Meiping: the reference route in action · 50秒 · D

Here is a concrete use case. The unchanged Chinese text workflow returns five different vase silhouettes, but none has the requested meiping form. Giving the model the English name improves this to four of five. Six actual museum reference photographs then retrieve five compatible meiping forms using a simple uniform image mean. This separates the value of knowing the English term from the additional value of image examples. We selected this demonstration after six exploratory scenarios and kept the other results, including cases where text already worked better.

讲述提示：三个对照都讲清：中文现有路径、英文原始文本、6参考图均值。不把单案例当总体准确率。

## 8. Auxiliary vision should earn its place · 40秒 · E

We compared visual components rather than adding them all by default. In the controlled reference experiment, DINO fusion and foreground color improve nDCG over the domain image model, while the top-five metric tells a different story. Color also helps some text and set queries. Shape, texture and background removal remain useful targeted experiments. The design decision is to use these cues softly where measured results support them, and keep the rest available in the experiment archive.

讲述提示：说明为什么做了经典CV研究，以及结果如何影响架构选择。

## 9. A one-minute system demonstration · 60秒 · Video

Play the sixty-second demonstration. The speaking timer pauses on this slide. It shows the actual application screens without additional narration.

讲述提示：不重复讲解。视频60秒单独计时，口头讲解计时自动暂停。

## 10. Reuse the gallery, measure the online path · 40秒 · E

Finally, preprocessing cost belongs to the offline pipeline. The domain model encoded all 4,692 images in about 102 seconds on our GPU. An unchanged gallery is reused for later tests and searches. With cached vectors, full-gallery cosine ranking takes about 1.56 milliseconds, excluding text encoding. A DINO half-gallery filter and rerank took longer and removed some positives. That is why the main workflow searches the whole cached gallery, rather than adding a filter simply because another model is available.

讲述提示：解释缓存，并以实际时间说明为什么不用更复杂的硬预筛。

## 11. A workflow for each question · 30秒 · E

Our contribution is a workflow for each kind of question. Domain adaptation handles categories. Composed visual evidence and learned boundaries support detailed requests and complete match sets. Real reference photographs connect specialist names to image matching. The final application makes these routes simple to use, while the professional workspace exposes their measured comparisons. Thank you.

讲述提示：用三条路线收束，交给提问。

## 演示前检查

1. 运行 `scripts/run.ps1` 并打开首页，提前完成一次“绿色窄口瓷瓶”和专业模式“梅瓶”查询，准备好模型与API缓存。
2. 验证第9页本地视频可播放；讲者提示默认关闭。
3. 备用：直接打开离线HTML，所有图表和梅瓶对照图片都保存在演示资产中。

## 操作展示的六个镜头

首页与三种入口 → 文字外观结果 → 完整集合与分页 → 博物馆参考照片和梅瓶结果 → 上传图片相似检索 → 专业模式实测图表。
