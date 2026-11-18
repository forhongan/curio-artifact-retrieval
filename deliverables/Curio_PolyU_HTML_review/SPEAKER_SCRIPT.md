# Curio · Speaker notes

## 1. Opening

Good afternoon. We are Vision Bot, and this is Curio. Every object behind me is a real museum object, and every one can be found by what it looks like, not by its title. In seven minutes, we will show why that is hard—and how we solved it.

各位下午好。我们是 Vision Bot，这是 Curio。我身后的每件器物都是真实馆藏，也都能只凭外观而不是标题被找到。接下来七分钟，我们说明这件事为什么难，以及我们如何解决。

## 2. Three queries

This is one progression of increasing query difficulty: porcelain, then a blue-and-white vase, then the specialist form meiping. Simple forms to abstract description, and colloquial text to bilingual specialist names, describe this increasing difficulty rather than separate routes.

这是一条难度逐步提升的过程：从瓷器，到青花瓷瓶，再到梅瓶。强调从简单形态到抽象描述、从通俗文本到双语学名，两组概念都描述难度递进，不是两条独立路线。

## 3. Live demo

Rather than a recording, let us show the real front end. Run one plain Chinese query, switch to meiping, and then show the reference-image route. The ranking is based on appearance and form, not the title string.

先运行“绿色窄口瓷瓶”，再切换“梅瓶”，最后展示参考图模式。操作时只强调“同一批 4,692 张图、真实返回结果”。

## 4. Data & Evaluation

Our gallery contains 4,692 open-access museum images and 20 bilingual query intents. Relevance is graded from zero to three with titles hidden. Labels are drafted by an AI assistant from pixels only, checked by a team member, and frozen before any experiment.

4,692 张图、20 个双语查询、0–3 盲评。AI 只读像素起草，组员人工核验，之后冻结。

## 5. One record

This is one concrete record. The AI description says “a green-edged octagonal plaque with raised animal figures”; it never saw the word “mirror”. A team member verifies the label, and the same pipeline is applied to the whole gallery.

以唐代青铜镜为例，盲标注没有使用“mirror”，人工核验后才进入冻结数据集。

## 6. Representation

CLIP leaves a large English–Chinese gap. SigLIP2 narrows it, image-tower LoRA helps further, and the latest text-tower LoRA with the bilingual_anchor adapter brings the gap to 0.054. The code, weights and HTTP endpoint are on codex/text-tower-lora-handoff, served at /api/search/text-lora.

最新重点是文本塔 LoRA，而不是笼统写成 production。它把中英文差距降到 0.054，分支、权重和接口都已给出。

## 7. An idea from Lecture 09

Remember the CLIP result from Lecture 09: phrase templates such as “A photo of a [category]” improved ImageNet accuracy by 1.3 percent. We had the same idea for SigLIP2, so we tested it on our gallery.

直接借用第九讲的页面，先制造“这次也会提升吗？”的悬念。

## 8. Prompt calibration

We tested 19 templates across six families. Every result stayed within ±0.006 of the raw query. The familiar phrase has essentially no effect on SigLIP2 in this gallery, so we report the measurement rather than assuming CLIP behavior transfers.

19 个模板、6 个家族，变化都在 ±0.006 内，结论是“几乎没有影响”。

## 9. Whole-corpus sets

We want to know which images belong in the set. We score the same image against ‘green vase’, ‘red vase’ and ‘green bowl’. Each comparison changes just one detail. We combine those score differences with the original score and the scores for colour and shape. One cutoff decides what to keep. F1 rose in both languages on familiar queries. On query types left out of training, English still improved and Chinese went down. Next, what if the words alone don’t give us a clear picture?

集合检索要决定哪些图片该留下。我们对同一张图打分，先用绿色瓶子，再一次只换一个条件：红色瓶子，或者绿色碗。把对照分差、原查询分数、颜色和器形的分项分数合起来，最后用一个阈值决定收不收。在已知查询类型上，中英文 F1 都提高了。留出整类查询时，英文仍提高，中文下降。接下来，如果文字本身说不清器物长什么样，我们怎么找参考图？

## 10. Reference route

Sometimes, an abstract search phrase, such as ‘雍正粉彩瓶’, does not tell the text tower what the object looks like. To solve this, Curio retrieves photos of real objects from museum APIs as visual prototypes. It then searches our local database with SigLIP or DINO.

文本塔不稳定时，先向博物馆 API 请求真实馆藏照片，把它们作为视觉原型，再检索本地图库。

## 11. Kendi case

For kendi (军持), the baseline text search returned no results for ‘明青花军持’. With this visual route, two representative kendi ranked first and second. This route matches appearance and shape, not the exact title text.

军持案例只保留两个数字：基线 top-20 为 0，新路线第 1、2 名都是军持。

## 12. 25 hard queries

In our manual review of 25 difficult queries, SigLIP improved top-three precision by eight percentage points over text-only search. DINO performed better on some shape-based objects, but its category error rate reached 8.0 percent.

25 条困难查询上，SigLIP 比文本 top-3 高 8 个百分点；DINO 在部分器物上更强，但类别错误为 8.0%，所以默认仍选 SigLIP。

## 13. Performance

For performance, encoding the full gallery takes about 103 seconds on our RTX 3070, and this only needs to be done once. With caching, a user search takes just 1.56 milliseconds.

离线一次编码，在线缓存排序；全库 103 秒，排序 1.56 毫秒。

## 14. Closing

Three routes, one collection. Every question is answered with pixels, not titles. Thank you.

三条路线，一批馆藏；每个问题都用像素回答，而不是标题。谢谢。

