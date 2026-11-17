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

A ranked list always returns a fifth item, even when it is irrelevant. We score all 4,692 images and make an explicit membership decision. The calibrated boundary with raw-color auxiliary gives the best balanced F1.

从“固定返回 top-k”改为“返回一个集合”，集合可以是 0、12 或数百张。

## 10. Reference route

The text tower knows the words but not always the look. For an abstract query such as a Yongzheng-period famille-rose bowl, we send the formal name, period and technique to museum APIs, retrieve real collection photographs as visual prototypes, qualify their provenance, and search the local gallery with SigLIP or DINO.

文本塔不稳定时，先向博物馆 API 请求真实馆藏照片，把它们作为视觉原型，再检索本地图库。

## 11. Kendi case

A kendi is the clean example. The baseline text route returned zero kendi in the top twenty. With the reference route, two representative kendi ranked first and second. The route matches appearance and form, not the title string.

军持案例只保留两个数字：基线 top-20 为 0，新路线第 1、2 名都是军持。

## 12. 25 hard queries

One case is not enough. Across 25 hard queries reviewed by hand, SigLIP improved top-three precision by eight points over text. DINO was stronger on some shape-driven objects, but its category error reached 8.0 percent. SigLIP is the steadier default.

25 条困难查询上，SigLIP 比文本 top-3 高 8 个百分点；DINO 在部分器物上更强，但类别错误为 8.0%，所以默认仍选 SigLIP。

## 13. Performance

Encoding the full gallery takes about 103 seconds once. Cached cosine ranking over all 4,692 images takes 1.56 milliseconds, and a new item needs one forward pass.

离线一次编码，在线缓存排序；全库 103 秒，排序 1.56 毫秒。

## 14. Closing

Three routes, one collection. Every question is answered with pixels, not titles. Thank you.

三条路线，一批馆藏；每个问题都用像素回答，而不是标题。谢谢。

