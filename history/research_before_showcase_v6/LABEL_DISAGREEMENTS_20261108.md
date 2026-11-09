# 四例器型标注分歧核查

2026-11-08。用户要求展示原图、核查官方描述、说明实际标注提示词。本页只做审计，不改冻结标签、选择文件或原分数。

![四张存在器型分歧的原图](../../artifacts/label_disagreements_20261108/four_disagreements.jpg)

| 别名 | 原始标注 | 独立复标 | 本日官方API核查 |
|---|---|---|---|
| V036 / cma_170279 | jar，中置信 | vase，中置信 | Cleveland：Pair of Leather Bag-Shaped Flasks with Covers；辽916–1125，绿釉陶；说明造型仿皮囊，边缘细节仿缝线。类别表没有flask，不能强行声称jar/vase有一方完全正确。|
| D037 / met_47660 | vase，高置信 | jar，高置信 | Met：Covered jar with plum blossoms；objectName=Covered jar；清，18世纪初，景德镇青花瓷。若用官方类别，复标jar吻合；可见器形的vase/jar边界仍需明确定义。此图属dev，不是test。|
| V052 / cma_125743 | other，低置信 | plate，中置信 | Cleveland：Buckle；Jewelry，agate，1900s，8.4cm。是带扣，不是盘。原标注在函数身份不清时弃权比强判plate合理；类别表没有buckle。|
| V012 / cma_101807 | other，中置信 | ornament，中置信 | Cleveland：Snuff Bottle (stopper)；清1644–1911，5.4cm。图为鼻烟壶塞及附着的勺状部件，不是完整瓶身。other/ornament都未准确表达stopper，词表也无该项。|

四个API本日均返回200。原始响应、提取JSON、来源URL和SHA256在`artifacts/label_disagreements_20261108/official_fetch_manifest.json`。官方功能身份用于此次事后核查，不伪称原AI看图时已经知道它，也不把当前官方文字混入原像素实验。

## 提示词与实际输入

原始`eval/cv_research_20261104/annotator_{a,b,qa}/TASK.md`保存任务正文，`scripts/run_terra_cv_labels.py`拼接调用后缀并附JSON schema。`artifacts/label_disagreements_20261108/ACTUAL_PROMPTS.md`重建三次实际prompt文本；附原始schema可核查枚举与必填项。

- 实际完成标注的是独立`codex exec --model gpt-6.1-terra`会话；早期子代理通信失败后改用此方式，并非本日重新调用子代理。
- A/B各12张联系表、72幅图；QA为4张联系表、24幅图。每张联系表6图，单个图片框最大560×355，附盲编号；并非逐张输入1024像素原图。QA重排联系表，因此同一原图的上下文位置有所变化。
- 任务正文要求仅看像素，禁止读catalog/blind_map/他人标注/预测；颜色排除背景，灰度颜色未知；窄口≤最大腹宽一半或明显细颈；敞口为碗盘类；圆腹是球/卵圆腹，不是圆盘；可见耳/柄算handles，普通圈足不算footed；未知null，不猜负例，不推断年代材质。置信度与不确定性说明必填。
- 器型schema只有20个选项：bowl/plate/vase/jar/cup/teapot/ewer/pot/figure/mirror/box/textile/ritual vessel/pendant/stele/ornament/cong/bi/lid/other；没有flask、buckle或stopper，也没有vase/jar的操作性边界。
- **执行缺陷**：正文写着先读docs/VISUAL_LABELING.md并用工具逐表保存，但实际调用末尾覆盖为“Work only from these images, without using tools or reading any files”和“Do not write files”。指南未内联，实际事件日志没有读取指南的工具调用，因此不能声称该全文已传递给标注模型。模型实际依赖TASK中明写的规则与schema；它返回JSON，父进程保存，不是模型逐表写盘。
- `confidence`是整条记录的主观high/medium/low，不是每属性概率。D037两次均high却不同，说明不能把它当校准可靠性。

## 对实验结论的影响

83.3%=24图中20图的器型字符串相同，仅表示这套输入格式/类别表下的重复一致性。4例包括词表不覆盖、类别边界未定义、图片视角不足和错误识别，不能统一归为随机AI错误。

现有主分数中的other按未知处理，不被偷偷换成正确的官方类别。不能一边看test结果，一边只修有分歧的4例后把新分数称为原独立测试。后续应另建版本：明确可见外形和官方身份两套标签；补足flask/stopper/配件/多对象等覆盖和弃权规则；内联所有说明，统一图像分辨率和输入格式；独立复核后重新评测。官方元数据也要区分完整器物/部件，不能仅凭标题出现Bottle把stopper映射成vase。
