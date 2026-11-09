# 去背景：固定分割比较与下游器型、材料实验

## 目的与方法

此前的 `eval_masks_catalogue.py` 虽然保存了 `*_shape`，但实际分类始终读取 `*_color`；因此它只能说明前景 HSV 在那个小样本上的表现，不能回答轮廓是否帮助器型识别。本次使用新的 [pixel-only benchmark](../src/curio/represent/background_benchmark.py) 和 [runner](../scripts/run_background_downstream_v4.py)，对每张相同图片固定比较：

1. 整图（no mask）。
2. 旧矩形 GrabCut，最长边 192、5% 边框、3 次迭代。
3. 边缘 Lab 背景分布、阈值、形态学处理和最大连通域（最长边 256）。
4. 自行设计的 v4：边界背景模型、中心先验、显著性、GrabCut、平面目标分支及无效掩码弃权。
5. 矩形 GrabCut（最长边 384、3% 边框、2 次迭代），保留全部前景。
6. 同上，再做闭运算、孔洞填充和最大连通域。

每种方法提取轮廓（16×16 归一化 occupancy、行列投影、Hu 矩、纵横比和紧致度等）、HSV 颜色、LBP 与 HOG 纹理；分别评估 `shape`、`color`、`texture`、三者拼接 `all`，以及 `color+texture`。标准化器仅在 train 上拟合；Logistic Regression 的 C 只通过 dev macro F1 选择，冻结后计算 test。无效 mask 的轮廓置零，颜色和纹理只在 mask 为空时回退整图；coverage 单独报告。馆藏单一值是监督及真值，多值、缺失或含糊的字段保持 unknown，未参与对应分类。整个特征提取过程不读取馆藏字段。

分类样本为旧的固定 338 张图（train 218 / dev 60 / test 60），类别在 train 至少有 5 张馆藏已知样本。它们不是 4,692 张全库结果，也不是新的盲测；test 已在其他研究中使用。实际可评估 test 数为 form 39（11 类）、broad material 52（4 类）、fine material 43（5 类），小样本稀有类使 macro F1 明显波动。test 预测出现了在该 test 子集缺席的类别，macro F1 包含这些预测类；balanced accuracy 的 sklearn 提示已保留在运行日志。颜色单独分类及 carved 技法缺乏最低支持，明确标为 insufficient，不将无描述自动当负例。

## 分割质量

同一套代码对 24 张近似轮廓/矩形掩码重新评估；这些是为了诊断而制作的近似真值，平面织物和细小部件尤其粗糙，不是独立人工 gold。`dev/test` 在这 24 张中以预先固定的哈希划分（9/15）。这些 mask 曾用于开发 v4 和探索参数，所以 test 也只是已暴露补充，不能解释为独立泛化测试。表中时间是此次单进程逐图提取掩码的均值，含各算法自身缩放，不含图片读取和特征提取；不同算法的工作尺寸不同。

| 方法 | dev IoU | test IoU | 24 张 Dice | 24 张 coverage | 秒/张 |
|---|---:|---:|---:|---:|---:|
| 整图 | 0.569 | 0.618 | 0.717 | 1.000 | 0.000 |
| 旧 GrabCut | 0.634 | 0.583 | 0.693 | 0.833 | 0.08 |
| 边界 + 连通域 | 0.696 | 0.645 | 0.765 | 1.000 | 0.01 |
| v4 | 0.617 | 0.598 | 0.728 | 0.958 | 0.09 |
| GrabCut 384 原始前景 | 0.705 | 0.655 | 0.771 | 0.958 | 0.31 |
| GrabCut 384 最大连通域 | 0.705 | 0.631 | 0.759 | 0.958 | 0.35 |

24 张的对象与平面例子都纳入均值。原始前景和最大连通域 dev 差异仅约 0.00003，不能视为具有实际意义的优胜；test 上强制最大连通域损失若干非连通细部。所有方法的 boundary F1 很低（约 0.04–0.07），加上近似轮廓的限制，不宜把绝对值解释成精细边缘精度。旧报告中“旧 GrabCut 0.693 IoU、12 秒/张”使用原始全尺寸和不同实现；它与此处 192 像素版本不是相同设置，不应混用。

## 下游分类（dev 选 C，test 固定补评）

| 馆藏任务与输入 | no mask dev/test macro F1 | dev 最优掩码 dev/test macro F1 | 说明 |
|---|---:|---:|---|
| Form，shape 单独 | .176 / .194 | 旧 GrabCut .502 / .374 | 轮廓确实含有器型信息；小样本 test 上提升约 .180。 |
| Form，all | .514 / .488 | GrabCut 384 raw .601 / .550 | 同一特征组合，去背景 test 提高约 .062；只有 39 个可评估样本。 |
| Broad material，all | .788 / .676 | no mask .788 / .676 | dev 选择整图；不支持默认去背景。 |
| Fine material，all | .337 / .170 | 旧 GrabCut .398 / .402 | test 样本仅 43，且五类高度不均衡，结果需更大样本复验。 |

各方法在 test 上的结果均保存在 [report.json](../artifacts/background_downstream_338_v4_20261108/report.json)，上表的“最优”只按 dev 选。对 form，384 最大连通域的 test 为 .587，高于 raw 的 .550，但不能据 test 反向改选。形状并非对组合必然有益：例如旧 GrabCut 的 `all` test .388，而 color+texture 为 .333；整图的对应值为 .488 和 .472，幅度均随组合变化。

纹样使用馆藏唯一 motif 的花卉/其他二分类，texture 单独输入：dev 选旧 GrabCut（F1 .800），test F1 .556；整图 dev/test .615/.375。这个 test 只有 20 张已知 motif（6 正、14 负），是探索性证据，不足以证明花纹检索有稳定提升。纹理这里是像素局部纹理描述，并不是对纹饰语义的专门训练模型。

## 失败样例与取舍

[失败样例图](../artifacts/background_downstream_338_v4_20261108/failure_examples.jpg) 将原图、近似 truth、384 raw、v4 和 border 放在一起。`cma_101159`、`cma_101168` 是画面被织物占满的平面物，矩形 GrabCut 和边界颜色模型把主要区域当作背景；v4 的平面分支能覆盖更大区域，但并非所有织物都成功。`met_460684` 的细提梁和局部结构造成轮廓偏差，最大连通域及填孔可能改变真正的空洞。`cma_101175` 等图像的近似矩形 truth 自身也无法精确描述边缘，应结合原图看，不应只凭数值判定算法。另有 [首八张完整方法对照](../artifacts/background_downstream_338_v4_20261108/qa_mask_comparison.jpg)。

## 合成背景替换及图图检索

将 24 张图中的近似前景粘贴到浅灰、深蓝灰、棋盘格三种新背景，对原图与替换图重新运行各分割，比较相同特征的余弦相似度（72 对）。整图 HSV 均值 .793；旧 GrabCut、border、v4、384 raw 分别为 .968、.948、.952、.974，说明此合成条件下前景颜色描述受背景影响较小。384 raw 的 shape 余弦 .602、v4 .642；无 mask 的 shape 恒为 1，因为整幅矩形不会变化，不能将其当作目标形状不变性。all 描述符各分量尺度不同，其 raw cosine 主要受数值尺度影响，不是经过校准的任务置信度。近似边界会把原背景残留在前景内，合成替换也不能代表实际拍摄视角和光照变化。

图图检索在同一 338 图内各 split 的已知馆藏类别子集完成：每张图作查询，排除同一 object group，以同一 form/material 类为相关，报告每查询 P@5 的均值；这是一项类别相似性的代理任务，不是专业图像检索完整相关性真值。对 form，shape 单独特征的 no-mask dev/test P@5 为 .152/.179；dev 最优与 no-mask 并列的 384 raw 为 .152/.221，不能认定存在稳定提升。`all` 下 dev 最优也是 384 raw（.143/.195），no-mask 为 .133/.163。这些值受已知类图库组成与类频率影响，未对 4,692 项全库检索作外推。细节在 [invariance_retrieval.json](../artifacts/background_downstream_338_v4_20261108/invariance_retrieval.json)。

目前证据支持将形状支路作为**可选研究特征**：在器型分类的受控样本中有收益，前景质量与任务收益并非一一对应。v4 自主算法在这组条件下没有优于更简单的 border/GrabCut；平面目标、细把手、低对比背景与遮挡仍是失败来源。不要默认在生产检索前删除背景，也不应将这些图图及分类结果宣称为文本到图像检索的 P@5 提升。

复现：

```powershell
.\.venv\Scripts\python.exe scripts\run_background_downstream_v4.py --workers 4
.\.venv\Scripts\python.exe scripts\eval_background_invariance.py
.\.venv\Scripts\pytest.exe tests\test_background_benchmark.py tests\test_background_v4.py -q
.\.venv\Scripts\ruff.exe check src\curio\represent\background_benchmark.py scripts\run_background_downstream_v4.py scripts\eval_background_invariance.py tests\test_background_benchmark.py
```
