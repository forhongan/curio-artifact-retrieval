# 馆藏标签语义纠错与可追溯差异

对分类误判图像及其官方记录的审阅发现，馆藏原文是权威依据，但从原文自动抽取单标签的代码可能丢失材料或把用途、宗教背景误读为纹样。旧版 v3 冻结结果予以保留；新版 `artifacts/cv_v4_20261108/catalogue_v4_semantic_review/` 只纠正自动派生标签，不改馆藏原始数据。

| 样例 | 馆藏文字与图像 | 旧派生标签 | 新派生标签 |
|---|---|---|---|
| CMA 98897、98898 | Metal Plate；metal (brass?); lined in blue silk；图像显示金属护片 | silk | metal、silk；主材未凭语法强判，单标签任务记未知 |
| Met 48103 | porcelaneous ware with silver band on edge；图像为碗 | silver | ceramic、silver；混合材料单标签记未知 |
| CMA 135020 | Bird feeder；图像是鸟食器，不表明有鸟纹 | bird motif | 未知；bird 在标题中表示用途 |
| CMA 114210、121894 | Buddhist 指经文/法轮；图像不是人物纹 | figures motif | 未知；不以 Buddhist 一词推断人物图案 |
| CMA 155644 | Miniature Votive Stupa，馆方宽泛分类 Sculpture | figure | 未知；Sculpture 无法唯一决定人物器型 |

具体语义调整为：识别 broad `metal`/`brass` 和 `porcelaneous`；多材都保留，避免把衬里或配件作为唯一材质；去掉泛 `Buddhist` 与 `Sculpture` 的人物映射；保守识别 `bird feeder` 的用途。它们是可解释的通用规则，并非对上述 ID 硬编码。`metal` 是大类而非把 brass 肯定为具体成分。

在 4,692 件中，242 件至少一个兼容标签发生变化：材质 127、器型 83、纹样 38，类别之间有重叠。`semantic_changes.jsonl` 逐项记录完整官方字段及变化前后 facet；`semantic_changes_summary.json` 固定两版哈希。新版分类与检索指标需要重新训练和重算；历史测试集此前已暴露，不应称作新盲测。

```powershell
.\.venv\Scripts\python.exe scripts/build_cv_v4_catalogue_v2.py --output artifacts/cv_v4_20261108/catalogue_v4_semantic_review
.\.venv\Scripts\python.exe scripts/audit_catalogue_semantic_changes.py
.\.venv\Scripts\python.exe -m pytest tests/test_catalogue_v4_v2.py -q
```
