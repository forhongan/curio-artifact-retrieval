# 先确认参考图，再测试图图检索

2026-11-08。用户本轮明确要求：测试已经找到正确参考图的情况；先核验 API 是否实际提供图片及其相关性，不能将离题照片的图图结果当作专业检索实验。

## 实际 API 核验

直接搜索文本为 `Qianlong famille rose vase`，请求：

```text
https://api.vam.ac.uk/v2/objects/search?q=Qianlong+famille+rose+vase&images_exist=1&page_size=100
```

V&A 返回 `info.record_count=21`、`info.image_count=37`；后者是图片总数，不能当作21张独立主图之外的独立藏品数。每条命中提供 `_primaryImageId` 和 `_images` 中的主图缩略图、IIIF图片服务及manifest地址；详情提供 `images` 列表。此次21条全部有主图ID，**21张主图均实际HTTP下载成功并通过图像解码**，不是仅凭有URL认定有图。数字图片长边限768，保留 `© Victoria and Albert Museum, London` 署名和该馆教育/非商业使用条款；这些图片不冒称CC0，不加入现有开放授权训练包。

原始顺序、完整官方详情、可用性字段、实际图片、下载请求与哈希：
`artifacts/checked_api_references_20261108/qianlong_vase/acquisition.json`。

有图和符合描述是两个不同判断。主代理阅读全部21条官方brief、physical、styles、productionDates，并实际打开 `api_images_1.jpg`、`api_images_2.jpg` 的全部21张图。在查看目标排名前形成 `reference_review.json`：

- 17件：官方明确支持乾隆时期、粉彩、瓷瓶/带盖瓶；实际图与器型和彩绘相符，纳入。
- `O493984`、`O180597`：官方为十九世纪、带乾隆款；不将款识当制作年代，排除。
- `O182629`：瓷盘，排除。
- `O144417`：brief称乾隆，但日期字段1725—1734、约1730，馆藏内部信息矛盾；本次不作为已确认参考。

这是对本次 API 查询结果的来源与图片核验，**不是预先为查询建立固定参考图答案库**。未根据后续目标排名挑选参考。该核验由Codex实际读图完成，不冒称人类文物专家鉴定。

## 用符合的图片做实际图图实验

17张已核验外部参考与全部4,692张目标图，用相同冻结SigLIP2图像塔和DINO编码；比较SigLIP均值、DINO均值、视觉簇多参考聚合和融合。候选文字不参与排名，来源字段仅在排名之后用于核对结果。

已打开三种结果联系表以及逐图参考的对照联系表。融合前三项为：

| 结果 | 馆藏核对 |
|---|---|
| `met_52976` | 山水纹瓷瓶，官方Qianlong mark and period，1736—1795 |
| `cma_122448` | 官方famille rose overglaze enamel、Qianlong reign，1736—1795 |
| `met_52024` | 木兰/海棠/牡丹彩绘瓷瓶，官方Qianlong mark and period |

之后出现明确误收：`met_48597`、`met_50828`、`met_50824`、`cma_122420`官方为康熙；`met_52765`为顺治且是罐。因而参考图来源问题得到控制后，仍能观察到图像表示将外观相近的跨时期器物混淆。这才是需要评估的图图匹配能力，而不是源搜索失败。

逐图示例使用API顺序中的前四张已确认参考，未按成功与否反选：`O165239`的前三张视觉近邻全部有乾隆时期官方说明；`O34034`第一项却为铜胎画珐琅工具瓶；`O493979`前三项属于顺治/康熙。这些反例说明单张参考的近邻可能偏向配色、纹样或器型，不能把一张成功例子外推为专业身份识别正确率。

记录：`result.json`、`gallery_scores.npz`、`individual_reference_examples.json`。没有把前三项碰巧符合写成总体检索准确率，没有以目标元数据过滤排名来制造成功。

## 复现入口

```powershell
.venv\Scripts\python.exe scripts/audit_reference_search.py fetch
.venv\Scripts\python.exe scripts/audit_reference_search.py rank `
  --review artifacts/checked_api_references_20261108/qianlong_vase/reference_review.json
```

`rank`要求对应查询的实际核验记录，检查图片哈希、已下载状态、图库重合及核验理由；没有确认符合的参考图就不运行这组图图实验。该脚本是本次验证入口，旧生产动态API尚未接入V&A或该核验步骤。
