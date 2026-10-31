# 开发连续性：2026-10-30 参考图改进阶段

用户要求必须落实四项：真实类别扩展；专业文本找到外部参考图再做图图匹配；可信性；器型/纹理等专门视觉方法的实际比较。不要再把元数据重排或上传图片入口替代完整第二项。

新增实现：`retrieve/image_search.py`、`retrieve/references.py`、`represent/image_encoder.py`、`represent/visual_parts.py`、`represent/patch_match.py`。FastAPI端点 `/api/search/reference`、`/api/search/image`、`/api/search/expanded`。前端三种输入和研究方法选择、实际参考图来源、扩展图库选择。默认 `/api/search` 不改为元数据排名。

冻结4692保持原 SHA；扩展101/66家族已下载并建成4793索引。参考bank43与两类图库隔离，来源及审计见 `REFERENCE_SOURCES.md`，负结果及实验边界见 `REFERENCE_RETRIEVAL_RESULTS.md`。原模型训练、选择、v2指标不作废，也不外推至扩展库。

本轮实际运行8意图×13方法全4692召回，244张不同图片、257条查询/图片对做AI像素盲判。所有真实排名/分数/grade/protocol/哈希在 `artifacts/reference_eval/`；不是全库真人gold。多参考/DINO在部分器型有效，传统遮罩仅888图有效、单通道表现差，不能称专门模型普遍提升。冻结DINO patch另以独立弱标签集评测，结果见 `PATCH_MATCH.md`。

DSH已实际咨询，`dsh_reference_review_response.txt`保存回复；须结合代码甄别其事实错误。它建议参考分组及patch，不是把任务改为只做评测/报告。用户未给外部LLM密钥，当前规则不伪称调用LLM。专业补充参考写 `data/references/supplemental.jsonl`，不改bank43实验。

服务启动：`scripts/run.ps1`，当前本机8000，原有模型和DINO都已缓存。启动脚本离线；专门参考采集CLI联网。不应修改冻结选择文件来伪造新方法经过原dev选择。Python `.venv/Scripts/python.exe`；`pnpm --dir web build`可用（也可本地tsc/vite.cmd）。所有文件未跟踪，不能用空git diff判断“没有修改”。

本阶段后续研究：学习式可靠分割、独立器型/细纹饰数据、语义局部纹样识别、参考按属性分组及可靠性估计、如有实际API配置才接入LLM规划。它们不能冒记为已实现。课程报告、成员工作归属、真人核验属于用户指定的下一阶段。

最终验收（2026-10-30）：pytest353 passed/7warnings，Ruff、TypeScript/Vite build通过。真实HTTP `scripts/verify_reference_service.py` 通过，报告 `artifacts/reference_eval/api_acceptance.json`；默认检索、参考选择、DINO与patch、严格不足返回空、qualified警告、扩展4793、上传及非法输入、图片路由均通过。8000服务本轮由 `scripts/run.ps1` 启动（PID53316，启动工具session51955）。运行时参考45=bank43+supplement2；默认probe weight0。浏览器已重新加载最新bundle并实际运行康熙梅瓶的DINO patch路径，3参考、召回100、6可用候选、2/3可用参考，返回12结果，来源和边界说明显示正确。
