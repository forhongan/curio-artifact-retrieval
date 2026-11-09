# 2026-11-08 算法主线与直接参考检索（最新）

用户已暂停长期goal，当前只按最新消息完成“先确认API有图及相关，再用正确图做图图”的核验。不得自行恢复长期任务或追加无关实验。新核验见 [API_REFERENCE_VALIDATION_20261108.md](API_REFERENCE_VALIDATION_20261108.md)：V&A直接搜索Qianlong famille rose vase返回21件/37幅图，21张主图均实际下载解码；全部官方说明及图片已检查，17件符合，排除2件十九世纪乾隆款、1盘、1内部年代矛盾。用17张正确参考对全4692图完成图图并实际看结果；前三项官方支持乾隆，后排仍有康熙/顺治混淆。新入口scripts/audit_reference_search.py，旧动态API尚未接入V&A核验。

后续用户明确要求标签只作为手段，停止非重点标签研究；当前主线是可变长度全库集合、实际读图、专业描述直接 API 文本搜索参考图后图图检索，以及技术研究整理。
完整整理入口为 [TECHNICAL_REPORT_DRAFT.md](TECHNICAL_REPORT_DRAFT.md)。不要把下方较早“只剩报告”描述当成最新开发范围。

- `POST /api/search/set` 默认 `VisualSetService/adaptive_evidence`：生产 SigLIP2-LoRA 全图分数，通用对照 margin、双 GMM 分布边界、可靠图像属性及像素/文本原型分项证据。`results` 是完整接受集合，`top_k` 只控制 preview。目标馆藏文字不参与决策。
- 8 条探索查询在 4,692 全库执行并实际读图，最终记录 `artifacts/visual_set_review_v7_20261108/`。复杂条件的误收减少，同时存在明显漏检；没有宣称全库集合准确率。详见 [VISUAL_SET_ALGORITHM_20261108.md](VISUAL_SET_ALGORITHM_20261108.md)。
- Sol 已完成 `DynamicReferenceRuntime.run`、`scripts/search_dynamic_references.py`。**默认 direct**：专业文本原样作为两个博物馆 API 的 q，按原始命中顺序取得参考；不作年代/器型词表筛选，不运行掩码。旧严格模式仅通过 `mode='curated'` 显式选择。
- canonical fresh 产物 `artifacts/dynamic_reference_runtime_20261108/direct/fresh_qianlong_final/`：Qianlong famille rose porcelain，48 live/0 cache，CMA16+Met22命中，7外部参考，全4,692图排序；原始 API 命中顺序、下载及同物/重复排除有逐项台账。主代理及 Sol 都已实际打开参考和三种结果图。API 搜索噪声及多物体/黑白构图偏差如实保留。
- 新动态 API `/api/search/reference/dynamic` 强制 direct，可显式 `search_text` 保留中文原输入并声明英文搜索文本；参考照片以 audit ID 对应的校验路由提供。旧 `/api/search/reference` 仍用 bank，不能说它已经是新动态路径。
- 去背景正式补评只在338图完成：[BACKGROUND_DOWNSTREAM_20261108.md](BACKGROUND_DOWNSTREAM_20261108.md)。主代理已看失败/QA图，v4没有胜过简单对照；不默认对所有图删背景。
- 19 prompt 模板的 dev 选择/test 补评无可靠增益；[PROMPT_CALIBRATION_20261108.md](PROMPT_CALIBRATION_20261108.md)。
- 修正馆藏标签仅保留作监督与补充证据，不继续扩大标注一致性研究；旧集合全局行号错位的 .03/.078 分数作废。
- 生产仍是 `artifacts/visual_selection.json` 的 **metadata-caption LoRA / siglip_adapted / image_only / weight0**。`backbone_catalogue_v4_semantic_20261108` 的误标 production 字段已修复并更新 manifest，数值预测未更改。
- 本轮全 pytest 472 passed；随后共享 HTTP 缓存目录修复及新增 API 定向测试10项通过，Ruff全库通过、前端构建通过。最终实际 loopback HTTP 验收 **passed**：`artifacts/research_api_acceptance_20261108.json`，7参考、全4,692图、参考图片路由校验通过；该验收使用48次缓存读取，不能冒记成48次新联网请求。验收脚本曾强制不完整的项目模型缓存，造成DINO加载503，移除强制路径后通过，尊重已有HF缓存配置。Sol历史validation_manifest之后仅有共享图片缓存建目录修复，历史源码哈希不代表当前源码。
- DSH本轮连接127.0.0.1被拒绝，未取得新意见；没有伪称已咨询。没有commit/push/deploy或新Release打包。

长期 goal 当前 paused（用户操作）；在结束前查看最新用户 steering 和实际验收产物，不能仅凭旧版完成清单标 complete。

# 2026-11-08 标注分歧审计（历史）

用户追问24图复标中4个器型分歧，要求原图、官方描述、实际提示词。已看原图并实际请求4个官方API（均200），保存artifacts/label_disagreements_20261108；整理docs/LABEL_DISAGREEMENTS_20261108.md及ACTUAL_PROMPTS.md。V036为辽绿釉皮囊式flasks而词表无flask；D037官方covered jar（原vase/复jar）；V052官方agate buckle而复标plate错误；V012为snuff bottle stopper，other/ornament都不精确。D037属dev，另外3图属test。

发现执行限制：TASK引用VISUAL_LABELING.md但CLI拼接后缀禁止工具/读文件，指南全文未内联，事件日志没有工具读取。因此不要声称指南被实际读取。实际依据为TASK中显式规则+20类schema；未定义vase/jar边界，缺flask/buckle/stopper。输入是6图联系表，每图最大560×355，不是逐张1024原图；QA重排联系表。83.3%为器型重复一致性而非正确率，包含词表/边界/错误因素。当前只审计不改冻结标签/分数。

# 2026-11-04 模型对照与预筛补充

用户追问CLIP/SigLIP2、编码成本、DINO/经典预排除以及生产LoRA与属性头区别。已完整实测，见 [CV_BACKBONES_20261104.md](CV_BACKBONES_20261104.md)。内部siglip一直是google/siglip2-base-patch16-224；生产是SigLIP2+合并LoRA，先前简称含糊已经纠正。

新增独立artifacts/cv_backbones_20261104，保留父选择/协议不变。同96新test/21意图，原始英文P@5：CLIP .600、真正第一代SigLIP .724、SigLIP2 .752、生产LoRA .762。第一代新下载固定revision 7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed。第一代在中文上更高；不是各语言单调升级。相同218train/48dev的LR属性控制：CLIP .838、第一代 .867、SigLIP2 .886、LoRA .819；第一代nDCG .899高于SigLIP2 .872。不要把属性头和开放文本能力当作同一个任务。

实际重新编码全4692图（batch16，RTX3070）：CLIP126.4s、SigLIP第一代113.9s、SigLIP2 104.4s、LoRA102.5s、DINO FP32 122.0s，另存256逐阶段/32单图/DINO FP16控制。离线计时包括读图/预处理/图像塔/归一化回传，排除模型加载/下载/保存；文件缓存与时钟非严格控制，不用总时间小差值宣称模型固有加速。图片向量重编码与旧四套索引余弦最小>.999999。

经典/DINO/融合预筛10/25/50/100%均测试，dev高保留要求都选100%。test保留50%时DINO平均正例保留94.3%，最差67.6%，所以不可称安全硬排除。全4692缓存向量随机交错计时：全量余弦排序1.56ms，DINO半候选重排3.99ms，融合6.28ms，共用文本编码排除在外。新脚本compare_cv_backbones/summarize_cv_backbones；产物acceptance与manifest已保存。生产未因此切换模型/引入硬预筛，新资源不在旧Release内。

# 2026-11-04 CV算法研究

Professor沟通后用户要求研究本身：经典视觉方法、仅像素图库的文本属性桥、对比/消融和质量-计算取舍，界面不是本轮目标。已完成第一轮新实验，主文档 [CV_RESEARCH_20261104.md](CV_RESEARCH_20261104.md)，不得拿旧8意图/旧60图结果替换本轮数据。

原训练218张AI像素标签；新dev48/test96按官方家族和来源分层哈希抽样，排除所有旧标注与旧参考池家族。两个独立gpt-6.1-terra CLI各标72张，另24张独立复标。初始子代理消息传输异常、CLI忽略用户配置失败已解决，实际用用户正常配置完成Terra图像附带调用；未换模型。冻结协议/新标签在eval/cv_research_20261104。复标器型83.3%，颜色共同已知95.9%，形状共同已知100%但只54/120判断，因此不能称人类gold或完美标注。

新增classical_cv.py：raw/GrabCut HOG、HSV、LBP、Hu/轮廓；全4692特征在artifacts/classical_cv，243.076秒/8worker，4492通过宽松面积检查（不是分割准确率），200回退。attribute_cv.py将固定文本属性映射到训练的LR预测，只看像素，未从图库标签或描述匹配。11组LR C只用dev选，全部选10（边界，未证明最优）；DINO/SigLIP图像头作为表示控制。

21固定意图/96新测试图，known-only P@5：raw_all .648、gc_all .762、DINO .743、dev选DINO+.25raw_all .829、SigLIP图像头 .886；杯形训练不足空结果计0。不是全4692准确率。原始SigLIP英文图文 .752；现有production展开英文 .762/中文 .648。RRF融合nDCG增益CI跨0，不能稳定胜出。扰动、源标签未知、分类AP、复标、计时保留raw JSON。级联50点估计 .781，缓存条件下比直接属性排序更慢且不省离线编码，未上线默认。

脚本run_cv_research/eval_cv_rewritten_control/audit_cv_research/summarize_cv_research/search_cv_attributes。完整pytest369/7warnings，Ruff与uv.lock检查通过。新增opencv/scikit-image依赖、research可选matplotlib，uv.lock更新。新资源不在较早同日发布的Release内；本轮未commit/push/重打包发布。默认生产仍原SigLIP/LoRA weight0。

# 2026-11-04 资源共享交付

完整资源已发布到本仓库GitHub Release `curio-resources-v2-20261104`，对应源码提交 `22f3f2f`。25个附件（22个ZIP）约11.11 GiB，包含4,793张批准的图库原图及处理图、45参考图、三套固定revision底模、训练适配器、索引、来源记录、评测与日志；默认运行下载约2.81 GiB。见 [RESOURCES.md](RESOURCES.md) 和 [resources_manifest.json](resources_manifest.json)。`python scripts/download_resources.py` 恢复默认运行资源，`--all` 恢复全部。不要因Git目录未含大文件而再误判为资源未共享。

资源打包未改变原正式v2字节或分区。空目录完整恢复28,976文件后逐项SHA256通过，SigLIP2/CLIP/DINO离线加载及图像向量一致性通过，恢复后的文本/参考图/扩展/上传API通过。362项测试及Ruff通过。后续新增研究应使用新产物目录和资源版本，不能把未进入本次快照的改动冒称为该Release内容。作废v1归档位于可选history组。

# 开发连续性：2026-10-30 参考图改进阶段

用户要求必须落实四项：真实类别扩展；专业文本找到外部参考图再做图图匹配；可信性；器型/纹理等专门视觉方法的实际比较。不要再把元数据重排或上传图片入口替代完整第二项。

新增实现：`retrieve/image_search.py`、`retrieve/references.py`、`represent/image_encoder.py`、`represent/visual_parts.py`、`represent/patch_match.py`。FastAPI端点 `/api/search/reference`、`/api/search/image`、`/api/search/expanded`。前端三种输入和研究方法选择、实际参考图来源、扩展图库选择。默认 `/api/search` 不改为元数据排名。

冻结4692保持原 SHA；扩展101/66家族已下载并建成4793索引。参考bank43与两类图库隔离，来源及审计见 `REFERENCE_SOURCES.md`，负结果及实验边界见 `REFERENCE_RETRIEVAL_RESULTS.md`。原模型训练、选择、v2指标不作废，也不外推至扩展库。

本轮实际运行8意图×13方法全4692召回，244张不同图片、257条查询/图片对做AI像素盲判。所有真实排名/分数/grade/protocol/哈希在 `artifacts/reference_eval/`；不是全库真人gold。多参考/DINO在部分器型有效，传统遮罩仅888图有效、单通道表现差，不能称专门模型普遍提升。冻结DINO patch另以独立弱标签集评测，结果见 `PATCH_MATCH.md`。

DSH已实际咨询，`dsh_reference_review_response.txt`保存回复；须结合代码甄别其事实错误。它建议参考分组及patch，不是把任务改为只做评测/报告。用户未给外部LLM密钥，当前规则不伪称调用LLM。专业补充参考写 `data/references/supplemental.jsonl`，不改bank43实验。

服务启动：`scripts/run.ps1`，当前本机8000，原有模型和DINO都已缓存。启动脚本离线；专门参考采集CLI联网。不应修改冻结选择文件来伪造新方法经过原dev选择。Python `.venv/Scripts/python.exe`；`pnpm --dir web build`可用（也可本地tsc/vite.cmd）。所有文件未跟踪，不能用空git diff判断“没有修改”。

本阶段后续研究：学习式可靠分割、独立器型/细纹饰数据、语义局部纹样识别、参考按属性分组及可靠性估计、如有实际API配置才接入LLM规划。它们不能冒记为已实现。课程报告、成员工作归属、真人核验属于用户指定的下一阶段。

最终验收（2026-10-30）：pytest353 passed/7warnings，Ruff、TypeScript/Vite build通过。真实HTTP `scripts/verify_reference_service.py` 通过，报告 `artifacts/reference_eval/api_acceptance.json`；默认检索、参考选择、DINO与patch、严格不足返回空、qualified警告、扩展4793、上传及非法输入、图片路由均通过。8000服务本轮由 `scripts/run.ps1` 启动（PID53316，启动工具session51955）。运行时参考45=bank43+supplement2；默认probe weight0。浏览器已重新加载最新bundle并实际运行康熙梅瓶的DINO patch路径，3参考、召回100、6可用候选、2/3可用参考，返回12结果，来源和边界说明显示正确。
