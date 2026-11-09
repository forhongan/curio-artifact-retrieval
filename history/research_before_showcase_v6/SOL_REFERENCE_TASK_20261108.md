# Sol delegated task: specialist description -> dynamic references -> image search

Work in F:/Curio (PowerShell). User explicitly requests a gpt-6.1-sol subagent
to develop this path in parallel. Do real algorithm implementation, run it,
open the actual reference/result images, improve based on observations, and
document the final behavior. Do not commit/push/deploy.

Current implementation and evidence:
- src/curio/retrieve/dynamic_references.py: live public Cleveland + Met search,
  source rights, structured period/form checks, strict target isolation.
- scripts/run_dynamic_reference_v4.py: query-file input, 8 example specialist
  queries, visual reference gates, SigLIP2 and DINO full 4692 scoring, multiple
  prototype aggregations and fusion; old catalogue diagnostic metrics.
- docs/DYNAMIC_REFERENCE_V4.md and artifacts/dynamic_reference_v4/delivery_supplement/
  include real reference sheets/results; only 4 of 8 have enough external refs.
- src/curio/retrieve/image_search.py and represent/visual_parts.py provide
  image-only shape/color/texture and DINO local matching methods.

Task:
1. Read existing code/results, improve the path as an algorithm usable for a
   user-provided specialist description, rather than a fixed answer bank.
2. Build a clean runtime module/CLI, reuse source acquisition and checkpoints.
   Target gallery ranking must use target pixels/embeddings only; reference
   museum descriptions may validate reference identity/date. References must
   not be the target, same object, or near-duplicate target photograph.
3. Explore reference consensus/diversity, semantic-vs-shape/color assistance,
   and pragmatic sparse-reference handling. Do not require 2 references purely
   by convention if a well-qualified single reference is useful; distinguish
   evidence levels and disagreements. No promise to visually certify dates.
4. Real queries include Ming blue-white, Wanli blue-white, Kangxi meiping,
   Qianlong famille rose and a sparse form such as Liao leather-pouch flask.
   Can query other public museum sources / browser if useful and justified.
   Existing cached runs may be used for regression; run at least one fresh
   user-description acquisition, with actual request provenance.
5. User explicitly wants actual read-image inspection, not more questionable
   automatic label-based benchmark scoring. Sample returned and excluded
   candidates; inspect originals/crops as useful, record concrete findings and
   fix clear algorithm problems. No new label research or AI accuracy study.
6. Write docs/DYNAMIC_REFERENCE_RUNTIME_20261108.md with algorithm, runnable
   command, outcomes, viewed examples, useful/failing aspects and limits. Keep
   all outputs in a new directory. Add meaningful mechanical tests and Ruff.

Ownership: edit dynamic_references.py, NEW runtime module(s), NEW CLI, and
related tests/docs. Avoid src/curio/serve/app.py (root agent edits it). Tell root
the runtime entry point and integration contract; root can add API if needed.
Do not touch visual_sets.py, probe_sets.py, catalogue labels/training or the
main technical report draft; root owns these. Use .venv/Scripts/python.exe.

The user's original full objective is in the thread. Main priorities now:
labels are tools, stop nonessential taxonomy work; build whole-gallery
variable-length yes-sets using full-image similarities plus reliable visual
attribute/pixel evidence as auxiliary assertions. Then finish specialist
reference path and write technical report in four stages (basic SigLIP,
detail/LoRA, specialist reference retrieval, other CV algorithms/negative results).
