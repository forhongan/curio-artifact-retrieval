# Evaluation scope and reproducibility

## Formal grouped v2 snapshot

The formal catalog is `data/snapshots/collection_v2.jsonl`: 4,692 objects,
split as train 3,474 / dev 568 / test 650, with SHA256
`4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68`.
Official accession parent/part relationships, conservatively grouped Cleveland
related works, catalog duplicate links, actual image hashes and identical
perceptual hashes are isolated by connected object group. The grouped policy
preserves disjoint original 60-image dev/test gallery anchors; other groups
use their original test > dev > train priority. No held-out family moves into
train. `data/processed/grouped_freeze_v2.json` records the reassignment, and
`data/processed/audit.json` independently checks it with
`validate_object_splits`, full raw/processed/thumbnail decoding and SHA checks.
Per-item official rights evidence is in `data/licenses.csv`.

The inclusion rule is a recorded start year before 1912. The 162 ranges ending
in or after 1912 remain disclosed date uncertainty; three pairs with identical
processed bytes remain separate records within the same object group/split.
These are audit warnings, not evidence of a verified entirely pre-1912 range.

The v1 snapshot had cross-split object/part families. All v1 training, dev
selection and available evaluation results are invalid for formal conclusions
and retained only under `archive/v1_invalid_split/`. Formal v2 training starts
again from pretrained weights; v1 adapters and probes cannot be reused as
formal v2 results. Historical pilot diagnostics remain explicitly separate.

## Metadata-silver feasibility

Run the diagnostic after the final catalog rebuild, while inputs are stable:

```powershell
.\.venv\Scripts\python.exe scripts/audit_retrieval_cpu.py `
  --data data/snapshots/collection_v2.jsonl `
  --index artifacts/siglip `
  --size 5000 --repetitions 30 `
  --output artifacts/diagnostics/backend_review_v2.json
```

The unchanged benchmark builder is invoked against a byte-for-byte temporary
catalog snapshot and writes only into a temporary directory. Formal `eval/`
files are never touched. The JSON report records the catalog SHA256, actual
object count and supported predicate count; data from an active ingestion job
must not be described as a final benchmark snapshot.

Inspect each predicate's exact grade-3 support, full grade-0/1/2/3 distribution,
grade-3 fraction, and fraction relevant at the evaluation threshold of grade 2.
At least three genuine metadata matches are required by the existing builder.
A query for which all gallery objects are relevant cannot distinguish retrieval
systems and should not be adopted. Broad-query flags are diagnostics rather
than an instruction to select whichever subset improves a model's results.

`distinct_graded_qrels` and the reported identical-label groups reveal queries
that differ in syntax or a nominal era but induce exactly the same graded
relevance vector. The benchmark builder retains one deterministic representative
per informative group, preferring a simpler predicate, before any split is
written. It rejects queries where every gallery item is relevant. Never count
a translation or paraphrase as a new semantic query.
Report both the number of distinct predicates and the smaller number of
distinct label vectors. `grade3_item_set_sha256` separately identifies coinciding
strict positive sets whose lower grades may still differ.

The report also lists actual type/material/motif/era conjunctions supported by
at least three objects. These are suggestions for a documented future protocol,
not automatically appended queries. If the catalog cannot support 120 useful
intents, retain a smaller transparent set or collect more genuinely different
objects and queries. Do not lower evidence requirements merely to reach 120.
Metadata-derived labels remain silver catalog constraints and are not evidence
of independently assessed visible image relevance.

`scripts/run_experiments.py` checks benchmark capacity before GPU indexing or
training. Its explicit `--dev-queries` and `--test-queries` options default to
40 and 80; unsupported totals fail rather than silently shrink or duplicate
queries. A smaller protocol must be chosen openly before dev tuning. Selection
freezes hashes of both splits' query/qrels files plus the benchmark manifest;
test execution rejects later changes and stale benchmark protocol versions.

## Implemented independent image-query audit

The main CV path is `image_only`: rule-expanded image/text similarity fused with
trained image-probe probabilities. It never ranks with BM25, metadata slots,
metadata era matches or metadata conflict penalties. Explicit facets remain
candidate restrictions. `visual` is retained as the metadata-hybrid comparator.
Choose the image-only probe weight on frozen visual dev data from
`0, 0.05, 0.15, 0.3`; silver catalog constraints are auxiliary evidence, not the
main selection target. Freeze the selected weight before the visual test audit.
The protocol is `visual-v2-grouped-all-image-candidates`. Production candidates
cover three SigLIP image spaces: `siglip` (base), `siglip_adapted` (metadata
caption LoRA), and `siglip_visual_adapted` (AI image-only caption LoRA). Each
space has a base-head and detail-head directory, yielding six probe groups:
`siglip`, `vision`, `siglip_adapted`, `vision_metadata_adapted`,
`siglip_visual_adapted`, and `vision_visual_adapted` under `artifacts/probes/`.
Each group's weights are valid only in its matching image space. All three
models remain image-only at inference despite their different training labels.

Three spaces × two probe groups × four weights produce 24 image_only dev
candidates. Production maximizes bilingual dev nDCG@10 within that family;
stable ties prefer the earlier candidate and lower weight. The eight
baseline/translated settings across CLIP and the three SigLIP models remain
comparison systems; the overall best comparison is reported separately.
The head ablation uses base SigLIP's base/detail heads with one separately
dev-selected positive weight, even if production selects weight zero.

All six probe groups select C on dev with `--defer-test`; frozen test execution
uses `--evaluate-test` to predict from saved NPZ files into `test_report.json`.
It cannot fit a classifier, select C, or modify the training `report.json`.
Final selection records catalog, annotation, dev input and model/probe hashes;
`visual_test_freeze.json` additionally pins that selection and the test inputs.
The image path falls back to rewritten dense scores when probes are unusable
or no classes are covered, records `effective_variant: dense_rewritten`, and
returns no metadata substitute when image embeddings are unavailable.
Covered negative classes use `1-P` only as image-probe soft support. The
single-label `object_type=lid` head represents an independent lid, not component
presence: covered/lidded objects and every negated lid constraint stay unknown
with `component_presence_not_trained`, contributing no lid bonus or evidence.
Positive independent-lid intent is supported only when lid is the sole object
type. Missing classes are reported as unknown, never proof of absence. Per-query outputs
preserve covered/unknown positive and negative classes. Probe weight zero is
recorded as zero actual probe applications.

The query-authoring input is an image-only observation sheet, not
museum descriptions or the ontology-derived catalog columns. The authoring
steps can be reproduced as follows:

1. Freeze the catalog, image hashes, object split and seed. Select held-out test
   objects without inspecting their textual labels; keep duplicate groups
   together and record the selected IDs and hashes. A separate dev sample is
   used for any query-writing rules or rerank-weight selection.
2. Create image cards containing only the pixels and a neutral card ID. Do not
   expose title, medium, tags, dates, museum descriptions or retriever results
   to the query author. The source-object ID is retained in a separate mapping.
3. Write one natural Chinese intent, its equivalent English version, and a short
   image-visible basis for each proposed semantic query. Prefer observable
   form, color and decoration. An uncertain material or date is not asserted
   simply because the catalog supplies it. Unsupported counts or spatial
   relationships are retained only as explicitly unverified challenge queries.
4. Review semantic duplicates and freeze qids plus both texts before ranking
   systems. Hash the observation sheet and the query JSONL. Declare whether
   authorship is human or AI-assisted; tooling cannot independently establish
   that the author never saw catalog metadata.
5. Independently grade every query against every image in the explicit audit
   gallery on the 0-3 scale, including explicit zeros. Never infer nonrelevance
   from missing judgments or unseen full-catalog images. Preserve rationale and
   the true label source, for example `ai_visual_audit`. Such a set is an AI
   image audit and must not be described as human gold or inter-rater agreement.
6. Run `curio.evaluate.harness --gallery-ids gallery_ids.json` against the
   original full index. The harness requires exhaustive labels within the
   named gallery and restricts ranking to that set without changing model,
   catalog or probe provenance. The comparison tool rejects different gallery
   fingerprints. Report the restricted-gallery size beside every quality result.

This design separates query authorship from metadata-derived qrels. It does not
generate relevance labels from the source object's identity, from retrieval
scores, or from overlap between query words and catalog fields.

The original AI image-only observation pool had 360 labeled images: 240 train,
60 dev and 60 test. These original labels remain historical provenance. After
v2 grouping, 22 original train images belong to held-out families and are
excluded from formal supervision, with their aliases and reasons recorded in
`eval/visual_labels/excluded_training_labels.json`. They do not enlarge the
dev/test audit galleries. The canonical
`eval/visual_labels/annotations.jsonl` contains 338 effective labels: 218 train
for visual captions/detail attributes, 60 dev and 60 test. The effective sets
have no connected-group overlap. Each 60-image evaluation gallery has 24
bilingual semantic intents. Object-disjoint local splits do not establish that the
pretrained model never encountered these public museum images. The audit is
small, uses one AI judgment per query-image pair, and shares AI captioning and
query-authoring biases; it cannot estimate full-gallery human relevance.

## Synthetic CPU scaling

The diagnostic's 5k gallery repeats genuine indexed image embeddings under
synthetic IDs. Cached image vectors stand in for query vectors, and the optional
two-class probe has seeded random weights. No text model, GPU inference,
training or download occurs. BLAS is limited to one thread, its runtime is
recorded, each variant is warmed separately, and the gallery probe predictions
are computed once. P50/P95 are reported for parse, mock encoding, retrieval,
fusion and service wall time over the configured repeated requests.

These timings measure CPU candidate-scoring/fusion scaling on the current
shared machine. They do not measure quality, 5,000 distinct real objects,
real text-encoder latency, HTTP/frontend latency, model cold load, or trained
probe accuracy. Keep this report separate from all formal retrieval metrics.
