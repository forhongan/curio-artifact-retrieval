# Curio API

The service is created with `curio.serve.app:create_app`. Start it with
`python -m curio.serve`, or run `uvicorn curio.serve.app:app`. Configure the
catalog with `CURIO_DATA` (JSONL, default `data/snapshots/collection_v2.jsonl`), the
embedding directory with `CURIO_INDEX` (default `artifacts/$CURIO_ENCODER`),
and the model with `CURIO_ENCODER` (`clip` by default, or `siglip`). Image
embeddings are built by `python -m curio.index.build --encoder clip`; use
`--encoder siglip --output artifacts/siglip` for SigLIP. The builder writes
normalized `image_embeddings.npy`, matching `item_ids.json`, and a manifest.
Use `--adapter path/to/adapter` to build an adapted index. The manifest records
the model revision, adapter path/hash, catalog hash and dimensions; incompatible
encoders or a changed catalog are rejected. Runtime encoder loading follows the
adapter declared by the index. All search variants use the configured index,
so compare zero-shot and adapted encoders using separate index directories.

If images or the index are missing, the server starts in a degraded mode.
The default `image_only` path returns no metadata substitute and reports
`effective_variant: "unavailable"`, `retrieval_mode: "unavailable"`. Explicit
diagnostic/comparator variants can answer with BM25 over catalog metadata.
`/api/health` identifies the diagnostic capability as
`metadata_only`; results include `retrieval_mode`, `degraded`, and warnings.
This fallback is a diagnostic text search, not visual retrieval. Model loading
is lazy and never occurs on startup when vectors are absent.

## `POST /api/search`

```json
{
  "query": "明代青花瓷瓶，上面有龙纹",
  "top_k": 12,
  "variant": "image_only",
  "filters": {"object_type": [], "material": [], "era": null, "source": []},
  "stream": false
}
```

`top_k` is 1–50. The API default and recommended path is `image_only`
(rule-expanded image/text similarity plus trained image-probe probabilities).
Comparators are `baseline` (original query dense retrieval),
`translated` (rule-expanded dense retrieval), `hybrid` (dense, BM25 and slot
rank fusion), `adapted` (adapted index when available, otherwise the base
encoder with a warning), optional `visual` (hybrid with image-feature probe
soft reranking), and explicit `lexical` BM25 diagnostics. `stream` is
currently `false` only. Client-selected filters are strict and are never
automatically relaxed. An era filter is an inclusive `[start_year, end_year]`.

In every RRF channel, exactly equal scores share their average one-based rank.
Item IDs only break final display ties and never affect channel contributions.

Each result contains stable item/source links, score details, matched metadata
slots, and `evidence`. Metadata matches are explicitly labeled `kind: metadata`
and say that image visibility was not checked. Dense similarity is not proof
that a motif, material, count, or spatial relation is visible. Count and spatial
language is surfaced as unsupported visual constraints until a verifier exists.
For hybrid and translated retrieval, negated values are removed from positive
query/slot scoring; metadata conflicts
can receive a penalty, while absent tags are not treated as proof of absence.
The `baseline` comparator feeds the unmodified original query to the encoder
and does not apply metadata conflict penalties. The index coverage count is
explicit; unindexed catalog rows do not enter visual results. Cold model-load
time is included in response wall time and separately labeled in `timings`,
while process latency percentiles cover warm searches only.

`image_only` never calls BM25, metadata slot scoring or metadata negative-conflict
penalties. Explicit user facets may still restrict candidates. Its
`matched_slots` is always empty and evidence kinds are only `similarity` and
`visual_probe`; source/title text is display metadata, never image-score support.
Its breakdown marks `bm25`, `slots`, and `negative_conflicts` as null.
Specialist visual paraphrases are project-authored rules, documented in
`SPECIALIST_QUERY.md`; they do not generate additional hard slot labels.

## Optional visual probes

## `POST /api/search/level3` (research mode)

This endpoint is an explicit two-stage prototype for mixed queries such as
`明代万历青花瓷瓶`. Stage 1 performs pure `image_only` visual recall over the
gallery. Stage 2 adds only catalogue evidence for object type, material, and
date-range constraints. It returns `level3_evidence` per result with
`satisfied`, `violated`, or `unknown` status and `basis: "catalogue"`.

```json
{
  "query": "明代青花瓷瓶",
  "top_k": 12,
  "recall_k": 100
}
```

The response is marked `variant: "level3_research"` and
`retrieval_mode: "two_stage_research"`. It does not infer dynasty,
authenticity, or internal structure from pixels, does not call an external LLM
or web search, and is not the production default. Its purpose is to make the
visual-versus-catalogue trade-off measurable before adding external reference
prototypes.

The service loads probes only when `CURIO_PROBES` explicitly points to a probe
directory, or the application factory receives `probes_path`. Set
`CURIO_PROBE_WEIGHT` to a value between 0 and 1 (default `0.15`). This is an
untuned configuration default, not a demonstrated improvement. `report.json`
must match the actual image-embedding/ID hash, data hash, index version,
feature dimension, and exact index-manifest hash. A probe trained in the base
embedding space cannot be used with a different adapted index. The service
checks the trained class vocabulary and probability matrix, then predicts the
entire gallery once at startup; requests never rerun the classifier.

`image_only` computes `61 / (60 + average_dense_rank)` plus
`probe_weight × mean(covered query-class support probabilities)`. Equal dense
scores share the same average rank, before item IDs break final display ties.
Positive support is `P(class | image embedding)`; a covered negated class uses
`1 − P(class | image embedding)` as soft support, never proof of absence.
`probe_coverage.positive` and `.negations` list each canonical query class as
`covered` or `unknown`. Missing model classes never become absence evidence.
The single-label `object_type=lid` class supports only positive queries for an
independent lid as the sole object type. Lid component requests such as
`covered jar`, `带盖器物`, and every negated lid constraint remain `unknown`
with `reason: "component_presence_not_trained"` and an explanatory warning.
They contribute neither a probe bonus nor lid evidence: the available head
does not detect whether another object has a lid.

Every search response also contains a `confidence` evidence summary. It is
deliberately not a probability of correctness: `visual_coverage` reports how
many parsed visual classes have a trained probe, `ranking_separation` reports
the top-1/top-2 score gap and a gallery-relative normalized gap, and
`signal_agreement` says whether dense embeddings and probes are both available.
`level`/`label` (`high`, `medium`, `low`, or `unknown`) combine those signals
with degraded state for a concise UI explanation. Historical fields such as
dynasty, authenticity, or provenance remain unknown unless a separate verified
evidence source is added.
Probe evidence exposes the original `probability`, `support_probability` and
`negated` flag. The engine accepts any trained task in its audited vocabulary,
including material/type softmax and independent colors/shape/motifs sigmoid
heads. All predictions use image embeddings, with no catalog attributes at
inference. Sigmoid probabilities are bounded individually and need not sum to 1.

Without usable probes, `image_only` reports `degraded: true` and
`effective_variant: "dense_rewritten"`, using exact rewritten dense cosine
scores. A query without any supported classes uses the same effective variant
with an applicability warning. Missing image embeddings yield no results,
never a metadata fallback. At probe weight zero, dense order is preserved and
evaluation counts no probe contribution. `dense_rank_normalized` exposes the
dense term, while `probe`, `probe_bonus`, and `probe_weight` expose image support.

`visual` starts with hybrid RRF, including the same strict filters and negative
metadata penalties; it remains the metadata-hybrid comparator. If a parsed positive vocabulary class has a
trained class, it divides the RRF score by `active_channel_count / 61` and adds
`probe_weight × mean(query-class probabilities)`. The explicit default leaves
validation and weight selection to dev experiments. A missing or incompatible
probe returns hybrid scores with `degraded: true` and
`effective_variant: "hybrid"`. A query without covered material/type classes
also uses hybrid, with an applicability warning. `top_k` remains unchanged.

Probe evidence has `kind: "visual_probe"`, `slot`, `value`, `probability`, and
text identifying a **weakly supervised visual classification prediction**.
It is never called visual confirmation or added to metadata `matched_slots`.
`score_breakdown.probe`, `probe_bonus`, `probe_weight`, and `rrf_normalized`
expose its contribution. The raw-query dense baseline never uses probes.

`/api/health` includes probe availability, configured weight, trained tasks and
one-time `probe_load_ms`. Search `timings` include `encode_ms`, `retrieve_ms`,
`fusion_ms`, `parse_ms`, `model_load_ms`, `cold_start`, `warm_total_ms`, and
`total_ms`. Total time includes cold model loading; warm total subtracts only
that load. Probe initialization time is a startup cost reported by health,
not an unreported repeated request cost. Warm process percentiles exclude cold
load requests, as before.

`/api/metrics.runtime_config` exposes the active catalog, index and manifest
SHA256 values, model identifiers, default retrieval variant, and loaded probe
file fingerprints/weight/tasks. `runtime_environment` records the server
process ID, OS/CPU/Python information and the loaded encoder device; it omits
filesystem paths, hostnames and usernames. These fields are independent of
whether a compatible evaluation report is available.

`scripts/system_check.py` records this configuration before/after acceptance,
rejecting configuration or process changes during the run. Its
`curio-system-acceptance-v2` report retains each search query, requested/effective
variant, request ID, result IDs and wall time, including every concurrent
request. The visual summarizer matches runtime catalog/manifest/probe hashes
and weight to frozen production. Old or mismatched acceptance reports remain
visible with an explicit statement that they cannot prove the current
configuration passed; a healthy status or filename alone is insufficient.

## Evaluation configuration and restricted galleries

`python -m curio.evaluate.harness --variants baseline,image_only,hybrid,visual --probes PATH
--probe-weight 0.15 ...` explicitly enables probes for evaluation. Select the
weight on dev only from `0, 0.05, 0.15, 0.3`, using a frozen independent visual
dev audit; metadata-silver results are auxiliary. `metrics.json` records hashes of the probe report and model
files, availability, configured weight, full index/data/query/qrels provenance,
and each summary's effective variants, degraded count and probe-application
count. Per-query outputs preserve actual timings and fallback warnings. Missing
probes yield exact hybrid rankings under the requested `visual` variant and are
reported as degraded, never as evidence of a probe improvement.
An `image_only` run with `dense_rewritten` effective variants similarly reports
a dense fallback rather than a probe improvement. Per-query and ranking files
retain the covered/unknown image-attribute list.

For an exhaustive visual audit over a known subset, pass `--gallery-ids PATH`
where PATH contains a nonempty JSON list of unique catalog IDs. The original
full index and full-gallery probe hashes remain unchanged: only search candidates
are restricted. Every query's qrels must explicitly grade exactly every gallery
ID, including grade 0; missing or extra IDs are rejected before model loading.
Outputs include sorted `gallery_ids`, canonical `gallery_sha256`, `gallery_count`
and `qrels_scope`. Label provenance remains whatever the qrels actually declare,
for example `ai_visual_audit`; it is not relabeled human. Comparisons require
identical gallery fingerprints/counts as well as identical queries and qrels.

## Other endpoints

### External reference and image routes (2026-10-30)

`POST /api/search/reference` takes JSON, for example:

```json
{"query":"康熙梅瓶","top_k":12,"limit":4,"method":"dinov2_patch","aggregation":"mean"}
```

The bilingual rule planner selects existing CC0 external museum references from
the locally collected bank (currently 43 frozen + 2 supplemental). This is not
live web search or an LLM call. Optional `reference_ids` explicitly select bank
entries. Supported methods: `siglip`, `dinov2`, `dinov2_patch`, `shape`, `color`,
`texture`, `siglip_shape`, `siglip_color`, `siglip_texture`, `ensemble`.
Aggregations are `mean`, `max`, `centroid`; mean and normalized centroid have
equivalent rankings. Local candidate ranking uses image features only. Metadata
selects reference exemplars and is displayed with provenance; it does not
establish local candidates' dates. Missing references return an explicit
`reference_unavailable` response with no results, without text fallback.
Qualified references expose `reference_qualifications` and warnings.

`POST /api/search/image?method=siglip&top_k=12` takes raw image bytes (not a
multipart form), limited to 12 MB and 24 megapixels. Invalid images return422.
No arbitrary remote URL or local filesystem path input is accepted.
`GET /api/references/{reference_id}/image` serves only registered reference files.

Both image routes search the frozen 4,692-image gallery. Response diagnostics
record the actual channels' model/index provenance, pre-fusion cosine scope,
reference agreement and uncalibrated confidence. `dinov2_patch` recalls100 by
CLS and reorders candidates with accepted foreground patch features; rejected
candidates retain their positions. `patch_diagnostics` exposes coverage and
fallbacks. Its final `score` is reciprocal rank, not a correctness probability
or raw cosine; CLS and local scores remain separately disclosed.

`POST /api/search/expanded` takes the same JSON as `/api/search`, using a
separate, lazy-loaded 4,793-image research index with shared production encoder.
The added101 images have no assigned train/dev/test split, and old frozen
evaluation scores do not apply to this gallery. The response explicitly labels
its gallery scope; expansion records are also served by item/image routes.
`/api/health` reports `reference_count` and `expanded_catalogue_count`.

### General endpoints

- `GET /api/health`: catalog size, dense-index availability and versions.
- `GET /api/items/{item_id}`: catalog metadata without local filesystem paths.
- `GET /api/images/{item_id}?thumb=true`: serves a catalog-whitelisted image
  only when its resolved path is inside `data/processed/`.
- `GET /api/facets`: observed metadata facets and counts.
- `GET /api/metrics`: process-local request/latency totals and optional
  evaluation metrics JSON from `CURIO_METRICS`. Evaluation is included only
  when its index manifest, image index, probe file hashes and configured probe
  weight match the backend's actually loaded artifacts; otherwise evaluation
  is null with an `evaluation_warning` instead of unrelated model scores.
- `POST /api/feedback`: appends `{request_id,item_id,relevant,comment}` to
  `CURIO_FEEDBACK` (default `data/feedback.jsonl`). Feedback has no identity
  fields and remains local to the configured server.
