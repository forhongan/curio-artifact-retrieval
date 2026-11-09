# Dynamic museum-reference retrieval v4

Date: 2026-11-08

The pipeline starts from the user description, plans positive visual concepts,
queries Cleveland and Met public APIs, checks source rights and structured
metadata, removes elements of the frozen 4,692-image target collection, applies
a visible-evidence gate, then ranks the full target collection with SigLIP2 and
DINOv2 image vectors. It never reads a preset answer. Target metadata is used
only after image ranking to create a diagnostic relevance pool.

## Reproducible outputs

The authoritative cached runs are:

- Strict path: `artifacts/dynamic_reference_v4/delivery_strict/`
- Fixed supplementary path: `artifacts/dynamic_reference_v4/delivery_supplement/`

```powershell
.\.venv\Scripts\python.exe scripts/run_dynamic_reference_v4.py `
  --output artifacts/dynamic_reference_v4/delivery_strict `
  --cache-dir artifacts/dynamic_reference_v4/final_strict4 `
  --offline --page-depth 3 --reference-policy strict

.\.venv\Scripts\python.exe scripts/run_dynamic_reference_v4.py `
  --output artifacts/dynamic_reference_v4/delivery_supplement `
  --cache-dir artifacts/dynamic_reference_v4/final_strict4 `
  --offline --page-depth 3 --reference-policy expert

.\.venv\Scripts\python.exe scripts/audit_dynamic_reference_v4.py `
  --output artifacts/dynamic_reference_v4/delivery_supplement `
  --strict-output artifacts/dynamic_reference_v4/delivery_strict
```

`expert` is a fixed supplementary reference phrase policy and is labelled
unverified when source fields fail strict validation. Remove `--offline` and
`--cache-dir` for live API requests. Query files use JSON objects with `qid` and
`query` fields; the eight defaults live in `scripts/run_dynamic_reference_v4.py`.

Every API request records URL, HTTP status, cache state, timestamp, response
type and byte count. Selected references retain original museum metadata,
CC0/public-domain status, source and discovery URLs, image SHA256, aHash256,
pHASH64, image dimensions, metadata checks, visible evidence and rejection
reasons. The audit recomputes all reference-image hashes and rechecks every
accepted reference against the target collection's IDs, accession families,
official related objects and image fingerprints.

## Reference isolation and selection

- Cleveland and Met are queried independently with bounded pagination and
  structured date filters where available.
- Only official image hosts, explicit CC0/public-domain records and images above
  the quality floor are accepted.
- IDs, source IDs, accession families, official related works, original/image
  hashes, aHash256 distance <= 4 and pHash64 distance <= 6 against all 4,692
  target images are rejected.
- Era requires structured source fields. A title, curatorial description or
  `Kangxi mark` cannot establish manufacturing date. Culture must be Chinese
  when a Chinese era is requested.
- All explicitly requested concepts are required. Forms use object identity
  fields; ceramic requests require a ceramic material; covers, lids, stands and
  fragments are excluded from full-object reference sets.
- Accession families are deduplicated within Cleveland. Distinct Met accession
  numbers remain separate references because those records denote independent
  objects.
- Metadata provides the reference search and validation layer. Target metadata
  never reaches the image scorer.

The visible gate runs before encoding and ranking. At least two references must
pass it for a full-gallery evaluation. Blue-white checks the proportion of blue
pixels; polychrome checks saturated color; celadon checks green pixels. Form
queries use a classical foreground silhouette check: meiping requires a tall
silhouette, a narrower neck than shoulder, a substantial waist and broad lower
body; jars, flasks, bowls and dishes have separate conservative shape ranges.
These are visible-evidence gates, not trained glaze, motif or period
classifiers. If segmentation abstains, the reference is rejected (`unknown`).

## Ranking protocol

Frozen SigLIP2 (`google/siglip2-base-patch16-224`) and frozen DINOv2-small
reference images are compared against every target image embedding. No text or
catalogue field is used by either scorer.

| Method | Ranking rule |
|---|---|
| SigLIP single | first reference cosine |
| SigLIP mean | arithmetic mean of all reference cosines |
| SigLIP top-3 | mean of candidate's best 3 reference cosines |
| DINO single / mean / top-3 | same rules in DINO space |
| fusion cosine | 0.75 SigLIP mean + 0.25 DINO mean |
| fusion RRF | 0.75 RRF(61, SigLIP) + 0.25 RRF(61, DINO) |

Scores are cosine or rank-fusion scores, never calibrated accuracy.

## Metric semantics

The current relevance pools are metadata intersections, not complete visual
gold. Consequently `precision@K` is not reported: its numerator and denominator
require every top-K item to be graded, and the project has no complete
all-gallery qrels. The primary diagnostics are:

- `Hit@K`: at least one target-pool item occurs in the top K.
- `Recall@K`: target-pool items recovered in the top K divided by pool size.
- `MRR@K`: reciprocal rank of the first target-pool item within K, else zero.

The supplementary aggregate covers only queries with at least two references
passing both metadata and visible gates. Queries without this coverage remain
`insufficient_visual_references`; they are not counted as method misses.

## Results

Eight real descriptions were run. Strict run:

| Query | Metadata refs | Visual refs retained | Evaluation |
|---|---:|---:|---|
| Ming blue-white porcelain | 8 | 5 | evaluated |
| Wanli blue-white porcelain | 4 | 3 | evaluated |
| Kangxi meiping | 5 | 2 | evaluated |
| Qianlong famille rose | 1 | 1 | insufficient |
| Liao leather bag-shaped flask | 0 | 0 | insufficient |
| Yuan blue-white dragon vase | 0 | 0 | insufficient |
| Southern Song Longquan celadon vase | 0 | 0 | insufficient |
| Ming Jiajing blue-and-white jar | 0 | 0 | insufficient |

Fixed supplementary aggregate over the four queries with two or more visible
references (`delivery_supplement/summary.json`):

| Method | Hit@10 | Recall@10 | Recall@20 | MRR@10 |
|---|---:|---:|---:|---:|
| SigLIP single | 0.750 | 0.056 | 0.108 | 0.750 |
| SigLIP mean | 1.000 | 0.097 | 0.138 | 0.592 |
| SigLIP top-3 | 1.000 | 0.103 | 0.138 | 0.583 |
| DINO single | 0.750 | 0.077 | 0.107 | 0.375 |
| DINO mean | 0.750 | 0.077 | 0.117 | 0.458 |
| DINO top-3 | 0.750 | 0.077 | 0.125 | 0.458 |
| fusion cosine | 1.000 | 0.105 | 0.168 | 0.573 |
| fusion RRF | 1.000 | 0.106 | 0.153 | 0.586 |

These are conditional averages over four queries, not an eight-query success
rate. The four-query bootstrap intervals remain wide. Mean pooling supplies
consensus and hits every evaluated query; single-reference retrieval is less
stable, although SigLIP single has the highest MRR in this small sample. Fusion
improves Recall@20 over either backbone alone in this sample, while SigLIP
coverage has the highest Recall@10.

## Contact-sheet review and concrete adjustments

The following images were opened and inspected:

- `delivery_supplement/reference_contact_sheets/ming_blue_white.jpg`
- `delivery_supplement/reference_contact_sheets/wanli_blue_white.jpg`
- `delivery_supplement/reference_contact_sheets/kangxi_meiping.jpg`
- `delivery_supplement/reference_contact_sheets/qianlong_famille_rose.jpg`
- `delivery_supplement/contact_sheets/ming_blue_white.jpg`
- `delivery_supplement/contact_sheets/wanli_blue_white.jpg`
- `delivery_supplement/contact_sheets/kangxi_meiping.jpg`
- `delivery_supplement/contact_sheets/qianlong_famille_rose.jpg`
- `delivery_supplement/failure_contact_sheets/{liao_leather_flask,jiajing_blue_white_jar,song_celadon}_isolated.jpg`
- `delivery_supplement/failure_contact_sheet.jpg`

Findings and applied changes:

1. Ming blue-white: all five retained references visibly show blue decoration.
   Its top rows contain convincing blue-white objects, but bowls, cups and dishes
   all compete because this broad query does not request a form.
2. Wanli blue-white: color gating removed a purple/brown object (the rejected
   reference had zero blue pixels). The remaining three references show genuine
   blue paint. Top results remain form-diverse; pixels do not establish Wanli.
3. Kangxi meiping: shape gating removed three wider objects and retained two
   clear meiping silhouettes. In the resulting top-10 rows, seven of ten are
   visually credible meiping outlines, up from one before gating. Colors vary,
   which is appropriate because the query did not specify color.
4. Qianlong famille rose: the five fixed supplementary references visibly show
   saturated enamelled decoration. The target top-10 is dominated by colorful
   enamel ceramics; dates cannot be recognized from those pixels.
5. Liao leather bag-shaped flask: API hits included the correct local Liao flask
   images, which isolation correctly rejected. After isolation no external
   references remain. The failure sheet visibly shows the actual local form;
   broad pilgrim-flask references are not an honest substitute.
6. Yuan blue-white dragon vase: no strict references passed; the corpus already
   includes those objects. Broad visual results cannot establish dragon motif or
   Yuan date. Use a separate local-patch motif ablation.
7. Southern Song Longquan celadon: selected candidates failed structured era
   coverage. The failure sheet shows that pale/brown vessels are too easy to
   confuse with celadon. Add external celadon references with verified
   Southern Song/Longquan attribution and tighter green/glaze checks.
8. Jiajing blue-white jar: failure candidates are bowls and a multicolour vase,
   visibly missing the requested blue-white jar. No full-gallery query score is
   created. Expand official-object retrieval or add a rights-cleared source.

## Fallback policy

1. Search wider official vocabulary with the structured date range retained.
2. Use the fixed supplementary expert phrase fallback only when it still yields
   at least two isolated, visually verified references; label it unverified.
3. If fewer than two remain, abstain. Do not use the old static reference bank or
   catalogue text to invent a substitute.
4. For sparse external categories, add a rights-cleared reference source with
   object-level provenance, accession and image-hash isolation, then rerun the
   same protocol.
5. Add shape masks and DINO local-patch matching for form and motif precision;
   global CLS matching cannot prove a specific dynasty, manufacture technique or
   small decorative motif.

No human annotations, full visual qrels or calibrated correctness are claimed.
The audit is a reproducible retrieval experiment with diagnostic metadata pools
and visibly reviewed contact sheets.
