# Specialist description -> direct museum text search -> image-image retrieval

Date: 2026-11-08. Runtime: `dynamic-reference-runtime-v1`; default mode: `direct_api_text`.

**The default path sends the exact specialist search text directly to external
museum APIs, uses their returned usable photographs in returned order, then
ranks the entire local gallery by image-image features.** It does not replace
museum search relevance with our own era/form/decoration taxonomy. There is no
mask, shape or color prerequisite in this direct path.

Backend integration is `POST /api/search/reference/dynamic`, which forces direct
mode. Actual loopback HTTP acceptance passed in
`artifacts/research_api_acceptance_20261108.json`: seven references, all 4,692
gallery images scored, and the audited reference-image route served verified
bytes. This integration check replayed 48 cached HTTP responses from the fresh
run below; it is not a second live acquisition. The verification script respects
the configured/default Hugging Face model cache; the project-local cache alone
does not currently contain DINO. The shared-image-cache directory creation fix
postdates the fresh run's historical source hashes.

## Reproduce the fresh direct run

```powershell
.\.venv\Scripts\python.exe scripts/search_dynamic_references.py `
  "Qianlong famille rose porcelain" --mode direct `
  --output artifacts/dynamic_reference_runtime_20261108/direct/fresh_qianlong_final `
  --reference-limit 8 --per-query 30 --page-depth 1 `
  --max-metadata-candidates 120 --top-k 24
```

For a genuinely new live run use a new output directory and omit `--cache-dir`.
For replay, use the direct run's HTTP cache with `--cache-dir PATH --offline`.
Cache reads are explicitly logged as such and retain the original acquisition
timestamp separately. Direct mode cannot replay a previously taxonomy-filtered
`--acquisition` JSON.

The original description is always preserved. For Chinese input an English
search override is explicit, for example:

```powershell
.\.venv\Scripts\python.exe scripts/search_dynamic_references.py `
  "乾隆粉彩瓷" --search-text "Qianlong famille rose porcelain" `
  --output artifacts/dynamic_reference_runtime_20261108/direct/chinese_with_override
```

This records `original_description`, `search_text`, and
`query_mapping="explicit caller override"`. There is no hidden translation or
query expansion. Without `--search-text`, even a Chinese description is sent
verbatim. Discovery may consequently be sparse; the runtime does not silently
call a rule planner and describe it as direct search.

## Exact query trace

For the fresh English example, the input and both API `q` strings are exactly:

`Qianlong famille rose porcelain`

No dynasty/date/culture/form constraints are appended as separate API filters.
The actual API URLs are:

```text
https://openaccess-api.clevelandart.org/api/artworks/?q=Qianlong+famille+rose+porcelain&cc0=1&has_image=1&limit=30&skip=0
https://collectionapi.metmuseum.org/public/collection/v1.1/search?q=Qianlong+famille+rose+porcelain&hasImages=true
```

Cleveland returned these 16 IDs, in this order:

```text
140219, 123608, 140217, 140218, 123609, 140216, 140214, 140215,
123610, 122448, 140220, 123596, 123598, 123597, 123600, 123599
```

Met returned these 22 IDs, in this order:

```text
205810, 659124, 42207, 50034, 49190, 40714, 50746, 461264,
50744, 461499, 48308, 50734, 50776, 51127, 45896, 46110,
51239, 52032, 50720, 49876, 201732, 49866
```

The runtime inspects Cleveland first, then Met, preserving each API's hit order.
It stops at the requested usable-reference count or metadata inspection budget.
This fixed source traversal is disclosed, not presented as a cross-museum
relevance score. `direct_acquisition.json` records every raw returned ID, even
when bounded inspection stops before reaching it.

The first ten Cleveland hits are local-gallery objects, so they are excluded
before downloading. `cma_140220` and `cma_123596` are the next usable external
hits. Cleveland's numeric child accessions `1944.174.1` through `.4` describe
the component dishes of the already selected `1944.174` set, so source-identity
deduplication excludes them. This is duplicate-object handling, not taxonomy
or date relevance filtering. Met inspection then supplies the remaining usable
references; the final machine-readable trace lists their exact IDs, source
positions, image URLs, source dates, metadata and download decisions.

The final run made **48 live HTTP requests, zero cache reads, all HTTP 200**,
between **10:17:56 and 10:18:09 UTC on November 8, 2026**. It inspected all 38
raw hits. Eight image downloads were attempted; `met_461499` was then excluded
by image isolation. Seven photographs remained (the requested cap was eight;
no substitute was invented):

| Reference | API position | Source title | Source date |
|---|---:|---|---|
| cma_140220 | Cleveland 11 | Bowl with Orchids | 1736–1795 |
| cma_123596 | Cleveland 12 | Set of Dishes | 1736–95 |
| met_205810 | Met 1 | Famille rose figures | 18th century |
| met_659124 | Met 2 | Plate | ca. 1755 |
| met_45896 | Met 15 | Covered Jar | not supplied |
| met_46110 | Met 16 | Basin | not supplied |
| met_201732 | Met 21 | Vase (cuvette à fleurs Courteille) | 1762 |

Missing source dates remain missing. No source-date rejection occurs in the
direct path. These dates are source provenance, not visual certification.

## What may reject an API hit

Direct mode uses only these usability/isolation checks:

- Explicit Cleveland CC0 or Met public-domain status and official image host.
- Usable image bytes/resolution (minimum side 128, maximum side at least 256).
- Gallery object/source IDs, accession/official related-object families, exact
  image hashes, aHash distance <=4, and pHash distance <=6.
- Duplicate reference photograph hashes/perceptual hashes and the same source
  accession family. Cleveland set/component accessions share a family; Met's
  numeric accession components are preserved as distinct identities.

Source titles, medium, culture, dates/reigns and complete original metadata are
saved for review. **An off-topic API hit is not rejected merely because our
own concept/date rules disagree with it.** The APIs determine text relevance.
Consequently this route can include an irrelevant photograph; that limitation
is exposed rather than concealed with hand-selected replacements.

## Image-image ranking

Every usable external photograph is encoded with the same pinned, unadapted
SigLIP2 and DINOv2 image encoders used to build the gallery indices. Checkpoint
revisions, gallery data/index hashes and a hash-checked reference-vector
checkpoint are verified. The scorer receives reference image vectors, gallery
image vectors and opaque IDs. It receives no gallery titles, periods, dates,
catalogue tags or diagnostic relevance labels.

All 4,692 gallery vectors are scored. Mean reference cosines remain available
as baselines. The disclosed diversity aggregation uses conservative complete
link visual clusters: each pair within a cluster has SigLIP cosine >=0.80 and
DINO cosine >=0.65. A cluster receives one total vote, shared among its images.
Per channel the score is `0.75 * cluster-weighted mean + 0.25 * best-reference
cosine`. The complete rankings are fused as
`0.70 * 61/(60+SigLIP rank) + 0.30 * 61/(60+DINO rank)`.
Equal scores receive equal ranks.

This is heuristic aggregation, not calibrated confidence. A visually unusual
reference can receive excessive weight; the saved mean baselines, pairwise
similarity matrix, individual reference top-five lists and their overlaps make
disagreement inspectable. A single usable photograph is sufficient to rank the
gallery and is labelled `single_api_reference`. Zero usable external images
produces `no_usable_external_reference`. No fixed reference bank fills the gap.

## Files and integration

Canonical direct evidence is in:

`artifacts/dynamic_reference_runtime_20261108/direct/fresh_qianlong_final/`

- `direct_acquisition.json`: exact text mapping, request URLs, raw hit IDs/order,
  every inspected candidate's decision, source metadata and selected images.
- `http_requests.jsonl`: actual URL, HTTP status, time, bytes and cache state.
- `metadata/`, `images/`: official source records and selected downloaded photos.
- `reference_embeddings.npz` / `.json`: actual frozen reference vectors and
  their source-image/model signature.
- `gallery_scores.npz`: all-gallery scores and per-reference image cosines.
- `result.json`: complete image-only rankings, top result details, reference
  agreement/diversity, index/model fingerprints and limitations.
- `reference_review.jpg`, `results_siglip_mean.jpg`, `results_dino_mean.jpg`,
  `results_fusion.jpg`: actual reference/isolated-example/result photographs.

```python
from curio.retrieve.dynamic_reference_runtime import DynamicReferenceRuntime

runtime = DynamicReferenceRuntime(root, device="auto")
result = runtime.run(description, output_path, search_text=None)
```

`run(...)` defaults to `mode="direct"` and remains compatible with the existing
description/output call. The explicit `search_text` override is optional.
Retain the instance to reuse gallery indices and frozen encoders. Live API
work should run in a background job. The result has `status`, `mode`, original
description/search text, reference IDs, full `rankings`, top `results`,
per-reference cosines and diagnostics. Direct mode reports
`assistance.status="disabled_direct_mode"` and `mask_prerequisite=False`.
Keep live acquisition output inside the project root because source image and
metadata paths are project-relative.

## Image inspection and limits

I opened the initial raw-direct run's `reference_review.jpg` and
`results_fusion.jpg`. Its references visibly included a plain-looking reverse
view of an orchid bowl, the four-dish set and its component photographs, an
archival grayscale figural object, and an enamelled plate. This revealed the
source set/component duplication and led to source-identity deduplication in
the final direct run. It also shows why museum text hits are imperfect visual
prototypes: text can describe decoration that is not visible in the returned
view, and a relevant textual hit can depict a different form entirely.

The initial direct fusion sheet visibly contains many enamelled plates and
bowls, but also blue-white/white/grayscale objects. The text query does not
make these image matches visually certified Qianlong famille rose. Reference
form, composition and photographic viewpoint remain strong biases.

I also opened the **final** `reference_review.jpg`, `results_siglip_mean.jpg`,
`results_dino_mean.jpg` and `results_fusion.jpg`. The reference sheet still
contains the plain-looking bowl view (`cma_140220`), grayscale mounted/figural
objects, and the European-looking `met_201732`. These are API relevance/view
limitations; no extra filtering algorithm was added to hide them. Final
SigLIP-mean results lean heavily toward archival grayscale photographs;
DINO-mean results mix colorful decorated vessels, generic blue-white ware and
grayscale objects. Fusion returns decorated bowls/plates and grayscale wares.
Its first ten IDs are `met_52032`, `met_50034`, `met_52947`, `met_51112`,
`met_47446`, `met_50823`, `met_42302`, `met_42300`, `met_50948`, `met_49355`.
These are appearance matches, not a period-attribution result. The seven
references form seven visual clusters under the disclosed thresholds, so this
case has substantial visual diversity rather than a concentrated consensus.

Before the direct-policy clarification, a separate **curated** experiment was
run and inspected for Ming blue-white, Wanli blue-white, Kangxi meiping,
Qianlong famille rose and Liao leather-pouch flask. Those historical outputs
remain under `regression/`; they are not evidence for the raw-direct policy.
Kangxi returned useful short-necked, high-shouldered forms with some incorrect
long-neck/rounded shapes; Wanli returned generic blue-white plates/vessels;
Qianlong's single four-bowl photograph over-constrained a broad description to
cups/bowls; Liao had only isolated gallery objects and abstained. The optional
older implementation is `run(..., mode="curated")`. No direct-text baseline
comparison or further mask research was continued after the user clarified
the direct API objective.

## Mechanical validation

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_dynamic_reference_runtime.py tests/test_dynamic_references.py `
  tests/test_dynamic_visual_gate.py -q

.\.venv\Scripts\python.exe -m ruff check `
  src/curio/retrieve/dynamic_reference_runtime.py `
  src/curio/retrieve/dynamic_references.py `
  scripts/search_dynamic_references.py tests/test_dynamic_reference_runtime.py
```

The direct-mode mechanical test deliberately provides an off-topic bronze/Japan
record for the Qianlong query: it is retained as a usable API hit, proving no
bespoke taxonomy/date rejection. It verifies exact `q` transmission, explicit
Chinese/English mapping, raw hit order and set/component deduplication. A
patched mask extractor fails the test if direct mode tries to call it.
Other tests cover single-reference full-gallery ranking, duplicate weighting,
matrix validation, cache provenance and exact-target-image isolation before
encoding. No new catalogue labels or uncertain-label accuracy metrics were
created. Image similarity cannot certify dates, authenticity or technique.

Final verification: **23 tests passed; Ruff passed**. Two warnings come from
the existing shared Pillow fingerprint helper's deprecated `getdata` call.
`validation_manifest.json` in the final direct output records the commands,
source/artifact hashes, request counts and inspected image paths.
