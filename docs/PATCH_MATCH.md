# Frozen DINO local matching: measured limits

The implemented patch matcher did not solve foreground coverage on this audit.
It accepted 181/1,403 examined images (12.90%), compared with the existing RGB
border-plane shape heuristic's 252/1,403 (17.96%) on the identical image set.
It accepted more query references, 4/18 versus 1/18, but 14/18 query rerankings
still fell back to the original DINO CLS order. Its tiny weak-label ranking
change is insufficient evidence of a retrieval improvement.

## Fixed algorithm and interface

`src/curio/represent/patch_match.py` uses the already cached frozen
`facebook/dinov2-small` checkpoint at revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`. It has no text model, fitted
segmenter, classifier training, or new large model. `DinoPatchMatcher` accepts
an existing `DinoEncoder` to share the loaded processor/model and avoid duplicate
GPU allocations. `encode_images` accepts a list of paths or PIL images and
returns patch descriptor dictionaries with diagnostics.

The model's `last_hidden_state[:, 1:]` tokens form a grid in the processor's
resized, center-cropped coordinate system. Tokens are L2 normalized. Cosine
distance from the normalized border-token centroid produces soft foreground
weights above a fixed robust border threshold. Fixed checks reject inconsistent
borders, objects at the border, small/large foregrounds, fragmented components
and weak contrast. These are untrained heuristics; passing them does not verify
the segmentation, complete object coverage, material or chronology. The
processor's center crop can remove parts of an object before these checks.

At most 96 highest-weight tokens describe the accepted foreground. Local
similarity is the mean of two soft-weighted nearest-token cosine means, one
in each reference/candidate direction (symmetric Chamfer). It allows repeated
or many-to-one matches and proves neither correspondence nor identity. The
shape diagnostic uses 32-bin soft row/column foreground profiles and a log
aspect-ratio comparison. The fixed combined score is 0.75 times local cosine
mapped to [0,1], plus 0.25 times the profile score. The weights are prototype
choices fixed before this audit, not selected for the result.

`match_patch_features(reference, candidate)` returns `local_score`,
`shape_similarity`, `combined_score` and `status`. Rejected foregrounds return
`unknown` and null scores. `rerank_patch_candidates` is intended for the first
100 DINO CLS candidates. It returns original candidate indices in the new order,
pairwise details, accepted count, coverage, and `reranked` or `fallback_cls`.
Only accepted candidates exchange their original ranking slots; unknown
candidates retain their slots. A rejected reference returns the original CLS
order explicitly. Neither unavailable scores nor rejected backgrounds become
confident local evidence. The optional `dinov2_patch` research method integrates
this module after DINO CLS top-100 recall, aggregates available reference pairs
by their mean and reports coverage/fallback. It does not replace the default
retrieval route. The audit below uses one reference per query, so it does not
validate multi-reference aggregation or the live service's latency.

## Predeclared object-form audit

Run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/eval_patch_shape.py
```

`artifacts/patch_shape_eval/protocol.json` was written before feature extraction
and ranking. It has SHA256
`d9e8cff9b235fcde8e7af28d70c16e87f36fde13b75bbef249f3bc480918e036`.
The protocol selects three deterministic test references for each of bowl,
plate, cup, ewer, censer and figure, requiring distinct official object groups
and an unambiguous catalogue class among those six classes. All 25 items in
the 18 query families are excluded from every ranking, leaving 4,667 eligible
objects from the original 4,692-object gallery. Global DINO CLS recall is
computed over that eligible gallery; only its top 100 candidates are reranked.
The 1,403 unique query/candidate images are cached by actual image bytes, pinned
model metadata, method and parameters.

The object-form relevance target is the existing catalogue alias matcher over
object type/name, classification and title. These are weak metadata labels,
not pixel judgments, exact silhouettes or evidence of visual relevance. Missing
or differently worded labels can appear negative under these metrics; the
top-ten missing-class fraction is therefore reported separately (5% averaged
across queries). This audit cannot estimate true human precision or nDCG. It
uses none of the separate eight-intent blind pool's grades and does not change
its protocol or results. No parameters were tuned after observing these scores.

| Method | Metadata matches@10 | Metadata nDCG@10 |
| --- | ---: | ---: |
| DINO CLS recall | 0.5722 | 0.5881 |
| Local patch cosine | 0.5667 | 0.5776 |
| Soft foreground profile | 0.5722 | 0.5895 |
| Fixed 0.75 local + 0.25 profile | 0.5778 | 0.5916 |

Only `cup_03` and the three censer references passed the foreground checks.
Mean usable query/candidate-pair coverage was 3.56%. The combined nDCG change
is only +0.0035, with no statistical or human-relevance conclusion warranted.
Pure local matching declined. Average metadata-positive recall in the top 100
was 0.1881; reranking cannot recover positives absent from its candidate set.
Whole-class recall also differs from instance or fine-grained shape relevance.

Coverage sets are partly complementary: 47 images passed both heuristics,
134 passed only the patch heuristic, and 205 passed only the classical one.
This observation is a diagnostic, not a validated union-mask or fusion policy.
Inconsistent border tokens caused 1,213 patch rejections. Frozen DINO tokens
carry global context and object information at the border; treating that border
as a uniform background is often inappropriate. No threshold was loosened to
make the coverage number look better.

The first uncached GPU run extracted all 1,403 images in approximately 32.4
seconds; this is feature extraction on this machine, not HTTP/search latency.
Later cached repetitions perform no image encoding, and the current manifest
reports their timing. `results.json`, `coverage_comparison.json` and
`run_manifest.json` retain measured scores, coverage and source/model/code
hashes. `prior_cached_results.json` and `repeatability.json` record identical
rankings and metric summaries across cached repetitions. A fresh extraction of
one reference from a path and the same PIL pixels gave zero maximum absolute
difference in tokens, weights, profile and foreground weights.

Fourteen focused unit tests check repeatability, shape versus texture behavior,
foreground abstention, unknown-slot preservation, explicit fallback, tie
ordering, malformed inputs, pinned model reuse and candidate alignment. The
audit is small and uses incomplete metadata supervision. It is a negative
coverage result with an inspectable local-matching prototype, not a successful
segmentation replacement or a validated improvement over the default system.
