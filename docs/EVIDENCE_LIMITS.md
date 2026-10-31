# Retrieval evidence and its limits

The evidence summary is a diagnostic of available model signals. It provides no
calibrated probability of relevance, correctness, authenticity or faithful query
understanding. Its states are `unknown`, `limited` and `evidence_available`.
The last state means recognized constraints have model-class coverage; it is
not a high-confidence correctness band. A rule parser's default `confidence=1`
is not an empirical estimate of parsing accuracy and must not be presented as
one. Full natural-language understanding remains unverified even when every
recognized slot has a corresponding model class.

`src/curio/retrieve/evidence.py` accepts original cosine similarities in dense
rank order over the eligible candidates, before result truncation and fusion.
The top-two difference describes embedding-score separation only. Reciprocal
rank fusion and rank-normalized scores discard the original magnitudes; their
gaps cannot stand in for cosine gaps. In particular, a fixed gap induced by rank
positions is not evidence that one query is more certain than another. A single
candidate has no top-two gap. Neither a large cosine gap nor a normalized gap is
a probability of correctness; no arbitrary gap threshold establishes a reliable
confidence band.

Coverage is computed from the actual parsed positive and negative constraints.
Era and explicitly unsupported count/spatial constraints remain unknown for
visual verification and count in the denominator. An unrecognized query has
unknown coverage rather than perfect coverage. Coverage describes recognized
constraints only: the parser can still miss an unrecognized attribute. Missing
probe classes and unsupported component-presence constraints remain unknown.
The independent-lid class cannot verify whether another vessel has a lid, and
the absence of a predicted or catalogued label cannot prove visual absence.

Signal availability, ranking application and agreement are separate fields.
Available probes with a zero selected weight contribute no probe reranking.
Optional agreement diagnostics compare each query probe's maximum constraint
support with its support for the selected dense winner over the same candidate
set. For negations, both values use `1-P`. A tie counts as shared top support;
if diagnostics are unavailable, agreement is `not_measured`. The fraction of
probe constraints sharing a top candidate is a descriptive agreement fraction,
not a correctness estimate. Shared embeddings and correlated supervision mean
agreement is not independent verification. An all-tied probe can agree while
adding no discrimination; support values are retained for inspection.

The full query's chronology cannot be verified from image retrieval. An era
token can influence a text embedding through learned historical style
associations; it does not establish an object's date, period or authenticity.
Where catalogue dates are used, report catalogue attribution separately from
visual similarity. Unknown or incomplete dates remain unknown. Range overlap
means catalogue ranges overlap, not that the requested era is uniquely proven.

Weak-caption fine-tuning changes encoder weights through supervision. Using
catalogue-derived captions during training does not itself turn later image/text
embedding inference into runtime catalogue-text matching. Runtime BM25,
attribute matching and date filtering are separate retrieval operations and
must be disclosed when used. Conversely, image-only inference does not make
its training labels independent visual truth: metadata-derived supervision can
encode catalogue errors, shortcuts and historical attribution biases.

Level-3 retrieval is an experimental combination of visual recall and catalogue
constraints. Catalogue attributes need `satisfied`, `violated` and `unknown`
states: missing or incompletely represented fields are unknown, and a failed
alias match alone is not an explicit contradiction. Historical-token-stripped
recall is a separate control, with its actual query recorded. Report stage-1
diagnostics as stage-1 diagnostics; do not carry them forward as calibrated
evidence about a new reranked winner. Measure end-to-end latency including
stage 2. Fixed fusion weights are prototype configuration, not a validated
selection, unless selected on a declared frozen development protocol. The
research route needs visual-only, metadata-only, combined and shuffled-metadata
controls before claiming an improvement.

The existing image-query observations, captions and relevance judgments remain
AI annotations with their recorded provenance. They are not human gold,
independent human agreement or calibrated human relevance. The small restricted
audit galleries do not establish full-gallery relevance. Human calibration
would require separately collected judgments, development-only fitting and a
held-out evaluation of risk versus coverage and calibration; a UI percentage or
an ordinal label cannot replace those measurements. The local DSH review is
recorded research advice, not independent relevance labels or experimental
validation of the proposed level-3 combination.
