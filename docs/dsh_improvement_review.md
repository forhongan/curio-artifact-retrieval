# DSH improvement review

An independent review was requested through the local DeepSeek Harness on
2026-10-30. The raw session snapshot is kept in
`docs/dsh_session_snapshot.json`.

The review agrees with keeping the current visual route pure and raises a
measurement issue before adding heavy models: the current visual evaluation is
24 intents over a 60-image gallery, and the intents are AI-generated. It should
not be used to claim small gains from a backbone swap or a crop policy. The
review recommends a human-authored, leakage-audited query set over the full
gallery, with an explicit distinction between category-level and instance-level
relevance, paired bootstrap intervals, and simple category controls.

For category coverage, it recommends mining the catalogue vocabulary, anchoring
aliases to stable external identifiers where possible, testing alias collisions,
and reporting coverage by concept and language. This supports the current
data-driven ontology expansion, but does not make catalogue text a visual label.

For level-3 retrieval, the recommended reproducible design is:

1. Freeze a structured query with visual constraints (form, palette, motif) and
   catalogue constraints (dynasty, period, culture, medium), with provenance for
   every constraint.
2. Stage 1 performs image-only recall over the full gallery and reports both
   item-level and type-level recall.
3. Stage 2 applies catalogue filters for historical constraints and soft visual
   reranking for visible constraints. Every result reports `satisfied`,
   `violated`, or `unknown`, tagged as visual, catalogue, or unverified.
4. Compare visual-only, metadata-only, combined, oracle metadata, and shuffled
   metadata controls. Also run a dynasty-token-stripped query to detect text
   leakage.

The review supports ordinal evidence bands and abstention when human labels are
not available. It recommends showing constraint coverage, top-score separation,
embedding/probe agreement, and unsupported-claim rate instead of an
uncalibrated percentage. If labels become available, calibrate on dev only and
report risk-coverage, AURC, ECE, Brier score, and per-language results.

The main warnings are to avoid training a dynasty classifier and calling it
visual verification, tuning repeatedly on the AI-generated gallery, uncontrolled
synonym scraping, using external auction images as trusted references, and
selecting multi-crop/backbone changes before the evaluation protocol is strong
enough to resolve their effect.

The proposed two-week order is: freeze the evaluation protocol, audit split and
near-duplicates, collect human query judgments, measure vocabulary coverage,
expand only the largest gaps, implement the level-3 two-stage prototype with
controls, calibrate confidence, and run at most one targeted modelling
experiment. This matches the local plan: multi-crop is retained as a
research-only ablation because its current English gain is negligible and its
Chinese result declines.
