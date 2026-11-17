# Set boundary v7 — protocol fixed before new test evaluation

2026-11-12. Base commit: 59af1e6a58fe203807ef85b73feae687b53c8eaa.

Research only; no production method/default or frozen v5 results are changed.

- Reuse 4,692 production LoRA image embeddings and fixed SigLIP2 text tower.
- Freeze the existing 60 dev / 140 test object-group split and all qrels. Grade 3 is
  positive, grades 0/1/2 negative, -1 unknown and excluded from fitting and known-only
  metrics. Unknown selected counts and precision bounds remain visible.
- First diagnose raw-score separability with exact tied-score PR curves and per-query
  test oracle F1. Oracle thresholds are retrospective diagnostics, never deployed.
- Query conditions come from the existing query specification, NOT candidate catalog
  data. This is a structured-query experiment, not a validated free-text parser.
- Condition text and single-condition replacement contrasts are specified by a fixed
  bilingual dictionary. Unsupported historical/technique conditions are marked missing;
  no claim of historical verification from pixels. Contrasts are comparative features,
  not mutually exclusive ground truth (colors/motifs can coexist).
- Compare frozen dense / logistic / HSV baselines to three new controls: standardized
  raw+generic-margin+z-score logistic; condition features logistic; condition features
  logistic with grade-1 partial negatives weighted 2. C=1, max_iter=2000, no search.
- Features: three baseline values; condition cosine mean/min; replacement contrast
  mean/min; supported-condition fraction; number of conditions. No candidate metadata.
- Each new model uses 3-fold object-group cross-fitting on dev to select a shared macro
  F1 threshold from out-of-fold scores, then refits on all dev. This is empirical
  threshold selection, not independent probability calibration or risk certification.
- Separate leave-one-intent-out (LOIO) evaluation: for each held-out intent, fit and
  select threshold using only other intents on dev images; test on that intent's
  disjoint test images. EN/ZH are evaluated separately and hold out the same intent.
  The vocabulary and pretrained encoder are not held out; this is boundary-level
  intent transfer, not a new-concept benchmark.
- Report all fixed variants, per-query and macro F1, micro P/R/F1, unknown acceptance,
  empty sets, full-gallery counts, and visual/professional tier means. A paired bootstrap
  resamples 20 intent clusters (averaging languages within each), 2,000 times, seed 7.
  Intervals are exploratory and do not cover annotation uncertainty/multiple comparisons.
- Full-gallery selected IDs are retained. Recall is only for the 140-image judged core.
- No automated production promotion. A later experiment must be versioned separately;
  do not tune on the results of this protocol and reuse this test as fresh evidence.
