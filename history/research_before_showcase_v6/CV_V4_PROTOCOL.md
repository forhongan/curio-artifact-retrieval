# CV v4: catalogue-first evaluation protocol

This is the pre-test protocol for the 2026-11-08 research branch. It does not
modify the frozen v2 production selection, the 2026-11-04 pixel audit, or any
prior score.

## Evidence layers

1. `official_catalog` is the primary label source for object identity, material,
   technique, era/culture and decoration when a museum record states it.  The
   title, object name, type, classification, medium/technique, period, tags and
   curatorial description are all valid evidence when training labels or
   scoring against catalogue claims.
2. `ai_visible_supplement` is optional additional supervision for clearly
   visible properties that the catalogue does not state.  The annotator sees
   the original image and complete relevant catalogue context.  Each claim
   records evidence, confidence and provenance.  `null` means unknown.  A
   supplement never overwrites an official fact.
3. `derived_ontology` maps official text into evaluation domains.  It is a
   deterministic parser; ambiguous values stay unlabelled.  Catalogue-derived
   labels evaluate catalogue compliance, while image-derived predictions
   evaluate visible recognition.  Results name the evidence source.

At retrieval time, image-only systems may consume query text and pixels.  They
must not use candidate titles, descriptions or catalogue labels to rank.  A
separate metadata-assisted system may use those fields and is reported as a
different experiment.  This anti-leakage boundary applies to ranking; it does
not demote museum records from their role as ground truth.

## Domains and metrics

Tasks are evaluated separately before reporting a combined result:

- object/type, including broad vessel/standalone/flat/part distinctions;
- broad material and fine material/technique;
- motif or iconography;
- broad dynasty/era;
- visible supplements such as palette, vessel subdivision, handles, mouth,
  foot, multiplicity and completeness;
- combined multi-constraint retrieval.

Single-object classification uses accuracy, balanced accuracy, macro F1,
per-class precision/recall, support and confusion matrices.  Retrieval uses the
same queries and exhaustive judgment set across methods.  In addition to P@5,
the boundary study returns an explicit yes set and reports set precision,
recall, F1, selected-set size, false positives and false negatives.  Query
level bootstrap intervals are retained.  A high catalogue-string score is not
described as visual understanding.

## Split and selection discipline

All fitting and threshold/weight selection use train and dev data only.  The
test split is evaluated after candidate methods and settings are frozen in a
selection manifest.  This catalogue was already used in earlier experiments,
so test results are a fixed supplementary comparison and are not described as
a newly blind holdout.  Every ranking system is evaluated within an explicitly
named gallery and gallery size.  Missing catalogue values are unknown, never
automatically negative.

## Background-isolation study

Object isolation is treated as a pixel-only preprocessing experiment.  It is
scored with: mask coverage against a small visually inspected mask audit;
foreground/background perturbation stability; sensitivity to compositing on
held-out backgrounds; abstention rate; and downstream retrieval change.  A
plausible screenshot is not mask accuracy evidence.  The method must handle
single objects and planar items as distinct image archetypes and abstain when
it cannot identify one coherent object.

## Prompt templates

CLIP-family templates are tested as a controlled query transformation, with
the same gallery, assets and reference queries.  Candidates include raw text,
`a photo of`, `a photograph of`, `a museum photograph of`, `an image of`,
`close-up photograph of` and punctuation/case controls.  Selection is on dev;
test is run after freezing.  The mechanism is cross-modal prompt alignment,
not proof that every prefix universally helps.

## Professional query to reference images

For abstract or specialist queries, a dynamic museum broad search may produce
eligible official reference images.  References are fetched from a public
museum database using the user query, not selected from a fixed answer key.
The system records query, sources, rights, hashes, exclusions and actual
reference images.  It compares single-reference, mean, max/coverage and
multimodal fusion after full-gallery visual recall.  Reference image matching is
reported separately from catalogue-only filtering.  Failure cases are viewed
and revised before the final evaluation.
