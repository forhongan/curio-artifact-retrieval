# CV v4 catalogue-first label audit

Status: implemented and generated on 2026-11-08.  The deterministic builder is
`scripts/build_cv_v4_catalogue_v2.py` (schema
`catalogue-first-builder-v2`).  It writes a separate versioned artifact and
does not alter `data/`, `eval/`, or
`artifacts/cv_v4_20261108/official/`.  The frozen evaluation and production
selection remain unchanged.

This work created no protected-path changes.  Unrelated research directories
already present under `eval/` remain untracked and were not modified here.

## Authority and schema

The builder preserves the complete official record under `official_catalog`
and derives explicit facets under `derived_ontology`.  Museum fields have
priority, and unresolved or comparison prose remains in
`unattributed_mentions` rather than becoming a positive label.

| Field | Meaning |
| --- | --- |
| `raw_class` | Source `type`/`classification`; it is not silently treated as an object form. |
| `object_form` | Direct form from the title head or an explicit catalogue class, with source and evidence. |
| `material_facets` | Explicit support material, Met medium, an explicit CMA type material class, or an explicit title material. Technique is not promoted. |
| `technique_facets` | Techniques parsed from the relevant technique or medium field. |
| `component_role` | Parent object or explicit stand/lid/case/stopper/ink-cake component, with parent form context. |
| `photographed_subject` | Explicit component role when the title establishes one; image unverified. |
| `color_facets` | Colour mentions with source field, sentence, literal match, scope and attribution status. |
| `motif_facets` | Motif mentions with source field, sentence, literal match and attribution status. |
| `group_context` | Group membership used for split isolation; it does not merge member labels. |

`object_type`, `material`, `colors`, `motifs`, and `era` are compatibility
views over attributed evidence.  `coating_colors` preserves colours attached to
a technique, medium or ware name; they remain in `color_facets` but do not by
themselves become a complete object palette.  Title colours and directly
attributed tags/prose remain eligible palette evidence.

## Final deterministic build

Command:

```powershell
.\.venv\Scripts\python.exe scripts\build_cv_v4_catalogue_v2.py --output artifacts\cv_v4_20261108\catalogue_v2
```

The official snapshot has 4,692 items: 3,474 train, 568 dev, and 650 test,
from 1,738 CMA and 2,954 Met records.  Its SHA-256 is
`4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68`.
The generated labels contain 4,692 records and have SHA-256
`9d7347a3e6135759e75c6cc37736de9e25fd4a60d0c9f3007a659d2bdaf14286`.

| Field | Known | Unknown |
| --- | ---: | ---: |
| object form | 3,902 | 790 |
| material facets | 4,155 | 537 |
| technique facets | 2,063 | -- |
| compatibility palette | 139 | 4,553 |
| motifs | 1,897 | 2,795 |
| era | 3,998 | 694 |
| coating/material colours | 1,589 | -- |

There are 119 explicit component records and 66 records that explicitly say a
whole object has a component.  The ledger contains 942 unattributed mentions
across 420 items; 322 items have at least one unknown-attribution reason.
There are 4,234 source object groups, including 164 multi-item groups and 622
group rows; no group crosses a split, and five multi-item groups contain
distinct raw classes.

## Source limitations

All 1,738 CMA records in the current snapshot have an empty or null
`support_materials` list; the cached Met records all have a nonempty `medium`.
Consequently, the builder cannot use a nonexistent CMA support list.  An empty
`support_materials` field remains meaningful: a material can still come from
an explicit CMA type material class or an explicit title, but words appearing
only in CMA technique text are not promoted to material.  An independently
checked populated support list takes priority over those fallbacks.

The compatibility palette is deliberately narrower than all catalogue colour
mentions.  Coating and ware-name colours remain inspectable in `color_facets`
and `coating_colors`; the palette view requires direct object/title/tag or
attributed surface prose.  The low palette coverage is therefore an honest
unknown boundary, not evidence that only 139 museum objects are coloured.

## Deterministic defect fixes against the legacy builder

These are transformation counts, not estimates of total catalogue truth.

| Count | Value |
| --- | ---: |
| Legacy material values removed | 3,413 |
| Legacy colour values removed | 2,107 |
| Legacy motif values removed | 257 |
| CMA items where technique terms are kept out of material | 592 |
| Explicit v2 component items | 119 |
| Items with unattributed catalogue mentions | 420 |
| Items with unknown attribution | 322 |
| Items whose old material is now unknown | 428 |

The 428 former material values include mixed-media CMA records whose support
field does not establish which title or technique term is the support.  The
builder preserves those terms as unknown review material instead of selecting
one silently.

## Item evidence

| Item | Official/title evidence | Derived result |
| --- | --- | --- |
| `cma_101719` | `Snuff Bottle (stopper)`, type `Glass` | Form `stopper`; parent `bottle`; material `glass`; photographed subject not image-verified. |
| `cma_123657` | `Snuff Bottle with Stopper (stopper)`, type `Ceramic` | Form `stopper`; parent `bottle`; raw class and material class remain separate. |
| `cma_130118` | `Goose and Goslings (Stopper)`, type `Jade` | Form and component `stopper`; raw class `Jade` supplies material `jade`; no invented parent form. |
| `cma_135011` | `Covered Jar (stand)`, technique `porcelain with blue underglaze decoration` | Form/component `stand`; parent `jar`; item type `Wood` gives material `wood`; porcelain/underglaze terms stay technique evidence. |
| `cma_140271` | `Stand for Screen with European Figures ...` | Form/component `stand`; parent form only `screen`; figures are not promoted to parent form. |
| `cma_121892` | `Box with Ink Cakes: Green Ink Cake in Shape of Coiled Dragon` | Form/component `ink cake`; parent `box`; motif `dragon`; green remains evidence and coating colour. |
| `cma_144684` | `Stem Cup ... (case)` | Form/component `case`; parent `cup`. |
| `met_41314` | `Porcelain with overglazed enamel colors ..., coral stopper` | Materials `porcelain`, `coral`; technique `enameled`; enamel is not material. |
| `cma_76525` | Discussion of red tea and other bowls | Direct white remains; red tea and other bowls' green/dark colours are unknown. |
| `cma_117940` | Other excavated drum stands typically show birds and tigers | Birds-on-tigers and unproven pigment mentions are unknown rather than this object's motifs/colours. |
| `cma_380090` | `dragon kilns` in kiln description | Motif `dragon` is rejected as a named non-surface phrase. |
| `cma_154392` | Mixed watercolour, metal, silk and embroidery description | Materials remain unknown when the title/technique does not establish the support; the case remains a reviewable mixed-media record. |

## AI supplement contract

The corrected static contract is generated in
`artifacts/cv_v4_20261108/catalogue_v2_ai_supplement/`.  It contains
`ai_supplement.schema.json`, `ai_supplement_prompt.md`, and a deterministic
240-item `ai_review_queue.jsonl` selected from 4,685 candidates by priority then
SHA-256 of the item ID.  The original `catalogue_v2/` labels remain at their
published SHA-256.  No model ran over the full corpus.

Official catalogue fields are immutable primary supervision.  A supplement is
an append-only visible observation for a training review queue, not a label
replacement.  The model receives the complete raw museum record and original
image, may read the full record, and may use tools to inspect crops, pixels, or
documented open-access resources.  It may return `claims: []` with a concrete
`no_supplement_reason` when the record and image are sufficient.

Observation fields are `visible_form_detail`, `visible_part`,
`visible_palette`, `visible_motifs`, `visible_surface`,
`visible_multiplicity`, `visible_orientation`, `photographed_subject`, and
`visible_material`, and `photographed_component`.  Every nonempty claim requires
a raw-record field and exact quote in `catalogue_evidence`, an explicit
`image_evidence.region` and `observation`, confidence, `observation_status`,
`training_role: candidate_append_only`, and `visibility: clear`.  The status is
`catalogue_confirmed_visible` only when the quote states the same property and
`image_only_obvious` when the record establishes identity but omits that
obvious property.  Partial or ambiguous observations are excluded.  The validator binds each response to the queued
image path/hash and full-record path/hash, rejects extra fields, duplicate item
IDs, invented or nonliteral catalogue evidence, and non-clear visibility.  A
visible fact may repeat an official fact because the observation has independent
pixel evidence; official structured and description facts remain authoritative.

The zero-claim example validates without changing the official label:

```powershell
.\.venv\Scripts\python.exe scripts\build_cv_v4_catalogue_v2.py --validate-supplements artifacts\cv_v4_20261108\catalogue_v2_ai_supplement\ai_supplement_no_claims_example.jsonl --supplement-phase model
```

The final concrete batch is in
`artifacts/ai_supplement_terra_20261108_batch04_final/`.  It used the installed
`codex-cli 0.166.0-alpha.5.1` with `gpt-6.1-terra`, high reasoning, a read-only
sandbox, image viewer enabled, the complete relevant raw record supplied, and
the original image attached.  All three responses passed validation: 3 items,
20 claims, 20 clear, 14 image-only obvious observations, 6 catalogue-confirmed
observations, and 0 no-claim items.  The separate
`training_queue.jsonl` records the immutable official catalogue block plus the
append-only observations.  Its raw activity logs show the model inspecting
pixel crops with workspace tools.  This is a small engineering batch, not an
accuracy estimate for Terra or the full queue.

`review_examples.csv` presents all 20 accepted claims side by side with their
catalogue quote, image region, visible observation, confidence, and visibility
for reviewer scanning.  Earlier `batch02_clear/` and `batch03_reconciled/` runs
remain audit history and do not supersede this final accepted batch.

The runner writes every response to a new run directory and refuses to reuse an
existing output directory.  It records the queue, prompt, schema, Codex CLI, and
model in `run_manifest.json`; each task records its argv, image and record hash
checks, prompt, raw record, and CLI event log.  The assembled queue retains the
official catalogue in its own field and requires `training_role`
`candidate_append_only` for each observation, so the supplemental values cannot
be mistaken for frozen labels.

Suggested rerun command for the same three queue items:

```powershell
.\.venv\Scripts\python.exe scripts\run_ai_supplement_terra.py --limit 3 --workers 3 --view --output artifacts\ai_supplement_terra_20261108_batch_next
```

The per-item `command.json` files record the full argv.  The principal invocation
was `codex.exe exec --ephemeral --skip-git-repo-check --model gpt-6.1-terra`
with `model_reasoning_effort="high"`, `features.view_image=true`, a read-only
sandbox, the generated output schema, the complete raw record in the prompt,
and the original image attached through `--image`.

The earlier `batch01_retry2/` run remains as audit history.  It produced 21
claims, including 3 partial observations, before the contract was tightened to
clear-only.  It is not the final training queue.

## Output manifest

All paths are under `artifacts/cv_v4_20261108/catalogue_v2/`.

| File | SHA-256 |
| --- | --- |
| `labels_catalogue_v2.jsonl` | `9d7347a3e6135759e75c6cc37736de9e25fd4a60d0c9f3007a659d2bdaf14286` |
| `summary.json` | `9453acaba433e73c435a4be86d2b689e49f6750bd473022a1d083769275d27b7` |
| `defect_fixes.json` | `c7210df125985da72224dfab335d1e56c7b5aaafc039f9c2398d438d918abe67` |
| `group_report.json` | `e3c3e61dfe3085e0ba4bacd030e634b4fbe63d940e44453a79df4ff2887df486` |
| `catalogue_issue_ledger.jsonl` | `66fb26906c3c1eed1b64a89223bb8ec4d22072895c55331b51e1423d322d1bc5` |
| `ontology_candidates.json` | `a0cb84899f45b39799cfba3fdcf2b9f0c0a286adcda6b0c806daf837354d8755` |
| `ontology_candidates.csv` | `079e2387bccaa0f76b332e5c5780da483c1c43b306ab345bb554cc417465bcf9` |
| `README.md` | `4b054d5cc45571c0a383e16519db631503ab013a4cca1611eb391155992c5762` |

Files under `artifacts/cv_v4_20261108/catalogue_v2_ai_supplement/` use the same
deterministic label rows with the corrected eligible-field metadata:

| File | SHA-256 |
| --- | --- |
| `labels_catalogue_v2.jsonl` | `1f17238b2898565cc442e6cf60a1fa74ccc3cd83c75a2c9e0e1b0086dff8a6c4` |
| `ai_review_queue.jsonl` | `80f018b14833c91e50d2e8e9f39c8b26539ff1261cecb335d669eceb05cbcd04` |
| `ai_supplement.schema.json` | `5ebda3340d3a976798881fd1f354de9bf70ffc7963f2d19137ccdf57bb5dbcf2` |
| `ai_supplement_prompt.md` | `2eb34b0b7ee23fd85301625c064d40924aa6fc70e159df2df1dd6f25e6e7d438` |
| `ai_supplement_no_claims_example.jsonl` | `6b1787b43729277f201a3ff307f10863dac1c00515021a1cd7058b1eb4597278` |

The concrete batch artifacts under
`artifacts/ai_supplement_terra_20261108_batch04_final/` have these hashes:

| File | SHA-256 |
| --- | --- |
| `responses.jsonl` | `4ea95e8efc6d153fdefe57c7a7df66aaed03dabfd57b13c5083b3feb0b976c47` |
| `validation.json` | `44d56c52c69d9a51215d0d9c9d26e5f1530c51b55b5a0cfe1cf4d5da524ffa35` |
| `training_queue.jsonl` | `48e4d180d60b9a4651a35e99cdb4ac8b69fd277ebf693f70607fdba08485f9e4` |
| `review_examples.csv` | `3a4114dd80093f6b27ca3e2f6c5d61c22589e23532df5388408425640bdd8dd9` |
| `summary.json` | `081e07e4fe30e28f97f52a57a133408ed803dd62a7d75eda8c1cf3ec388ee62c` |

Validation run after the final edits: `pytest tests/test_catalogue_v4_v2.py -q`
reported 15 passed.  Ruff reported all checks passed.  The original frozen
catalogue label file is restored to its published SHA-256
`9d7347a3e6135759e75c6cc37736de9e25fd4a60d0c9f3007a659d2bdaf14286`.
