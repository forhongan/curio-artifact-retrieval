# Dynamic Reference Strict-Audit

Date: 2026-11-08

This audit covers the eight professional museum-reference queries in
`scripts/run_dynamic_reference_v4.py`. Museum metadata is used only to acquire and
select reference photographs. Gallery ranking still uses image pixels only.

The table below uses the uncached first live-run acquisition counts recorded in
the existing `run_full` diagnostics. A later offline rerun reused that directory
and updated subsets of its accepted/rejected arrays, so those files are no longer a
reliable reconstruction of the first live acquisition. The failure examples and
contact sheet evidence were captured before the overwrite.

## Old failure

The previous implementation treated era mentions anywhere in
`title/culture/date/period/reign` as object-era evidence. A curatorial description
could therefore make an Edo-period Japanese jar pass a Ming request because its
description mentioned Ming export ceramics. The metadata normalization currently
drops Cleveland `description`, but the validator was unsafe for any record that
retained prose and also searched titles.

Selection had two additional defects:

- It counted requested forms as the entire strict test. Fewer than two form matches
  caused an automatic fallback to every accepted candidate, regardless of era or
  culture. Observed examples included French, Japanese, Korean, Parthian, and
  Seleucid objects in era-specific sets.
- `龙` was a substring of `龙泉`, so the Southern Song Longquan query invented a
  dragon motif constraint.

The strongest examples in the pre-change `references_*.json` were
`cma_140301` and `cma_447085` (Edo Japanese) selected for Ming, Parthian
`met_324038` and Seleucid
`met_326384` selected for Liao, and French/Japanese/Korean vases selected for
Southern Song. Its old contact sheet for Ming also showed the Japanese jar alongside
Chinese blue-and-white pieces.

## Change

`src/curio/retrieve/dynamic_references.py` now separates evidence roles:

- Era is read only from `culture`, `date`, `objectDate`, `creation_date`, `period`,
  and `reign`. Titles and descriptive prose cannot establish era or culture.
- Chinese-era requests require a structured Chinese culture. Structured dynasty or
  reign dates must intersect the requested range when a date is available.
- Every requested concept is required for a strict match. Form evidence must occur
  in object identity fields. Explicit ceramic/porcelain requests require material
  evidence. Reign and broader-dynasty constraints must agree.
- `龙泉` no longer matches the `龙` motif alias, while the explicitly written
  `青瓷` remains a porcelain/material requirement.
- Selection contains only strict rows. Broad discovery search may still run, but
  unverified rows cannot become selected references or receive a verified label.

`scripts/run_dynamic_reference_v4.py` now writes `selection_status` per query and
`acquisition_summary.json`, reports skipped queries, and evaluates only queries with
at least two verified references. `--cache-dir` allows an offline rerun to read a
completed live cache while writing new diagnostics elsewhere. The runner continues
to record request URLs, cache status, object metadata, original source response
hashes, and image hashes.

## New counts

Post-fix cached/offline acquisition output:
`artifacts/dynamic_reference_v4/strict_visual_v1/`

| Query | Old accepted/selected | New accepted | Strict accepted | New selected | Status |
|---|---:|---:|---:|---:|---|
| Ming blue and white | 21 / 5 | 14 | 0 | 0 | insufficient |
| Wanli blue and white | 16 / 5 | 12 | 0 | 0 | insufficient |
| Kangxi meiping | 4 / 4 | 4 | 1 | 1 | insufficient |
| Qianlong famille rose | 12 / 5 | 12 | 3 | 3 | strict |
| Liao leather flask | 3 / 3 | 3 | 0 | 0 | insufficient |
| Yuan blue-and-white dragon vase | 22 / 5 | 23 | 0 | 0 | insufficient |
| Southern Song Longquan celadon vase | 6 / 5 | 6 | 0 | 0 | insufficient |
| Ming Jiajing blue-and-white jar | 22 / 5 | 22 | 0 | 0 | insufficient |
| **Total** | **106 / 37** | **96** | **4** | **4** | **1 qualified query** |

The selected Qianlong references are `cma_123596`, `cma_123598`, and
`cma_140220`, each Chinese, Qing, and Qianlong in structured attribution. The other
seven queries now return no references rather than incompatible cultures. The
availability bottleneck is the Met search/page cap and sparse era-specific results,
not a reason to relabel broad images as strict.

## Validation

- `.venv\Scripts\python.exe -m pytest tests/test_dynamic_references.py -q`:
  9 passed.
- `.venv\Scripts\python.exe -m pytest -q`: 392 passed, 7 warnings.
- `.venv\Scripts\python.exe -m ruff check
  src/curio/retrieve/dynamic_references.py scripts/run_dynamic_reference_v4.py
  tests/test_dynamic_references.py`: all checks passed.
- Repository-wide Ruff still reports four pre-existing unused-name findings in
  `scripts/audit_cv_v4_catalogue.py`; that file was not changed.
- Cached offline rerun command:
  `.venv\Scripts\python.exe scripts/run_dynamic_reference_v4.py --offline --output
  artifacts/dynamic_reference_v4/strict_visual_v1 --cache-dir
  artifacts/dynamic_reference_v4/run_full`.
- The valid post-fix contact sheet is
  `artifacts/dynamic_reference_v4/strict_visual_v1/contact_sheets/qianlong_famille_rose.jpg`.
  It was visually inspected; its first three reference images are Chinese
  Qianlong famille-rose ceramics.

## Remaining limits

- Only one of eight queries currently has two or more strict references in the
  captured cache. The visual metrics therefore describe one qualifying query and
  must not be reported as an eight-query aggregate.
- Structured culture is required for Chinese-era queries. Records with an empty
  culture field are rejected even if their numeric date falls in the requested
  range; this favors false negatives over false era claims.
- Museum free-text search and shallow result caps still miss known objects. Broad
  fallback search increases recall, but strict matching remains metadata-bound.
- The Qianlong target pool is a diagnostic metadata intersection, not human qrels.
  Visual grades remain AI contact-sheet review rather than inter-annotator gold.
