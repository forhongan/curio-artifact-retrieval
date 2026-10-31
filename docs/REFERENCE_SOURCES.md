# Official museum references and separate category expansion

The image search service uses actual external museum images as visual references.
It never treats rule-generated words or the local gallery's descriptive metadata
as reference images. The frozen benchmark remains `data/snapshots/collection_v2.jsonl`
(4,692 images). References and expansion images have separate roles and manifests.

## Published artifacts (30 October 2026)

| Artifact | Count | SHA256 |
| --- | ---: | --- |
| Frozen gallery | 4,692 | `4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68` |
| `data/references/bank.jsonl` | 43 references | `d515773d497a7d65af25d26cff544febba77ed96e647e917d72640b4f6fa21f5` |
| `data/snapshots/expansion_v3.jsonl` | 101 images | `2d7f4a7c3cc446381e75f9a39155aa20cd514c60eaa8ee3c617d997bfe5a68c7` |

All reference images have local files in `data/references/images/`; each accepted
row records the official object and image URLs, source ID, CC0 rights, collection
query terms, canonical metadata SHA256, original image SHA256, dimensions, 256-bit
average hash and 64-bit perceptual hash. Official object-response snapshots are
in `data/references/metadata/`; URL-keyed HTTP cache entries also record the request
URL and retrieval timestamp. Raw ingestion files are read only.
`data/references/acquisition_manifest.json` binds all 43 entries to the bank SHA
and actual official-response cache timestamps. It distinguishes 29 discoveries
from read-only raw metadata from 14 newly searched discoveries. Query vocabulary
on a raw-discovered entry does not claim that each vocabulary query was executed.

The bank's overlapping category counts are blue-and-white 6, celadon 6, meiping 5,
gourd vessels 5, bronze 5, jade 6, lacquer 4, textiles 5, porcelain 18, books 6.
These counts are not independent objects or a guarantee that any combination of
form, glaze, reign and motif is covered. The bronze group includes censers, a belt
buckle and one ritual wine warmer; it does not establish a reference for every
ritual vessel form. The textile group contains sample books rather than robe
coverage. Fragment/shard entries remain auditable in the bank but are excluded
from selection unless the user explicitly requests fragments. Lacquer includes
separately photographed box parts, so category image counts exceed family counts.

`data/references/coverage.json` reports acquisition queries, category counts,
whole-object counts, gaps and failures. `exclusions.jsonl` records rejected
gallery identities, families and duplicate images. `failures.jsonl` preserves
network, rights and quality failures rather than inventing successful references.

## Primary sources and rights

* [Cleveland open-access API documentation](https://openaccess-api.clevelandart.org/):
  `/api/artworks/` supports query terms, Chinese culture, `cc0=1`, `has_image=1`,
  pagination, object records and image URLs. Acceptance requires the individual
  object's `share_license_status == "CC0"` and an official
  `openaccess-cdn.clevelandart.org` HTTPS image. Official related works and
  inseparable parts are relationship evidence, not title-based identity guesses.
* [Met collection API documentation](https://metmuseum.github.io/): searches use
  `/public/collection/v1.1/search` with Chinese geography and image filters.
  Individual object records remain under `/public/collection/v1/objects/{id}`.
  The former v1 search endpoint was retired on 1 October 2026. Acceptance requires
  `isPublicDomain is True` and an official `images.metmuseum.org` HTTPS image.
  The API documentation explicitly describes eligible collection images as CC0.

Discovery can read existing cached raw museum metadata, but each accepted object
must also have an official API response cached in the acquisition's own directory.
Downloaded images must have a short edge of at least 256 pixels and a long edge
of at least 512 pixels and must successfully decode. There is no third-party web
image scraping, query-string image fabrication or rights inference from age.

## Query planning and selection

`plan_reference_query()` explicitly identifies its planner as
`bilingual_rules_v1`, with `is_llm=false`. Literal Chinese/English rules map terms
such as 青花, 龙泉, 梅瓶 and 葫芦 to professional English museum queries. Unknown
Chinese terms are reported as unmapped. This component does not call or claim to
be an LLM or a semantic translation system.

`select_references()` requires every recognized category, the requested form
(vase/bottle, bowl, dish, jar, box, tripod, ritual vessel or robe), and recognized
reign/dynasty constraints. Form evidence comes from the object's title/name/type.
Era evidence comes from explicit date/culture/period/reign fields; descriptions
that compare an object with a different period cannot satisfy it. Source accession
series and repeated titles diversify otherwise equally matching references;
selection groups do not assert that all series members are one physical object.
An unsupported combination returns no references and
`constraint_status="unverified_no_reference"`. In particular, separate coverage
of blue-and-white ceramics and meiping does not justify claiming a blue-and-white
meiping reference. No fallback silently drops a recognized form or reign.

## Isolation checks

`python scripts/audit_reference_isolation.py` writes
`data/references/isolation_audit.json`. The final audit passes with zero issues:

* All 43 references are checked against the entire frozen gallery and all 101
  expansion images, independently of split.
* All expansion images are checked against the frozen gallery and references.
* Exclusion uses source IDs, recorded object groups, official accession lettered
  parts and parent records, Cleveland related works/inseparable parts, exact
  raw/processed/thumbnail SHA256, average-hash Hamming distance at most 4/256,
  and perceptual-hash Hamming distance at most 6/64.
* Official rights, canonical metadata SHA256, actual image bytes and expansion
  thumbnails are verified. The frozen gallery and bank hashes remain unchanged.

The 256-bit value called `phash` in the older frozen gallery is actually an average
hash; the reference code names it `ahash256` and separately computes a real pHash.
These conservative checks reduce leakage; undocumented object relationships and
radically different views can still require human review. They are not proof of
complete physical-object identity resolution.

## Separate searchable expansion shard

`expansion_v3.jsonl` contains 101 real images from Cleveland (52) and the Met (49),
with local 1,024-pixel images and thumbnails under `data/processed/`. Its explicit
object types are book 75, calligraphy 17, painting 6 and textile 3. Many book and
album pages also carry painting tags: the overlapping tag counts must not be
presented as independent artifact totals. Manual titles mentioning calligraphy
do not by themselves label the visible page as calligraphy.

The shard has 66 conservative source-accession/official-relation families, all
`split="unassigned"`; it is available for an opt-in searchable gallery and is not
a new training/evaluation split. Source classifications remain source data rather
than a generated department label. `data/expansion_v3/report.json` records actual
object-type counts, family policy and checksums. Five textile pages linked to a
reference album were removed and recorded in `final_exclusions.json`; four new
explicit calligraphy images replaced them. Acquisition snapshots, images,
exclusions and failures are preserved separately under
`data/expansion_v3/acquisition/`. The extension never joins the reference bank.

## Precise-period supplemental references

`data/references/supplemental.jsonl` adds two independently acquired CC0 whole-vase
photographs without changing the 43-reference evaluation bank. Its SHA256 is
`8291cda9edfd953d72fda0e385f5c33c6544449759f07420cba93e3b9999cfe0`.
The loader concatenates this optional manifest after the original bank, preserving
the bank's entries if an ID is repeated. The isolation audit reports the two counts
and checksums separately and checks supplements against both galleries and bank43.

* Met `44749`, *Gourd-shaped bottle with lotus scrolls*: the explicit source period
  is Ming dynasty, Jiajing mark and period (1522–66). The actual whole-object image
  is blue-and-white. It supports a Ming blue-and-white gourd bottle reference;
  it must not satisfy a Wanli reign request.
* Cleveland `111782`, *Vase*: the explicit source culture identifies the Wanli
  reign (1572–1620), and the date is 1573–1620. Although the official medium says
  “porcelain with underglaze blue decoration,” visual inspection reveals green
  figures in addition to blue decoration. It is categorized as
  `underglaze_polychrome`, excluded from strict `blue_white` selection, and carries
  this qualification in its row. The source does not explicitly establish a wucai
  technique attribution. Literal queries such as 万历青花五彩瓶 can retrieve it as
  a qualified polychrome visual reference; the API response includes the limitation.

`scripts/build_supplemental_references.py` preserves detailed official searches,
local images, metadata, failures and exclusions under
`data/references/supplemental_acquisition/`; `supplemental_report.json` explicitly
reports the unmet target of three strict Ming blue-and-white vases. Only one strict
supplement met all constraints. Other precise Ming and Qianlong famille-rose
candidates were gallery IDs, or searches returned HTTP 403. There is no accepted
Qianlong famille-rose supplement. The report does not relax identity, rights, era
or visual-material constraints to reach a target count.

## Rebuilding

```powershell
.venv/Scripts/python.exe scripts/build_reference_bank.py --per-category 5
.venv/Scripts/python.exe scripts/acquire_expansion.py --target 100
.venv/Scripts/python.exe scripts/audit_reference_isolation.py
.venv/Scripts/python.exe -m pytest tests/test_references.py -q
```

Rebuilding acquisition changes the generated bank; retain the published bank SHA
when reproducing an evaluation. `--offline` on the reference builder only uses its
own HTTP/image cache and records cache misses. `--output PATH` keeps exploratory
acquisition in a separate directory. The builder checks an existing expansion
catalog as an additional exclusion set. The expansion acquisition preserves the
existing reference-bank SHA, uses separate caches, and checks both gallery and
reference families before acceptance.
