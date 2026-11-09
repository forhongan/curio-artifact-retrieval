# COMP5523 v4 label survey

## Scope and provenance

This survey reads every raw visual annotation source under `eval/` and joins blind aliases to `data/snapshots/collection_v2.jsonl`. It covers the five train files, dev `labels_sol`, test `labels_root`, and the three CV-research repeatability files. It separately inventories visual query judgments and metadata qrels so they are not mistaken for image attribute labels or independent human gold.

The reproducible entry point is `scripts/v4_label_survey.py`. It writes the tables and joined data to `artifacts/cv_v4_20261108/label_survey/`. The script only reads `eval/`; it does not rewrite frozen annotations, qrels, or scores. The official snapshot SHA-256 is `4dbb5e4045bbab4fd245f7fce2d2fc363aabfe41a45d31b91703ddaac5ebee68`; the canonical visual annotation SHA-256 remains `cb2d78cf16f08e9aea66ceff89da981dfc6e5a6ff0b556a6e7d77ae8b13a73f3`.

All visual labels are AI-derived. `object_type`, colors, shape flags, motif flags, captions, and uncertainties were generated from images by Terra A/B, Root AI, and Sol sessions; CV-research A/B/QA are additional AI repeatability judgments. The metadata qrels are deterministic ontology-derived silver labels. None is human gold.

## Source inventory

| Source | Rows | Actual official split counts | Annotator | Captions | Official object type | Official material | Official motif list | Official colors |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `visual_train/labels_terra_a.jsonl` | 72 | train 67, test 4, dev 1 | terra_a | 72 | 51 | 65 | 10 | 0 |
| `visual_train/labels_terra_b.jsonl` | 96 | train 84, dev 6, test 6 | terra_b | 96 | 68 | 90 | 21 | 0 |
| `visual_train/labels_root_train.jsonl` | 24 | train 21, dev 2, test 1 | root_ai | 24 | 15 | 23 | 6 | 0 |
| `visual_train/labels_root_supplement.jsonl` | 24 | train 24 | root_ai | 24 | 19 | 24 | 6 | 0 |
| `visual_train/labels_root_final.jsonl` | 24 | train 22, test 2 | root_ai | 24 | 14 | 23 | 4 | 0 |
| `visual_dev/labels_sol.jsonl` | 60 | dev 60 | sol | 60 | 32 | 52 | 13 | 0 |
| `visual_test/labels_root.jsonl` | 60 | test 60 | root_ai | 60 | 36 | 55 | 17 | 0 |
| `cv_research_20261104/annotator_a/labels.jsonl` | 72 | dev 24, test 48 | gpt-6.1-terra-cli-a | 0 | 52 | 65 | 13 | 0 |
| `cv_research_20261104/annotator_b/labels.jsonl` | 72 | dev 24, test 48 | gpt-6.1-terra-cli-b | 0 | 42 | 67 | 16 | 0 |
| `cv_research_20261104/annotator_qa/labels.jsonl` | 24 | dev 8, test 16 | gpt-6.1-terra-cli-qa | 0 | 16 | 21 | 4 | 0 |

The old visual-label pool contributes 360 rows across 240 train, 60 dev, and 60 test blind items. The CV-research repeatability source contributes another 168 rows. Total survey input is **528 raw rows covering 503 unique official items**: 478 items occur once and 25 occur twice because the CV-research sample overlaps older dev/test items and one older train item. Captions exist for all 360 old-pool rows and none of the 168 structured-only CV-research rows.

The actual train rows are not all training eligible after v2 grouping: 218 map to official train, 9 to dev, and 13 to test. The distinction is retained per row and per source; the “train” directory name should not be used as the current split.

The 22 JSONL files under `eval/` all load. Derived `visual_labels/annotations.jsonl` has 338 effective rows; it is not an independent source and is excluded from raw totals. Visual query judgments contain 48 rows across dev/test, explicitly grading 60 images per query (2,880 query-image decisions total: grade 0 = 2,557, 1 = 208, 2 = 32, 3 = 83). Metadata qrels contain 120 rows and 63,073 graded item occurrences, with grades only 1/2/3 (47,809 / 10,472 / 4,792): absence is encoded by omission. All graded IDs resolve to the official snapshot.

## Official catalogue support

| Pool | Items | `object_type` | `object_name` | `classification` | `material` | `medium` | nonempty `motifs` | nonempty `colors` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train blind pool | 240 | 167 | 240 | 118 | 225 | 240 | 47 | 0 |
| Dev blind pool | 60 | 32 | 60 | 27 | 52 | 60 | 13 | 0 |
| Test blind pool | 60 | 36 | 60 | 30 | 55 | 60 | 17 | 0 |
| All surveyed unique items | 503 | 328 | 503 | 243 | 463 | 503 | 106 | 0 |
| Full catalogue | 4,692 | 3,131 | 4,692 | 2,869 | 4,382 | 4,692 | 1,091 | 0 |

Exact support for the currently registered vocabulary is in `summary.json` under `registered_vocabulary`. The strongest official slots are vase (1,111), porcelain (2,096), ceramic (1,113), and floral motif (663). Of 28 registered object types, 17 have exact official support; of 18 materials, 11; of 15 motifs, only the seven fields populated by ingestion; and none of the 11 colors or nine shape concepts has a populated official structured field.

This does not mean colors and shapes lack catalogue text. Catalogue `medium`, title, tags, description, and object name contain color, glaze, form, and method language. It means the snapshot has no official structured values for `colors` or `shape`. Catalogue text matches are text support, not verified visual observations.

## Raw schema

The old label files contain `object_type`, eight color flags, five shape flags, ten motif flags, two captions, confidence, uncertainty, and annotator. The CV-research files contain only `object_type`, eight colors, five shapes, confidence, and uncertainty. Structured material, part/completeness, multiplicity, orientation, surface/technique, and vessel subclass fields are absent everywhere.

Raw `object_type` is effectively closed to 20 schema choices, but one Terra B row returned an out-of-schema `bottle` value. It is retained as observed data rather than normalized into another class. `other` has 50 unique-item instances and represents unresolved visible form, not official ground truth; 175 of the 503 surveyed items have an empty official `object_type`.

Field-level counts are in `field_counts.json`. Across the unique-item rollup, shape positives/known/null are: narrow mouth 140/401/127, wide mouth 138/407/121, handles 46/466/62, footed 71/471/57, and round body 164/470/58. Color positives/known/null range from black 114/432/96 to pink 38/432/96. Motif flags are much more incomplete: for example floral is 108/253/275 and geometric 141/286/242. Unknown is not a negative.

## Frequent unregistered concepts

Rates use 360 dated AI caption rows, not 528 source rows, because CV-research has structured labels only. “Catalogue rows” counts a normalized term in title, object name, classification, medium, description, or tags. One caption or catalogue item can match multiple terms in a category.

| Category | AI caption rows | Unique AI items | Catalogue rows | Highest-frequency evidence |
|---|---:|---:|---:|---|
| Shape/form and parts | 362 / 317 | 228 / 219 | 645 / 886 | neck 69, body 55, rectangular 37, rim 28, base 24, foot 20, mouth 17, lobed 14, cylindrical 11 |
| Colors/appearance | 285 | 222 | 209 | pale 101, dark 78, blue and white 24, patterned 22, golden 16, multicolored 12, mottled 10 |
| Materials/appearance | 52 | 49 | 4,549 | textile 23, glossy 13, metal 4, gold 3, stone 3, translucent 3 |
| Multiplicity | 81 | 73 | 521 | two 26, repeated 12, repeating 9, four 7, several 7, three 6, one 5 |
| Orientation | 57 | 55 | 119 | standing 22, seated 14, vertical 9, horizontal 6, upright 4 |
| Surface/technique | 182 | 136 | 2,857 | raised 46, painted 32, relief 32, carved 25, glossy 13, incised 8, pierced 6, woven 6 |
| Motifs | 132 | 85 | 1,035 | animal 28, scene 20, creature 13, leaves 13, medallion 9, scroll 7, tree 6, building 6 |
| Vessel subclass | 79 | 66 | 1,109 | bottle 22, covered 17, lidded 11, dish 6, ewer 5, stopper 3, double-gourd 4 |

“Pale” and “dark” are appearance modifiers rather than sufficient color truth, so they should not be silently mapped to white/black. “Blue and white” is a compound visual convention and is also present in the specialist glossary, but it is not an official `colors` field. “Painted,” “carved,” and “incised” can describe either manufacture or visible treatment; catalogue medium strengthens provenance but cannot prove what a particular photo shows.

The strongest review candidates from both sources are `bottle` (AI 22, catalogue 316), `covered` (17, 206), `relief` (32, 143), `painted` (32, 1,106), `carved` (25, 123), `standing` (22, 44), `seated` (14, 54), `two` (26, 99), `medallion` (9, 44), and `blue and white` (24, 8). `lidded` is frequent in AI captions (11) and visual judgments but absent literally from catalogue text. `meiping` (catalogue 78), `buckle` (59), `stopper` (61), and `flask` (18) are catalogue-supported vessel/form specializations with little or no matching caption occurrence; their absence in AI captions is not evidence that the objects lack those forms.

The caption query set already tests many of these concepts. `queried_by_visual_judgments` in `concept_candidates.csv` records whether a term occurs in the 48 dev/test visual queries. This shows that the problem is not discovery alone: several frequent concepts are already queried while missing from the closed attribute schema.

## Evidence examples and known boundaries

| Concept | Source and item | Evidence | Provenance |
|---|---|---|---|
| bottle / lidded | Terra B `T157` → `cma_101765` | “A round clear vessel with a green stopper and colorful painted panels.” | AI caption; official snapshot object type is cup |
| double-gourd | Terra B `T146` → `met_46815` | “A glossy black double-gourd shaped vase.” | AI caption; catalogue title is Gourd-shaped bottle; official type is vase |
| part/completeness | Terra B `T151` → `met_450127` | “A brown fragment of a vessel with a flared opening and raised decoration.” | AI caption; catalogue title is Fragment of an Imported Chinese Ewer |
| surface | Terra B `T138` → `cma_154680` | “A pale round plate with faint incised decoration.” | AI caption; catalogue medium contains incised language |
| multiplicity | Terra A `T004` → `cma_96995` | “A low multicolored round box with a decorated domed lid.” | AI caption; no structured multiplicity/part field |
| orientation | Terra A `T008` → `met_59776` | “A grayscale seated human figure on a low base with an upright headdress.” | AI caption; no structured pose field |
| motif | Root final `T218` → `cma_101215` | “A narrow brown and gold textile with repeating rounded medallions and dense geometric bands.” | AI caption; no official medallion field, geometric is AI-only flag |
| color/appearance | Root final `T222` → `cma_124055` | “A tall mottled green and brown bottle with a narrow upper section, two small side lugs and a tapered body.” | AI caption; official colors are empty |
| vessel subclass/catalogue-only | `met_141619` | Official title/object name identify a Ring, medium jade; AI CV-research B labels it `cong`. | Official text versus AI structured judgment |

These examples show why source frequency must not be promoted directly to required vocabulary. `T157` is called “bottle” by AI while the official `object_type` is cup; `T146` is catalogued as a bottle but labeled vase; `D037` / `met_47660` is catalogued as a covered jar but labeled vase in AI data. Aliases such as `meiping`, `buckle`, and `flask` may require visual form definitions and separate classification choices rather than synonym replacement.

The caption vocabulary is also imperfect: morphological differences occur (dark appears 78 times in the normalized survey versus 75 literal substring hits), captions can leave positive motif flags unstated, and camera/stand/pedestal language can be conflated with object parts or orientation. Lexical counts are screening evidence, not a final taxonomy.

## Recommendation

Review additions in this order: visible form/parts, surface treatment, multiplicity, pose/orientation, color/appearance, scene/animal/medallion motifs, and specific vessel subclasses. Add a field only after defining its observation rule and unknown state. Preserve official catalogue values separately from AI caption and structured-label evidence. In particular, do not create a required material classifier from caption words alone, and do not treat `pale`/`dark` as interchangeable color classes.

The main outputs are `summary.json`, `source_manifest.json/csv`, `field_counts.json`, `concept_summary.json`, `concept_candidates.json/csv`, `evidence_examples.jsonl`, `joined_annotations.jsonl`, `overlap_by_item.jsonl`, `eval_jsonl_inventory.json`, `visual_query_judgment_counts.jsonl`, and `metadata_qrel_counts.jsonl`. Rebuild with:

```powershell
.\.venv\Scripts\python.exe scripts\v4_label_survey.py
```
