# Specialist query rules

`curio.query.ontology.SPECIALIST_GLOSSARY` is a short project-authored bilingual
paraphrase table. These are manual query rules, not an externally verified
terminology database. The parser appends plain visual English to encoder input;
the original query stays intact. `parsed.specialist_expansions` records each rule,
its appended text and `basis: project_manual_rule`.

| User wording | Appended visual text |
| --- | --- |
| 青花 | blue and white porcelain decoration |
| 粉彩 / famille rose | porcelain with soft multicolored enamel decoration |
| 镂空 / 鏤空 | openwork pierced decoration |
| 描金 | gilt golden painted decoration |

The expansion does not create material, color, motif, era or museum slot labels.
The pre-existing 青花 rule still recognizes porcelain/blue/white directly from
the user's text. Broad descriptions such as multicolored enamel do not imply a
particular color, motif, dynasty or source museum.

转心/轉心 is preserved with a warning: an internal rotating structure cannot be
confirmed from a single image, and this rule table does not turn it into a
definite visual structure. 流心 is also preserved and explicitly marked
unverified; it is never silently corrected to 转心 or another object name.
Unknown wording remains encoder input rather than fabricated hard constraints.

These rules are shared query preprocessing. `baseline` still sends the raw
query to the encoder. `image_only` uses rewritten image/text similarity and
covered image-probe probabilities, never metadata era/slot scores or BM25.
The presence of a paraphrase or an image-probe prediction is not confirmation
that the requested property is visible.

The separate shape vocabulary recognizes `narrow_mouth`, `wide_mouth`,
`handles`, `footed`, and `round_body`. For the current small query vocabulary,
narrow mouth and narrow neck share `narrow_mouth`; this is an approximate
grouping, not a claim that the two anatomical features are identical.
Likewise footed is a broad shape class: it does not certify a requested exact
number of legs. Count and spatial relationships retain explicit warnings.
Only a trained image head covering the canonical class can supply probability
support; absent head classes remain unknown.
