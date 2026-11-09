# Boundary no-match controls

Eight English queries with an explicitly empty relevance set over the same
exhaustive 96-image visual-audit gallery used by the dev/test boundary study.
They were authored separately from the 48 audit intents and include two
domain-adjacent distractors (`ceramic teapot...sports car`, `bronze mirror...QR
code`) so a system cannot pass solely by treating every museum-domain query as
relevant or every unfamiliar query as empty.

The public API cannot create negative files. The basis for each empty set is
the visible content of the audited gallery and the query's required assertion,
not a catalogue metadata search. This is an AI-assisted visual control, not
human relevance gold.
