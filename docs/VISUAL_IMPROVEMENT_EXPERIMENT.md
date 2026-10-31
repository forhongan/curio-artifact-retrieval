# Visual improvement experiment

This report records the first post-freeze ablation. The production index and
`image_only` route were not modified.

## Multi-crop result

The experiment uses the existing `siglip_adapted` encoder. For each gallery
image, `full` encodes the original image once. `multicrop` encodes the original
image, a center square, and four overlapping corner crops, then averages their
normalized vectors. Query text and qrels are unchanged.

The evaluation is the 24-query, 60-image `eval/visual_dev` gallery. Its labels
are AI image-only judgments, so this is an ablation signal rather than human
ground truth.

| Language | Mode | nDCG@10 | AP@10 | P@5 | MRR@10 | Gallery encoding |
|---|---|---:|---:|---:|---:|---:|
| English | full | 0.9300 | 0.9061 | 0.3500 | 0.9792 | 2.53 s |
| English | multicrop | 0.9304 | 0.9084 | 0.3417 | 1.0000 | 6.33 s |
| Chinese | full | 0.9053 | 0.8701 | 0.3417 | 0.9792 | 2.57 s |
| Chinese | multicrop | 0.8851 | 0.8433 | 0.3417 | 0.9583 | 5.23 s |

The English gain is within the uncertainty of this small gallery, while the
Chinese result declines. The extra image encoding cost is about 2x–2.5x. The
current evidence therefore does not justify replacing the production index.
Multi-crop remains available as a research-mode candidate for failure cases or
as a later local-region reranker after a larger human-labeled evaluation.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe scripts/eval_multicrop.py --mode full --language en
.\.venv\Scripts\python.exe scripts/eval_multicrop.py --mode multicrop --language en
.\.venv\Scripts\python.exe scripts/eval_multicrop.py --mode full --language zh
.\.venv\Scripts\python.exe scripts/eval_multicrop.py --mode multicrop --language zh
```

## Ontology expansion

The query vocabulary now includes catalogue-supported categories such as
`saucer`, `tile`, `ring`, `jewelry`, `garment`, `book`, `fragment`, and
`bottle`, plus high-frequency materials, motifs, colors, and shape phrases.
The generated [ontology coverage report](ONTOLOGY_COVERAGE.md) lists support
counts. These additions improve query parsing and dense text rewriting; they
do not claim that the current weakly supervised probes can verify the new
classes. The API reports them as `unknown` until a matching visual head is
trained and audited.
