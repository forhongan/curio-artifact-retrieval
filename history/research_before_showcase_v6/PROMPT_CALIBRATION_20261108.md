# SigLIP2 prompt-template calibration (2026-11-08)

Status: completed and reproducible. The experiment uses `google/siglip2-base-patch16-224`
at revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, the frozen
`artifacts/siglip/image_embeddings.npy` gallery vectors, and the existing visual
audit queries. It changes ranking inputs only. Image vectors, judgments,
production selection, and other CV research files are unchanged.

The fixed dev choices are:

| Language | Fixed template | Dev source |
| --- | --- | --- |
| English | `strip_leading_article` | Remove one leading English `a`, `an`, or `the`. |
| Chinese | `prefix_image_of` | `image of {text}`. |

These choices were selected on `eval/visual_dev` before test evaluation by the
highest mean nDCG@10, with raw winning a numerical tie. `nDCG@5`, P@5, AP, recall,
MRR, and significance were not selection criteria. The selected templates do
not show a reliable improvement. On the fixed supplementary test comparison,
English has nDCG@10 difference `-0.0044` versus raw, with paired 95% CI
`[-0.0256, +0.0137]`; Chinese has `+0.0040`, CI `[-0.0303, +0.0406]`. Neither
selected comparison remains significant after Holm correction (`p=1.0000` in
both languages).

## Inputs, split boundary, and registry

The experiment used 24 dev and 24 test queries. English and Chinese refer to the
same 24 queries per split. Each split has one complete 60-image audit gallery;
every query has a grade for every gallery image. Thus each metric is a query-level
mean over 24 paired observations, and pairing by query is valid.

The test split had appeared in earlier CV experiments. The two commands below
are enforced in code: the first writes `selection_manifest.json` and its SHA-256;
the second verifies those bytes, the prompt registry hash, encoder revision, and
image-index bytes before reading the test queries. This preserves the ordering
discipline, but the test result remains a fixed supplementary comparison, not a
newly blind holdout. The query and qrel hashes are recorded in the selection
manifest.

All 19 registered inputs, in deterministic registry order, are:

| Family | Templates |
| --- | --- |
| Baseline | `raw` |
| Case/punctuation | `lowercase`, `capitalize_first`, `terminal_period`, `remove_terminal_punctuation` |
| Article | `strip_leading_article`, `prepend_the` |
| Prefix | `prefix_a_photo_of`, `prefix_photo_of`, `prefix_a_photograph_of`, `prefix_a_picture_of`, `prefix_an_image_of`, `prefix_image_of`, `prefix_museum_photograph_of`, `prefix_close_up_photograph_of` |
| Suffix | `suffix_plain_background`, `suffix_museum_object` |
| Caption-like | `caption_image_showing`, `caption_illustrating` |

Prefixes preserve source text case and leading article except where the prefix
itself supplies an article; this avoids combining an uncontrolled lowercase
change with the prefix intervention. `capitalize_first` uppercases the first
character and lowercases the remaining characters. `terminal_period` replaces
terminal punctuation with one period. `strip_leading_article` is an exact no-op
for Chinese strings without an English leading article.

## Methods

Ranking is text vector cosine against the split's frozen gallery. Raw query text
is the reference. Each template receives the same gallery and qrels. The metrics
are:

- P@5 with the repository relevance threshold `grade >= 2`;
- nDCG@5 and nDCG@10 using exponential gains `2**grade - 1`;
- truncated AP@10 with the full relevant-set denominator;
- Recall@10 at `grade >= 2`; and
- MRR@10 at `grade >= 2`.

Every displayed mean is a query-level mean. Uncertainty uses 2,000 paired
query bootstrap resamples with seed 5523 and a 95% percentile interval. Paired
deltas are `template - raw`. The two-sided paired sign-flip test uses 10,000
resamples with seed 5523 and fixes exact zero differences at zero. Holm correction
is applied separately to the nDCG@10 family of 18 non-raw templates within each
split and language. The artifact also tests every other metric with a separate
18-template Holm family; these are secondary outcomes and do not change selection.

## Frozen selections

| Split | Language | Raw P@5 | Selected P@5 | Raw nDCG@5 | Selected nDCG@5 | Raw nDCG@10 | Selected nDCG@10 | Raw MRR@10 | Selected MRR@10 | ΔnDCG@10 [95% CI] | p | p Holm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | en | 0.3667 | 0.3750 | 0.8894 | 0.9223 | 0.8882 | 0.9168 | 0.9514 | 1.0000 | +0.0286 [-0.0006, +0.0649] | 0.1176 | 1.0000 |
| dev | zh | 0.3083 | 0.3083 | 0.8097 | 0.8158 | 0.8327 | 0.8334 | 0.8958 | 0.9236 | +0.0007 [-0.0222, +0.0311] | 0.9771 | 1.0000 |
| test | en | 0.4583 | 0.4583 | 0.8880 | 0.8816 | 0.9076 | 0.9033 | 0.9306 | 0.9306 | -0.0044 [-0.0256, +0.0137] | 0.6802 | 1.0000 |
| test | zh | 0.3833 | 0.4000 | 0.7416 | 0.7738 | 0.7930 | 0.7971 | 0.8174 | 0.8542 | +0.0040 [-0.0303, +0.0406] | 0.8207 | 1.0000 |

The English nDCG@5 and nDCG@10 directions differ between dev and test. The
Chinese test differences are slightly positive but uncertain. These data do not
establish that either fixed template improves retrieval.

## Development results

All values below are query-level means. ΔnDCG@10 and its p values compare each
template with raw after the stated Holm correction.

### English

| Template | P@5 | nDCG@5 | nDCG@10 | AP@10 | Recall@10 | MRR@10 | ΔnDCG@10 [95% CI] | p | p Holm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3667 | 0.8894 | 0.8882 | 0.8818 | 0.9861 | 0.9514 | — | — | — |
| lowercase | 0.3667 | 0.9081 | 0.9106 | 0.9028 | 0.9740 | 0.9722 | +0.0224 [+0.0024, +0.0440] | 0.0478 | 0.8125 |
| capitalize_first | 0.3667 | 0.8894 | 0.8882 | 0.8818 | 0.9861 | 0.9514 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| terminal_period | 0.3667 | 0.8894 | 0.8882 | 0.8818 | 0.9861 | 0.9514 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| remove_terminal_punctuation | 0.3667 | 0.9014 | 0.9023 | 0.9031 | 0.9792 | 0.9722 | +0.0142 [-0.0045, +0.0352] | 0.2277 | 1.0000 |
| strip_leading_article | 0.3750 | 0.9223 | 0.9168 | 0.9132 | 0.9740 | 1.0000 | +0.0286 [-0.0006, +0.0649] | 0.1176 | 1.0000 |
| prepend_the | 0.3667 | 0.8865 | 0.8884 | 0.8754 | 0.9792 | 0.9514 | +0.0002 [-0.0196, +0.0220] | 0.9876 | 1.0000 |
| prefix_a_photo_of | 0.3500 | 0.8785 | 0.8877 | 0.8684 | 0.9948 | 0.9514 | -0.0005 [-0.0199, +0.0221] | 0.9653 | 1.0000 |
| prefix_photo_of | 0.3583 | 0.8861 | 0.8859 | 0.8716 | 0.9740 | 0.9514 | -0.0023 [-0.0208, +0.0188] | 0.8226 | 1.0000 |
| prefix_a_photograph_of | 0.3417 | 0.8760 | 0.8786 | 0.8518 | 0.9601 | 0.9514 | -0.0096 [-0.0311, +0.0150] | 0.4560 | 1.0000 |
| prefix_a_picture_of | 0.3500 | 0.8787 | 0.8913 | 0.8773 | 0.9948 | 0.9583 | +0.0032 [-0.0128, +0.0238] | 0.8143 | 1.0000 |
| prefix_an_image_of | 0.3583 | 0.8788 | 0.8878 | 0.8673 | 0.9948 | 0.9514 | -0.0003 [-0.0186, +0.0211] | 0.9765 | 1.0000 |
| prefix_image_of | 0.3667 | 0.8884 | 0.8867 | 0.8645 | 0.9740 | 0.9514 | -0.0015 [-0.0207, +0.0201] | 0.8948 | 1.0000 |
| prefix_museum_photograph_of | 0.3333 | 0.8658 | 0.8642 | 0.8395 | 0.9323 | 0.9514 | -0.0240 [-0.0423, -0.0056] | 0.0196 | 0.3528 |
| prefix_close_up_photograph_of | 0.3333 | 0.8719 | 0.8755 | 0.8559 | 0.9601 | 0.9722 | -0.0127 [-0.0424, +0.0252] | 0.5190 | 1.0000 |
| suffix_plain_background | 0.3667 | 0.8938 | 0.8999 | 0.8843 | 0.9948 | 0.9583 | +0.0117 [-0.0010, +0.0286] | 0.1455 | 1.0000 |
| suffix_museum_object | 0.3500 | 0.8567 | 0.8567 | 0.8138 | 0.9462 | 0.9097 | -0.0314 [-0.0665, -0.0032] | 0.0517 | 0.8271 |
| caption_image_showing | 0.3667 | 0.8876 | 0.8890 | 0.8635 | 0.9601 | 0.9514 | +0.0008 [-0.0183, +0.0233] | 0.9431 | 1.0000 |
| caption_illustrating | 0.3500 | 0.8861 | 0.8943 | 0.8730 | 0.9948 | 0.9583 | +0.0061 [-0.0099, +0.0248] | 0.5718 | 1.0000 |

### Chinese

| Template | P@5 | nDCG@5 | nDCG@10 | AP@10 | Recall@10 | MRR@10 | ΔnDCG@10 [95% CI] | p | p Holm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3083 | 0.8097 | 0.8327 | 0.7592 | 0.8976 | 0.8958 | — | — | — |
| lowercase | 0.3083 | 0.8097 | 0.8327 | 0.7592 | 0.8976 | 0.8958 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| capitalize_first | 0.3083 | 0.8097 | 0.8327 | 0.7592 | 0.8976 | 0.8958 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| terminal_period | 0.3083 | 0.8098 | 0.8276 | 0.7579 | 0.8976 | 0.8958 | -0.0051 [-0.0092, -0.0017] | 0.0154 | 0.2618 |
| remove_terminal_punctuation | 0.3083 | 0.8153 | 0.8299 | 0.7538 | 0.8837 | 0.8958 | -0.0028 [-0.0099, +0.0052] | 0.5046 | 1.0000 |
| strip_leading_article | 0.3083 | 0.8097 | 0.8327 | 0.7592 | 0.8976 | 0.8958 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| prepend_the | 0.3083 | 0.8087 | 0.8307 | 0.7742 | 0.9323 | 0.8958 | -0.0020 [-0.0179, +0.0133] | 0.8217 | 1.0000 |
| prefix_a_photo_of | 0.3167 | 0.8115 | 0.8129 | 0.7542 | 0.8906 | 0.8889 | -0.0198 [-0.0509, +0.0078] | 0.2225 | 1.0000 |
| prefix_photo_of | 0.3167 | 0.8189 | 0.8201 | 0.7580 | 0.8698 | 0.9097 | -0.0126 [-0.0474, +0.0253] | 0.5142 | 1.0000 |
| prefix_a_photograph_of | 0.3000 | 0.8019 | 0.8085 | 0.7590 | 0.8576 | 0.9028 | -0.0242 [-0.0605, +0.0180] | 0.2459 | 1.0000 |
| prefix_a_picture_of | 0.3083 | 0.8068 | 0.8185 | 0.7653 | 0.8767 | 0.9005 | -0.0141 [-0.0461, +0.0183] | 0.4131 | 1.0000 |
| prefix_an_image_of | 0.3000 | 0.8003 | 0.8227 | 0.7580 | 0.8767 | 0.9028 | -0.0100 [-0.0462, +0.0303] | 0.6338 | 1.0000 |
| prefix_image_of | 0.3083 | 0.8158 | 0.8334 | 0.7734 | 0.8837 | 0.9236 | +0.0007 [-0.0222, +0.0311] | 0.9771 | 1.0000 |
| prefix_museum_photograph_of | 0.3083 | 0.7723 | 0.7791 | 0.7009 | 0.8698 | 0.8519 | -0.0536 [-0.1017, -0.0130] | 0.0197 | 0.3152 |
| prefix_close_up_photograph_of | 0.2917 | 0.7248 | 0.7373 | 0.6698 | 0.7795 | 0.7917 | -0.0954 [-0.1744, -0.0324] | 0.0022 | 0.0396 |
| suffix_plain_background | 0.3167 | 0.8083 | 0.8114 | 0.7441 | 0.8628 | 0.9010 | -0.0212 [-0.0605, +0.0195] | 0.3111 | 1.0000 |
| suffix_museum_object | 0.3083 | 0.8115 | 0.8158 | 0.7537 | 0.8715 | 0.9167 | -0.0168 [-0.0626, +0.0308] | 0.4990 | 1.0000 |
| caption_image_showing | 0.2917 | 0.8067 | 0.8247 | 0.7632 | 0.8628 | 0.9236 | -0.0080 [-0.0396, +0.0305] | 0.6893 | 1.0000 |
| caption_illustrating | 0.3000 | 0.8057 | 0.8077 | 0.7517 | 0.8212 | 0.8958 | -0.0250 [-0.0672, +0.0186] | 0.2664 | 1.0000 |

In dev English, `prefix_museum_photograph_of` has a lower point estimate than
raw (`-0.0240`) but does not remain significant after Holm (adjusted p
`0.3528`). In dev Chinese, `prefix_close_up_photograph_of` has a lower point
estimate (`-0.0954`) and remains significant after Holm (adjusted p `0.0396`).
These are family-specific split results, not a general claim that all photograph
prefixes are harmful.

## Fixed supplementary test

The test table was generated after the selection manifest was frozen. It is
reported in full even where other templates happen to rank above the fixed
choice; no test template is promoted after seeing the result.

### English

| Template | P@5 | nDCG@5 | nDCG@10 | AP@10 | Recall@10 | MRR@10 | ΔnDCG@10 [95% CI] | p | p Holm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.4583 | 0.8880 | 0.9076 | 0.8753 | 0.9688 | 0.9306 | — | — | — |
| lowercase | 0.4583 | 0.8880 | 0.9076 | 0.8753 | 0.9688 | 0.9306 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| capitalize_first | 0.4667 | 0.8931 | 0.9094 | 0.8840 | 0.9479 | 0.9583 | +0.0018 [-0.0173, +0.0200] | 0.8503 | 1.0000 |
| terminal_period | 0.4667 | 0.8970 | 0.9136 | 0.8823 | 0.9688 | 0.9514 | +0.0060 [-0.0045, +0.0190] | 0.3870 | 1.0000 |
| remove_terminal_punctuation | 0.4583 | 0.8880 | 0.9076 | 0.8753 | 0.9688 | 0.9306 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| strip_leading_article | 0.4583 | 0.8816 | 0.9033 | 0.8692 | 0.9688 | 0.9306 | -0.0044 [-0.0256, +0.0137] | 0.6802 | 1.0000 |
| prepend_the | 0.4417 | 0.8849 | 0.9038 | 0.8825 | 0.9479 | 0.9479 | -0.0038 [-0.0169, +0.0076] | 0.5962 | 1.0000 |
| prefix_a_photo_of | 0.4583 | 0.8887 | 0.9132 | 0.8768 | 0.9688 | 0.9792 | +0.0056 [-0.0118, +0.0264] | 0.6946 | 1.0000 |
| prefix_photo_of | 0.4583 | 0.8758 | 0.8948 | 0.8572 | 0.9410 | 0.9201 | -0.0128 [-0.0268, -0.0006] | 0.0840 | 1.0000 |
| prefix_a_photograph_of | 0.4500 | 0.8722 | 0.8925 | 0.8633 | 0.9549 | 0.9479 | -0.0152 [-0.0297, -0.0029] | 0.0357 | 0.5711 |
| prefix_a_picture_of | 0.4583 | 0.8863 | 0.9090 | 0.8842 | 0.9688 | 0.9514 | +0.0013 [-0.0152, +0.0187] | 0.8876 | 1.0000 |
| prefix_an_image_of | 0.4583 | 0.8768 | 0.8970 | 0.8659 | 0.9618 | 0.9375 | -0.0106 [-0.0238, +0.0026] | 0.1363 | 1.0000 |
| prefix_image_of | 0.4417 | 0.8644 | 0.8950 | 0.8542 | 0.9618 | 0.9236 | -0.0126 [-0.0279, -0.0003] | 0.0804 | 1.0000 |
| prefix_museum_photograph_of | 0.4250 | 0.8494 | 0.8787 | 0.8582 | 0.9497 | 0.9271 | -0.0289 [-0.0502, -0.0096] | 0.0058 | 0.1044 |
| prefix_close_up_photograph_of | 0.4333 | 0.8694 | 0.9031 | 0.8674 | 0.9670 | 0.9583 | -0.0045 [-0.0199, +0.0106] | 0.5817 | 1.0000 |
| suffix_plain_background | 0.4500 | 0.8744 | 0.9024 | 0.8709 | 0.9740 | 0.9444 | -0.0052 [-0.0223, +0.0122] | 0.5664 | 1.0000 |
| suffix_museum_object | 0.4333 | 0.8576 | 0.8870 | 0.8600 | 0.9549 | 0.9306 | -0.0207 [-0.0374, -0.0049] | 0.0182 | 0.3094 |
| caption_image_showing | 0.4500 | 0.8770 | 0.8915 | 0.8655 | 0.9618 | 0.9458 | -0.0161 [-0.0305, -0.0027] | 0.0362 | 0.5711 |
| caption_illustrating | 0.4250 | 0.8546 | 0.8954 | 0.8487 | 0.9635 | 0.9236 | -0.0122 [-0.0283, +0.0004] | 0.1277 | 1.0000 |

### Chinese

| Template | P@5 | nDCG@5 | nDCG@10 | AP@10 | Recall@10 | MRR@10 | ΔnDCG@10 [95% CI] | p | p Holm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.3833 | 0.7416 | 0.7930 | 0.7091 | 0.9167 | 0.8174 | — | — | — |
| lowercase | 0.3833 | 0.7416 | 0.7930 | 0.7091 | 0.9167 | 0.8174 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| capitalize_first | 0.3833 | 0.7416 | 0.7930 | 0.7091 | 0.9167 | 0.8174 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| terminal_period | 0.3917 | 0.7562 | 0.7799 | 0.7145 | 0.8958 | 0.8160 | -0.0131 [-0.0515, +0.0251] | 0.5253 | 1.0000 |
| remove_terminal_punctuation | 0.3833 | 0.7416 | 0.7930 | 0.7091 | 0.9167 | 0.8174 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| strip_leading_article | 0.3833 | 0.7416 | 0.7930 | 0.7091 | 0.9167 | 0.8174 | +0.0000 [+0.0000, +0.0000] | 1.0000 | 1.0000 |
| prepend_the | 0.3667 | 0.7082 | 0.7596 | 0.6801 | 0.9083 | 0.7646 | -0.0334 [-0.0770, +0.0056] | 0.1483 | 1.0000 |
| prefix_a_photo_of | 0.3917 | 0.7434 | 0.7918 | 0.7401 | 0.9236 | 0.8102 | -0.0012 [-0.0565, +0.0480] | 0.9687 | 1.0000 |
| prefix_photo_of | 0.3750 | 0.7332 | 0.7656 | 0.6981 | 0.8875 | 0.8035 | -0.0274 [-0.0757, +0.0114] | 0.2961 | 1.0000 |
| prefix_a_photograph_of | 0.3833 | 0.7348 | 0.7915 | 0.7397 | 0.9236 | 0.8208 | -0.0016 [-0.0600, +0.0511] | 0.9621 | 1.0000 |
| prefix_a_picture_of | 0.4000 | 0.7705 | 0.8062 | 0.7577 | 0.9097 | 0.8583 | +0.0131 [-0.0345, +0.0640] | 0.6350 | 1.0000 |
| prefix_an_image_of | 0.3833 | 0.7270 | 0.7728 | 0.7148 | 0.9014 | 0.8021 | -0.0202 [-0.0759, +0.0281] | 0.5161 | 1.0000 |
| prefix_image_of | 0.4000 | 0.7738 | 0.7971 | 0.7374 | 0.9135 | 0.8542 | +0.0040 [-0.0303, +0.0406] | 0.8207 | 1.0000 |
| prefix_museum_photograph_of | 0.3750 | 0.7310 | 0.7691 | 0.6991 | 0.8767 | 0.8150 | -0.0239 [-0.1087, +0.0514] | 0.6098 | 1.0000 |
| prefix_close_up_photograph_of | 0.3750 | 0.7668 | 0.8109 | 0.7513 | 0.9149 | 0.8646 | +0.0178 [-0.0332, +0.0719] | 0.5404 | 1.0000 |
| suffix_plain_background | 0.3417 | 0.6975 | 0.7486 | 0.6742 | 0.8771 | 0.8201 | -0.0444 [-0.0945, +0.0073] | 0.1121 | 1.0000 |
| suffix_museum_object | 0.3333 | 0.6489 | 0.7006 | 0.6186 | 0.8646 | 0.7205 | -0.0924 [-0.1846, -0.0100] | 0.0580 | 1.0000 |
| caption_image_showing | 0.3333 | 0.6576 | 0.7331 | 0.6708 | 0.8875 | 0.7866 | -0.0599 [-0.1189, -0.0045] | 0.0562 | 1.0000 |
| caption_illustrating | 0.3583 | 0.6999 | 0.7605 | 0.7016 | 0.8889 | 0.7897 | -0.0326 [-0.0900, +0.0232] | 0.3060 | 1.0000 |

## Mechanism and failures

The SigLIP2 text tower is sensitive to wording, but not uniformly. Prefixes
and suffixes alter the complete caption context; stripping an article removes a
token while preserving the object description. Case and punctuation are
conditional interventions: `lowercase` was a literal no-op on all dev Chinese
queries and all test Chinese queries, while it changed all dev English queries
and none of the test English queries. `terminal_period` was a no-op on dev
English because its queries already ended with `.`; `remove_terminal_punctuation`
was a no-op on test English and all test Chinese. The artifact's per-query
scores and mechanism tables show these literal, embedding, and top-10 changes.

Actual token cuts can be inspected in `dev/text_embeddings.json` and
`test/text_embeddings.json`: vectors are ordered query-major then template
registry order, and the `.npy` files carry SHA-256 hashes. A literal no-op can
still produce an embedding difference at floating-point tolerance; compare the
input-change count before attributing an effect to case or punctuation.

The reasons for uncertainty are concrete:

- Each split has only 24 query clusters; each cluster supplies two language
  observations of the same images. Repeated image structure and shared query
  families mean the effective information is below 48 independent samples.
- The 60-image gallery is a curated AI visual audit, not exhaustive relevance
  over 4,692 catalogue images. Its graded qrels are not human gold.
- The test split had prior exposure elsewhere, so even a correctly frozen choice
  is not a new blind test.
- Prefix edits can shift the text representation away from the reference prompt
  distribution. `prefix_museum_photograph_of` has lower nDCG on both dev
  languages; `prefix_close_up_photograph_of` is especially weak in Chinese.
- English leading-article removal changes 23/24 dev and 24/24 test inputs, but
  its dev gain does not survive the interval/Holm rule and reverses on test.
- Different case/punctuation behavior across splits reflects the source query
  formatting: dev English is capitalized and terminal-punctuated; test English
  is neither. Treating the variants as equivalent across splits would be wrong.
- Family-wise correction covers the registry's tested templates. Adding new
  templates after inspecting test requires a new family and cannot be described
  as validation of this frozen selection.

## Reproduction

Run from `F:\Curio` with the project environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_prompt_calibration.py tests/test_metrics.py -q
.\.venv\Scripts\ruff.exe check scripts/run_prompt_calibration.py tests/test_prompt_calibration.py
.\.venv\Scripts\python.exe scripts/run_prompt_calibration.py --phase select-dev
.\.venv\Scripts\python.exe scripts/run_prompt_calibration.py --phase evaluate-test
.\.venv\Scripts\python.exe scripts/run_prompt_calibration.py --phase verify
```

`select-dev` writes and freezes the choice. `evaluate-test` refuses to read the
test split unless the selection bytes, prompt registry, model revision, and image
index are unchanged. `verify` checks every listed artifact hash. The complete
artifact manifest is `artifacts/prompt_calibration_20261108/artifact_manifest.json`.

The current selection-manifest SHA-256 is
`7e6b785858eeca216cd20460ee6d2f304dfc4b0d98ec14e1c8e0092cc1eb8feb`.
The experiment implementation is `scripts/run_prompt_calibration.py`; focused
tests are `tests/test_prompt_calibration.py`.

## Artifact guide

| Path | Contents |
| --- | --- |
| `dev/metrics.md`, `test/metrics.md` | Complete template metric tables and selected-template comparison. |
| `dev/metrics.csv`, `test/metrics.csv` | Mean and bootstrap interval by language, template, and metric. |
| `dev/paired_comparisons.csv`, `test/paired_comparisons.csv` | Paired difference, interval, raw p, and Holm p for each template/metric. |
| `dev/query_metrics.csv`, `test/query_metrics.csv` | One row per language, template, and query with all six metrics. |
| `dev/mechanism.md`, `test/mechanism.md` | Case/punctuation input, embedding, and top-10 change counts. |
| `dev/results.json`, `test/results.json` | Full metrics, per-query rows, top-10 rankings, and paired results. |
| `dev/text_embeddings_*.npy`, `test/text_embeddings_*.npy` | Reproducible text vectors with ordering and hashes in sibling JSON. |
| `selection_manifest.json`, `selection_manifest.sha256` | Frozen dev choices, data/index hashes, registry, and inference rule. |
| `artifact_manifest.json` | SHA-256 for every generated artifact. |

Only the primary mean nDCG@10 table is used to choose and report the fixed result.
The secondary metrics remain available for diagnosis and do not justify a
post-test template switch.
