# Visual Retrieval of Chinese Museum Artifacts

Technical report draft for COMP5523. This manuscript covers the data, algorithms, training, experiments and limitations. Member contributions and the assigned peer review must be added by the group. The system supports text ranking, variable-size image sets and external-reference image matching. Domain adaptation is useful, while extra visual channels have selective rather than universal benefits.

## Domain and research questions

Museum visitors describe both visible properties and specialist concepts: porcelain, a green narrow-necked vase, or a Qianlong famille rose vase. These queries mix appearance with historical identity. Our main research question is how image representations and auxiliary visual evidence support increasingly specific retrieval when the gallery supplies only image pixels. We also ask whether a system can identify a variable-size relevant set instead of always returning a fixed number of nearest neighbours.

Online candidate scores use image embeddings, image-derived predictions or pixel descriptors. Gallery titles, dates and descriptions are not used for these decisions. Museum records provide training supervision, reference-source qualification, evaluation evidence and display information. Their use during training is distinct from looking up candidate labels during retrieval. CLIP establishes the contrastive dual-encoder baseline (Radford et al., 2021); SigLIP and multilingual SigLIP2 provide stronger comparison encoders (Zhai et al., 2023; Tschannen et al., 2025).

## Data and relevance construction

The frozen gallery contains 4,692 images from the Metropolitan Museum of Art (2,954) and Cleveland Museum of Art (1,738). Source URLs, rights and image hashes are retained. Official object families, related works and duplicates define 4,234 object groups; grouped train, development and test splits contain 3,474, 568 and 650 images. Group separation prevents photographs of the same object from crossing these boundaries.

We separately inspected a teammate's 10,200-image release: Met 3,904, Minneapolis Institute of Art 2,955, Cleveland 1,810 and Art Institute of Chicago 1,531. It contains no Palace Museum source. We downloaded model artifacts and catalog records, not the full image library. Expansion images are retrieval candidates, not automatically independent test examples.

The revised v5.1 evaluation core has 200 images: 60 development and 140 test. It reuses 120 historical visual-evaluation images and adds 80 deterministic, material-balanced test samples, with object-group separation. The main assistant viewed all 20 contact sheets and read the cached official catalog fields, recording multiple visible attributes and notes for each image. The 200 image/source reviews generate 4,000 decisions for 20 semantic intents, each expressed in English and Chinese. These are AI-assisted judgments, not 4,000 independent expert annotations. Historical test exposure and the non-random sampling limit external validity.

Relevance requires every stated condition. Compatible form families allow bottle/vase or bowl/waterpot for broad requests while retaining detailed forms. Several motifs may coexist. Grayscale photographs leave actual color unknown; primary body material is distinguished from accessories. A confirmed contradiction produces partial relevance (grade 1) or irrelevance (0), and all conditions satisfied gives grade 3. Evidence gaps produce unknown (-1), including a broad Qing attribution when the requested reign cannot be established. Specialist dates use source evidence; similar appearance cannot establish a reign.

Ranking uses nDCG@10 with graded gains and known-only P@5 with grade 3 as binary relevance. Unknowns are excluded with judgment coverage, so known-only precision is not the unfiltered displayed top-five accuracy. Sets use precision, recall and F1 over the fully judged test core, with micro counts, empty-prediction counts and precision bounds that treat accepted unknowns pessimistically or optimistically. Whole-gallery scores cover all 4,692 images, but exhaustive recall truth covers only 140 test images. English and Chinese are paired views of 20 intents, not 40 independent queries.

## Visual representations and workflows

The ranking route encodes a query and compares its normalized text vector with cached image vectors using cosine similarity. The production vision-only LoRA adapts SigLIP2 to museum image-caption pairs; its text tower remains unchanged. Bilingual query expansion is evaluated separately. Gallery embeddings are recomputed only when a relevant image-model checkpoint or preprocessing changes. LoRA supplies parameter-efficient adaptation (Hu et al., 2021), not an alternative to the visual encoder.

The teammate's new checkpoint additionally adapts vision q/k/v/out projections and the spatial pooling head attention output and MLP layers. It has 995,328 trainable parameters with rank 12 and alpha 24. Training uses 3,474 original training images, bilingual captions, 218 detailed AI captions and counterfactual attribute negatives. It was not trained on all 10,200 gallery images. We loaded the adapter and independently encoded 16 sample images to verify its released cache; minimum cosine agreement was approximately .999997. The unchanged text tower was also verified.

Classical controls describe foreground shape with contour statistics and HOG (Dalal & Triggs, 2005), color with HSV histograms, and texture with local binary-pattern statistics (Ojala et al., 2002). The automatic foreground pipeline initializes background from borders and applies GrabCut (Rother et al., 2004) with quality-based abstention. Raw-image and foreground descriptors are compared. DINOv2 supplies self-supervised image features (Oquab et al., 2023). A finite query-to-attribute bridge and train-only linear heads permit classical/DINO image representations to answer supported text attributes; they predict from pixels rather than compare stored item labels.

Auxiliary text fusion adds a development-selected fraction of normalized image-derived attribute scores to the normalized dual-encoder score. Unsupported attributes contribute no auxiliary evidence. We also test DINO neighbourhood averaging as soft score support. Hard exclusion is evaluated separately because an early visual filter can permanently remove correct results.

The set route scores every image before deciding membership. The existing GMM compares query cosine against generic museum-photo cosine and searches for a separated support mode. Two simpler controls use a shared development-selected cutoff or a logistic boundary over cosine, generic margin and whole-gallery standardized cosine. An HSV-assisted cutoff adds train-only color evidence. Thresholds maximize development macro F1; they impose no fixed output size. These fits use the same intents on disjoint development images, so they are not evidence of unseen-query calibration. The logistic output is a support score, not a verified probability of correctness. Runtime boundary files are bound to the image-index hash.

The specialist route directly searches museum APIs for reference photographs. Qualified mode supports V&A, MIA, AIC and Cleveland, with declared bounded bilingual vocabulary. It checks image identifiers, downloads and decodes bytes, verifies recognized source conditions, rejects dates inconsistent with the requested production period, and isolates references from target objects and duplicate photographs. A reign inscription alone is insufficient. Missing usable references returns an explicit empty state. Source checks establish provenance support; picture relevance remains a separate review question.

Reference and gallery pixels are compared with matching SigLIP2 and DINO encoders. Controls include a query-selected single reference, uniform mean, per-candidate maximum, balanced visual groups and query-assisted soft reference weights. The last method uses text to weight references and is disclosed as hybrid. Candidate metadata is absent from all these scores. AIC requests were blocked in this environment; MIA search and decoded images worked. V&A images retain attribution and educational-use restrictions, separate from open gallery training data.

## Results and analysis

Table 1 compares raw text on identical candidates and v5.1 test judgments. The production model remains the default. Point estimates favour it over the newer bilingual head adapter, but the paired intent confidence interval for their difference crosses zero; this is not a universal superiority claim.

| Model | EN nDCG | ZH nDCG | EN P@5 | ZH P@5 |
|---|---:|---:|---:|---:|
| CLIP | 0.6233 | 0.1408 | 0.550 | 0.110 |
| SigLIP2 | 0.7219 | 0.6385 | 0.640 | 0.620 |
| Production LoRA | 0.7954 | 0.7217 | 0.700 | 0.670 |
| Bilingual head LoRA | 0.7511 | 0.7104 | 0.650 | 0.640 |

We also ran the teammate's unmodified query-rewrite and compositional engine in an isolated process, with matched probes and identical .15 probe weight for both checkpoints. Its image-only rewrite workflow gives EN/ZH nDCG 0.8391/0.8063 on the production checkpoint and 0.7980/0.7903 on the new checkpoint. The compositional alternatives did not consistently improve this revised core. These controls do not replace production's frozen zero-probe-weight configuration. Both workflows were additionally run with the 10,200-image indices; unjudged added images were never assumed negative.

Four declared 10,200-gallery queries produced a pooled 25 unique top-result images. Eighteen were actually viewed; seven AIC images remained unknown after HTTP 403 and browser verification barriers. Blue-white retrieval was visually convincing, but Ming queries included Yuan/Qing objects and Qianlong queries included Yuan/Song objects or jade. Thus larger galleries and bilingual adaptation do not themselves solve historical attribution. The teammate's silver evaluator deliberately accepts grade 2 when type/material match despite an era or motif mismatch; it is not a stricter ground truth for our conjunctive task. We also verified a BCE date parsed as positive CE years in one MIA record.

Color assistance gives small ranking gains: production EN nDCG rises from 0.7954 to 0.8049 with raw HSV and 0.8190 with foreground HSV. Their paired bilingual intervals include zero. Shape and DINO attribute-head fusion select zero weight on development for the production model; texture is mixed. DINO graph averaging at .1 has a small positive exploratory interval, without correction for searching multiple methods. This supports selective soft assistance rather than universal replacement.

For a controlled image-image experiment, up to four correct development references per intent are matched to disjoint test objects. Eighteen intents have usable references. Production image similarity reaches nDCG 0.7174; DINO alone reaches 0.5941. Fixed exploratory fusion reaches 0.7347 with DINO and 0.7306 with foreground color, while their P@5 decreases. Shape and texture do not yield a consistent gain. Correct-reference selection here is controlled by dev judgments and does not measure external-search success. Inspected failures show background/photographic color and monochrome references can dominate simple visual similarity.

Table 2 shows variable-size set decisions. Logistic support improves recall and macro F1 over the adaptive GMM, with a precision cost; a simple score cutoff remains competitive, especially in Chinese. HSV assistance improves English more than Chinese. Macro precision excludes queries with no judged accepted items; accompanying machine-readable results report its defined count and micro statistics. No method is presented as a perfect membership test.

| Boundary | EN F1 | ZH F1 | EN micro P | EN micro R |
|---|---:|---:|---:|---:|
| Adaptive GMM | 0.4686 | 0.4613 | 0.6641 | 0.3452 |
| Shared score cutoff | 0.5292 | 0.5257 | 0.4426 | 0.7500 |
| Logistic boundary | 0.5734 | 0.5149 | 0.6497 | 0.5079 |
| HSV assisted cutoff | 0.5831 | 0.5240 | 0.6209 | 0.5198 |

For the same specialist query, Qianlong famille rose vase, eight V&A source records and actual reference pictures were checked. We then viewed all fourteen targets pooled from seven top-five variants and their source records. Raw and rewritten production text each returned 2/5 conforming examples; uniform-reference image matching returned 3/5, as did query-assisted weighting and visual-group balancing. A remaining false positive is painted enamel on copper alloy, visually similar to porcelain; another is a Shunzhi jar. This one-query improvement establishes a useful route, not robust performance across specialist queries or visual authentication.

![Same query text retrieval and external reference matching](../artifacts/relevance_v5/report_comparison.jpg)

Figure 1. Same query, top five text results above and qualified-reference image results below. Target photographs are Met/Cleveland open-access images. The lower row improves the first three matches but retains material and period failures.

## Efficiency and limitations

Existing RTX 3070 batch-16 full-gallery timing measured CLIP 126.4 s, first-generation SigLIP 113.9 s, SigLIP2 104.4 s, production LoRA 102.5 s and DINO FP32 122.0 s. These measurements include reading, preprocessing, vision encoding and normalization; they exclude loading, downloading and saving. Small timing differences are hardware/cache observations, not proof that an architecture is inherently faster. They are separate performance experiments and were not rerun for each quality evaluation.

With cached vectors, full-gallery cosine ranking took about 1.56 ms in the earlier controlled timing; DINO half-gallery filtering/reranking took about 3.99 ms and fusion 6.28 ms, excluding common text encoding. At 50% retention DINO preserved 94.3% of positives on average but only 67.6% on its worst query. Development safety therefore selected 100% retention. Extra preprocessing is not justified as a speed optimization at this gallery size. Earlier 338-image foreground/downstream tests also failed to establish that the more elaborate background method beats simple controls, so background removal remains optional with abstention.

The evaluation core is small, AI-assisted and historically reused; specialist positives are sparse. Query-level bootstrap intervals do not capture annotation uncertainty, and whole-gallery false negatives remain incompletely judged. Set boundaries need future unseen-query and prevalence testing. Reference acquisition has access and vocabulary limits; reference resemblance cannot recover invisible date, authenticity or technique. Useful next research would prioritise additional independently adjudicated query-result pairs and local motif verification, rather than adding unvalidated labels or indiscriminate model complexity.

## Reproducibility and AI use

Frozen judgments, image hashes, model hashes, split membership, scores, per-query metrics and source ledgers are stored with the project. Comparisons reuse cached gallery embeddings; sixteen gallery images were newly encoded for adapter verification, and external reference photographs are encoded as needed. Scripts reproduce the raw model, actual teammate workflow, image-component and specialist-case comparisons. The running service exposes ranking, whole-gallery sets and qualified external-reference matching with explicit failure states. Single-query and batched FP16 text inference can slightly shift threshold-adjacent decisions; the API acceptance records these numerical differences separately.

AI assisted implementation, experiment execution, image/source review and this technical draft. Earlier independent AI annotation remains disclosed; no new subagents were used in this revision. The main assistant checked code, artifacts and returned images, and retained failed or inconclusive results. The group's domain choices, professor discussion and teammate implementation are distinct inputs. This draft does not invent individual contributions, expert certification or the not-yet-assigned peer review.

## References

Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. Proceedings of CVPR, 1, 886–893. https://doi.org/10.1109/CVPR.2005.177

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv. https://arxiv.org/abs/2106.09685

Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(7), 971–987. https://doi.org/10.1109/TPAMI.2002.1017623

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P., Li, S., Misra, I., Rabbat, M., Sharma, V., … Bojanowski, P. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv. https://arxiv.org/abs/2304.07193

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv. https://arxiv.org/abs/2103.00020

Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: Interactive foreground extraction using iterated graph cuts. ACM Transactions on Graphics, 23(3), 309–314. https://doi.org/10.1145/1015706.1015720

Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaff, O., Harmsen, J., Steiner, A., & Zhai, X. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv. https://arxiv.org/abs/2502.14786

Victoria and Albert Museum. (n.d.). Collections API. Retrieved November 8, 2026, from https://developers.vam.ac.uk/

Zhai, X., Mustafa, B., Kolesnikov, A., & Beyer, L. (2023). Sigmoid Loss for Language Image Pre-Training. arXiv. https://arxiv.org/abs/2303.15343
