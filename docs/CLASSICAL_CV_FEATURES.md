# Classical image-only features

`src/curio/represent/classical_cv.py` implements deterministic, untrained pixel features for controlled comparisons. It reads pixels only: no CLIP, text, captions, labels or catalog metadata. A separate supervised classifier is necessary to map a textual attribute to these features; this module does not solve the semantic text-to-image alignment.

Image orientation is EXIF-corrected, longest side downscaled to 192, with aspect ratio preserved and padded to 128 square. Both branches calculate HOG (9 orientations, 16-pixel cells, 2×2 blocks), HSV histograms (12×4×4 bins) and uniform LBP (radius 1/2; 8/16 neighbors). Color and texture use a 1×1 plus 2×2 spatial pyramid, with square-root normalized histograms. `raw_*` uses the complete image; `gc_*` first applies automatic GrabCut initialized by a 5% border rectangle for 3 iterations, then masks outside pixels to neutral gray. `gc_shape` additionally has silhouette occupancy profiles, seven signed log Hu moments and size statistics. `raw_all`/`gc_all` concatenate and L2-normalize channels. These parameters are fixed independently of dev/test performance.

GrabCut can fail on low contrast, touching objects, complex background or non-centered scenes. It flags unusable masks for area outside 3–90%, short image sides under 12px or OpenCV failure. Quality heuristics cannot verify segmentation correctness: accepted does not mean ground-truth foreground. On rejection `gc_hog/color/texture` are identical copies of raw features and `gc_shape` is zero; `gc_valid` and per-image reason record fallback. Do not report fallback as successful segmentation. This method is automatic, with no interactive annotations.

Feature extraction reads a JSONL manifest with exactly `item_id`, `image_path`, `image_sha256`, verifies each image hash, uses a fixed per-image OpenCV seed, and stores ID ordering, features, diagnostics and library/code/input/output hashes. The `.npz` contains `gc_valid` and the named feature matrices. A manifest can be generated from a richer catalog by copying only these three fields, then run:

```powershell
.venv/Scripts/python.exe scripts/build_classical_cv.py --manifest artifacts/classical_cv/pixels.jsonl --output artifacts/classical_cv --workers 8
.venv/Scripts/python.exe -m pytest tests/test_classical_cv.py -q
```

Requirements: `opencv-python-headless==5.0.0.93` and `scikit-image==0.26.0` in addition to existing project requirements. Versions are recorded in the output manifest. Multi-process execution isolates OpenCV random seeds; extraction time includes reading, processing, and saving every image. Comparative retrieval quality, training labels, CPU/GPU timing, and evaluation protocol belong in the experiment report.

The frozen 4,692-image dataset was extracted with 8 processes in **243.076 seconds** on 2026-11-04. The checks accepted 4,492 masks and explicitly fell back for 200; this is not mask accuracy, since only simple area and processing checks are applied. Matrices and provenance are in `artifacts/classical_cv/`, including the pixel-only input manifest. For example raw HOG has 1,764 dimensions, raw HSV 960, raw LBP 140, and concatenated raw/GrabCut are 2,864/2,938. `features.npz` SHA-256 is `1e830f0692c068959540fe1acd2a441cd254b74c51a052ca09cbfb64d2ca4d60`.
