# Reproducible image specialists

Run from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe scripts/build_visual_parts.py --workers 8
.venv/Scripts/python.exe -m pytest tests/test_visual_parts.py -q
```

The default builder reads every row of `data/snapshots/collection_v2.jsonl`
(the frozen 4,692-item catalog), verifies each processed image against its frozen
`image_sha256`, and writes derived outputs to `artifacts/visual_parts`. It does
not modify source images, train on gallery images, or inspect catalog labels.

This implementation is **classical heuristic segmentation**, not SAM, DINO,
learned segmentation, or semantic material/motif recognition. It fits a robust
planar RGB background to the image border, thresholds color residuals, closes
one-pixel gaps, and retains the largest connected foreground component. The
resized EXIF-oriented image has a maximum side of 256 pixels. Masks preserve
internal background holes; they are not filled solid.

The method abstains when borders are inconsistent, the estimated object occupies
too little or too much of the frame, foreground touches the frame, or multiple
substantial disconnected foreground regions appear. It also abstains on large
enclosed missing regions (over 12% of the filled mask), which can indicate a white
object body confused with the background; this conservatively excludes some real
openwork objects and large handles as well. Each descriptor has its own
validity flag. Rejected descriptors are zero vectors. Candidate masks are retained
even when rejected so failures can be inspected. Validity means the heuristic
passed its checks; it is not calibrated segmentation accuracy. A complex scene
with a clean border can still pass, and low-contrast objects, pedestals, shadows,
transparent objects and cropped works remain difficult.

| Feature | Dimensions | Construction |
| --- | ---: | --- |
| shape | 132 | Tight foreground bounding box, normalized 32-bin row/column occupancy and centroid profiles, aspect ratio, extent and axis spreads |
| color | 192 | Eroded foreground HSV histogram with 12 hue, 4 saturation and 4 value bins; square-root probability map |
| texture | 34 | Foreground interior rotation-invariant uniform local binary pattern histogram (10), gradient orientation (16), gradient magnitude (8) |

All valid vectors are L2 normalized. Shape normalization tolerates translation and
scale, but preserves orientation and aspect ratio. Texture describes local image
patterns and gradients, including photographic lighting and resolution effects;
it cannot establish that a pattern is a dragon, inscription or particular material.
Color is photographic appearance and is not illumination invariant.

`features.npz` contains `shape`, `color`, `texture` float32 arrays and Boolean
`valid_shape`, `valid_color`, `valid_texture` arrays. `item_ids.json` is their exact
row order. `diagnostics.jsonl` records every image hash, mask hash, mask location,
rejection reasons, threshold and quality diagnostics. `manifest.json` records
the catalog hash, ordered input-image manifest hash, output hashes, feature
definitions, segmentation parameters, version and validity counts. Individual
mask hashes are in diagnostics. Load through `load_visual_parts()` to verify the
core artifact hashes before scoring. `mask_qa_accepted.jpg` and
`mask_qa_unknown.jpg` show deterministic evenly spaced examples and their masks.

For a query image, call `extract_visual_parts(image_or_path)` from
`curio.represent.visual_parts`. It returns descriptor keys, a `features` mapping,
a `valid` mapping, uint8 `mask`, and `diagnostics`. A feature similarity may only
be used when **both** query and gallery feature flags are true. Missing features
must be omitted and surviving weights renormalized. Never interpret an invalid
zero vector as a measured dissimilarity. The descriptors enable actual
query-image-to-gallery scoring; any text-to-feature bridge is a separate model
and must be evaluated independently.

## Completed frozen-catalog run (October 30, 2026)

The builder processed all **4,692** frozen images and verified every processed
image hash. It produced valid shape/color/texture rows for **888** images (18.9%);
the other **3,804** rows abstain and contain zero vectors. This low coverage is
an intentional consequence of the studio-background checks, not a segmentation
accuracy estimate. The run took 67 seconds with eight workers on this machine.

Both 48-example QA sheets were visually inspected. Many vase outlines are useful;
the rejected set includes scene/background leakage and also some usable masks.
The accepted set still contains errors: the transparent lower body of
`met_47334` is omitted, the dark figure `cma_111433` is incomplete, and several
objects retain cast shadows (for example `met_46727`). The validity flag therefore
records passing heuristic checks only. No manually labeled mask accuracy is
claimed. A trained foreground model or human validation remains necessary before
treating these masks as reliable object delineations.

Current descriptor artifact index hash:
`6681902de488af6d5506830346011ac2964646300dcd9116a2ecf26450b8c676`.
