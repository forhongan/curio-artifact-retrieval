"""Color observability, not color identity, for photo-based membership."""

import numpy as np
from PIL import Image, ImageOps

VERSION = "photo-color-observability-v1"


def color_observability(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=float)
    spread = np.ptp(rgb, axis=2)
    fraction = float((spread > 8).mean())
    # Near-monochrome includes genuine neutral objects with neutral backgrounds:
    # the image alone cannot establish their original surface color either.
    return {
        "version": VERSION,
        "observable": fraction >= 0.01,
        "chromatic_pixel_fraction": fraction,
        "channel_spread_p99": float(np.quantile(spread, 0.99)),
    }
