"""Fixed classical pixel descriptors; no text model or catalog metadata."""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern

VERSION = 'classical-cv-v1'
PARAMETERS = {
    'max_side': 192, 'feature_side': 128, 'grabcut_iterations': 3,
    'grabcut_border_fraction': 0.05, 'foreground_area_range': [0.03, 0.90],
    'hog_orientations': 9, 'hog_cell': 16, 'hog_block_cells': 2,
    'color_bins': [12, 4, 4], 'spatial_pyramid': [1, 2],
    'lbp_radius_points': [[1, 8], [2, 16]],
}


def _normalize(a):
    a = np.asarray(a, dtype=np.float32).ravel()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def _padded(rgb, mask, size=128):
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    width, height = max(1, round(w * scale)), max(1, round(h * scale))
    image = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    roi = cv2.resize(mask.astype('uint8'), (width, height), interpolation=cv2.INTER_NEAREST)
    top, left = (size - height) // 2, (size - width) // 2
    canvas = np.full((size, size, 3), 127, dtype='uint8')
    valid = np.zeros((size, size), dtype=bool)
    canvas[top:top+height, left:left+width] = image
    valid[top:top+height, left:left+width] = roi.astype(bool)
    return canvas, valid


def _histograms(values, valid, bins, ranges):
    out = []
    h, w = valid.shape
    for level in (1, 2):
        for iy in range(level):
            for ix in range(level):
                mask = valid[iy*h//level:(iy+1)*h//level, ix*w//level:(ix+1)*w//level]
                patch = values[iy*h//level:(iy+1)*h//level, ix*w//level:(ix+1)*w//level]
                pixels = patch[mask]
                if len(pixels):
                    hist, _ = np.histogramdd(pixels.reshape(-1, len(bins)), bins=bins, range=ranges)
                    hist /= hist.sum()
                    out.append(np.sqrt(hist).ravel())
                else:
                    out.append(np.zeros(np.prod(bins)))
    return _normalize(np.concatenate(out))


def _descriptors(rgb, mask):
    canvas, roi = _padded(rgb, mask)
    # Neutralize outside the region to avoid image-background color leaking into masked HOG.
    masked = canvas.copy()
    masked[~roi] = 127
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
    edge = _normalize(hog(gray, orientations=9, pixels_per_cell=(16, 16),
                          cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True))
    hsv = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV)
    color = _histograms(hsv, roi, [12, 4, 4], [(0, 180), (0, 256), (0, 256)])
    textures = []
    for radius, points in ((1, 8), (2, 16)):
        lbp = local_binary_pattern(gray, points, radius, method='uniform')
        textures.append(_histograms(lbp[..., None], roi, [points + 2], [(0, points + 2)]))
    texture = _normalize(np.concatenate(textures))
    return edge, color, texture


def _shape(mask):
    _, roi = _padded(np.zeros((*mask.shape, 3), dtype='uint8'), mask)
    ys, xs = np.where(roi)
    if len(xs) == 0:
        return np.zeros(74, dtype='float32')
    patch = roi[ys.min():ys.max()+1, xs.min():xs.max()+1].astype('uint8')
    patch = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_NEAREST)
    moments = cv2.HuMoments(cv2.moments(patch)).ravel()
    hu = -np.sign(moments) * np.log10(np.abs(moments) + 1e-30)
    h, w = roi.shape
    return _normalize(np.r_[patch.mean(axis=0), patch.mean(axis=1), hu,
                            (xs.max()-xs.min()+1)/w, (ys.max()-ys.min()+1)/h, roi.mean()])


def extract_classical(image):
    """Return fixed feature vectors, GrabCut diagnostics, and the binary mask.

    For rejected GrabCut masks gc descriptors are raw fallbacks and shape is
    zero; the diagnostic flag must be consulted before interpreting the branch.
    """
    if isinstance(image, (str, bytes)) or hasattr(image, '__fspath__'):
        with Image.open(image) as opened:
            im = ImageOps.exif_transpose(opened).convert('RGB')
    else:
        im = ImageOps.exif_transpose(image).convert('RGB')
    im.thumbnail((192, 192), Image.Resampling.LANCZOS)
    rgb = np.asarray(im)
    h, w = rgb.shape[:2]
    raw_mask = np.ones((h, w), dtype=bool)
    raw_hog, raw_color, raw_texture = _descriptors(rgb, raw_mask)
    reason = None
    gc_mask = np.zeros((h, w), dtype=np.uint8)
    if min(h, w) < 12:
        reason = 'image_too_small'
    else:
        margin_x, margin_y = max(1, int(w*.05)), max(1, int(h*.05))
        seed = int.from_bytes(hashlib.sha256(rgb.tobytes()).digest()[:4], 'little') & 0x7fffffff
        cv2.setRNGSeed(seed)
        try:
            cv2.grabCut(rgb, gc_mask, (margin_x, margin_y, w-2*margin_x, h-2*margin_y),
                        None, None, 3, cv2.GC_INIT_WITH_RECT)
            gc_mask = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype('uint8')
            area = float(gc_mask.mean())
            if not 0.03 <= area <= 0.90:
                reason = 'foreground_area_out_of_range'
        except cv2.error:
            reason = 'grabcut_failed'
    accepted = reason is None
    if accepted:
        gc_hog, gc_color, gc_texture = _descriptors(rgb, gc_mask.astype(bool))
        gc_shape = _shape(gc_mask.astype(bool))
    else:
        gc_hog, gc_color, gc_texture = raw_hog.copy(), raw_color.copy(), raw_texture.copy()
        gc_shape = np.zeros(74, dtype='float32')
    features = {'raw_hog': raw_hog, 'raw_color': raw_color, 'raw_texture': raw_texture,
                'gc_hog': gc_hog, 'gc_color': gc_color, 'gc_texture': gc_texture,
                'gc_shape': gc_shape}
    features['raw_all'] = _normalize(np.concatenate((raw_hog, raw_color, raw_texture)))
    features['gc_all'] = _normalize(np.concatenate((gc_hog, gc_color, gc_texture, gc_shape)))
    return {'features': features, 'mask': gc_mask,
            'diagnostics': {'grabcut_accepted': accepted, 'fallback': not accepted,
                            'reason': reason, 'foreground_fraction': float(gc_mask.mean()),
                            'width': w, 'height': h}}
