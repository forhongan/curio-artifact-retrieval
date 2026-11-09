"""Query-adaptive image sets with compositional, image-derived evidence.

The boundary is a visual support heuristic, not a calibrated correctness
probability. Candidate titles/labels are used only to render the accepted set.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image, ImageOps
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from curio.query import parse_query
from curio.retrieve.probe_sets import ProbeSetService
from curio.represent.visual_parts import load_visual_parts


def visual_boundary(scores, generic_scores, *, neighbors=None, posterior_floor=.90,
                    margin_floor=.01):
    """Separate query-specific support from generic museum-object similarity."""
    scores, generic_scores = np.asarray(scores), np.asarray(generic_scores)
    if scores.ndim != 1 or scores.shape != generic_scores.shape or not np.isfinite(scores).all() or not np.isfinite(generic_scores).all():
        raise ValueError('Expected parallel finite full-gallery score vectors')
    margin = scores - generic_scores
    posterior = np.zeros(len(scores))
    diagnostics = {'margin_floor': margin_floor,
                   'posterior_floor': posterior_floor, 'basis': 'query cosine minus generic museum-photo cosine'}
    if len(scores) < 4 or np.std(margin) < 1e-6:
        diagnostics.update(status='insufficient_distribution_evidence')
        return np.zeros(len(scores), dtype=bool), posterior, margin, diagnostics
    values = margin.reshape(-1, 1)
    with threadpool_limits(limits=1):
        one = GaussianMixture(1, random_state=5523, reg_covar=1e-6).fit(values)
        two = GaussianMixture(2, random_state=5523, n_init=3, reg_covar=1e-6).fit(values)
    high = int(two.means_.ravel().argmax())
    posterior = two.predict_proba(values)[:, high]
    gain = float(one.bic(values) - two.bic(values))
    separation = float(np.ptp(two.means_.ravel()) / np.sqrt(two.covariances_.ravel().mean()))
    modes_supported = gain >= 10 and separation >= 1.
    accepted = (posterior >= posterior_floor) & (margin >= margin_floor) & modes_supported
    diagnostics.update(status='boundary_found' if modes_supported else 'no_separable_support',
                       bic_gain=gain, standardized_separation=separation,
                       component_means=two.means_.ravel().tolist(),
                       component_weights=two.weights_.tolist(), seed_count=int(accepted.sum()))
    # A second, local split separates an object's broad visual family from
    # stronger full-description evidence. It returns a variable-size cluster.
    local_indices = np.flatnonzero(accepted)
    if len(local_indices) >= 12:
        local_values = scores[local_indices, None]
        with threadpool_limits(limits=1):
            local_one = GaussianMixture(1, random_state=5523, reg_covar=1e-6).fit(local_values)
            local_two = GaussianMixture(2, random_state=5523, n_init=3, reg_covar=1e-6).fit(local_values)
        local_gain = float(local_one.bic(local_values) - local_two.bic(local_values))
        local_separation = float(np.ptp(local_two.means_.ravel()) / np.sqrt(local_two.covariances_.ravel().mean()))
        if local_gain >= 10 and local_separation >= 1.:
            local_high = int(local_two.means_.ravel().argmax())
            local_posterior = local_two.predict_proba(local_values)[:, local_high]
            accepted[local_indices] &= local_posterior >= posterior_floor
            posterior[local_indices] = np.minimum(posterior[local_indices], local_posterior)
        diagnostics.update(local_bic_gain=local_gain, local_separation=local_separation,
                           local_refined_count=int(accepted.sum()))
    # This model-specific weak-signal guard is query-level, not a universal
    # per-image probability. It catches OOD color-only matches in this index.
    if scores.max(initial=-1.) < .05:
        accepted[:] = False
        diagnostics.update(status='weak_absolute_support')
    diagnostics['query_peak_cosine_floor'] = .05
    if neighbors is not None:
        neighbor_support = accepted[np.asarray(neighbors)].mean(axis=1)
        accepted &= neighbor_support >= .25
        diagnostics.update(neighbor_support_floor=.25, graph_retained=int(accepted.sum()))
    return accepted, posterior, margin, diagnostics


class VisualSetService:
    """Compare a free-text query with every indexed image and return all yeses."""

    def __init__(self, search_service, *, dino_dir=None):
        self.service = search_service
        self.dino_dir = Path(dino_dir or search_service.root / 'artifacts/dinov2')
        self._neighbors = None
        self._generic = None
        self._heads = None
        self._parts = None
        self._aux_loaded = False
        self._text_scores = {}
        self._frame_colors = {}

    def _frame_color_presence(self, index, color):
        """Weak fallback: absence of color anywhere also excludes foreground color.

        Presence in the full frame cannot localize color to the object. This
        fallback only rejects visible contradictions when the mask abstains.
        """
        if index not in self._frame_colors:
            with Image.open(self.service.root / self.service.items[index]['image_path']) as opened:
                image = ImageOps.exif_transpose(opened).convert('RGB')
                image.thumbnail((256, 256))
                hsv = np.asarray(image.convert('HSV'), dtype=float) / 255
            h, s, v = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
            chromatic = (s >= .12) & (v >= .12)
            signals = {
                'green': chromatic & (h >= 70) & (h <= 190),
                'blue': chromatic & (h >= 180) & (h <= 280),
                'red': chromatic & ((h <= 25) | (h >= 335)),
                'yellow': chromatic & (h >= 35) & (h <= 75),
                'purple': chromatic & (h >= 270) & (h <= 335),
                'orange': chromatic & (h >= 15) & (h <= 45),
                'white': (s <= .18) & (v >= .70), 'black': v <= .22,
            }
            self._frame_colors[index] = {name: float(mask.mean()) for name, mask in signals.items()}
        return self._frame_colors[index][color]

    def _score_prompt(self, prompt):
        if prompt not in self._text_scores:
            self._text_scores[prompt] = self.service._dense(prompt)
        return self._text_scores[prompt]

    def _compositional_evidence(self, parsed):
        """Compare visible clauses against plausible visual alternatives.

        These are text prototypes scored against pixels, not gallery labels.
        Equal support abstains from admitting an image into a confident set.
        A winning prototype is still not a calibrated correctness guarantee.
        """
        count = len(self.service.items)
        keep = np.ones(count, dtype=bool)
        evidence = []
        forms = parsed.slots.get('object_type', [])
        materials = parsed.slots.get('material', [])
        context = ' '.join([*materials, *forms]) or 'museum object'
        motifs = {
            'lotus': 'lotus flowers and lotus leaves',
            'dragon': 'a dragon', 'phoenix': 'a phoenix',
            'bird': 'birds', 'fish': 'fish', 'floral': 'flowers',
            'landscape': 'a landscape scene', 'geometric': 'geometric patterns',
            'figure_scene': 'human figures', 'cloud': 'clouds',
            'horse': 'horses', 'lion': 'lions', 'tiger': 'tigers',
            'butterfly': 'butterflies', 'leaf': 'leaves',
        }
        requested = set(parsed.slots.get('motifs', []))
        # "Lotus flowers" is one specific constraint, not two independent ones.
        if 'lotus' in requested:
            requested.discard('floral')
        alternatives = {**motifs, 'bamboo': 'bamboo', 'peony': 'peony flowers',
                        'chrysanthemum': 'chrysanthemum flowers', 'plain': 'no decoration'}
        for motif in sorted(requested & motifs.keys()):
            positive = self._score_prompt(f'a {context} decorated with {motifs[motif]}')
            competitors = [self._score_prompt(f'a {context} decorated with {description}')
                           for name, description in alternatives.items()
                           if name not in requested and not (motif == 'floral' and name in {'lotus', 'peony', 'chrysanthemum'})]
            # Generic flowers are not contradictory to a specific flower.
            if motif == 'lotus':
                competitors = [self._score_prompt(f'a {context} decorated with {description}')
                               for name, description in alternatives.items()
                               if name not in requested | {'floral', 'leaf'}]
            gap = positive - np.max(competitors, axis=0)
            keep &= gap > 0
            evidence.append((f'visual_text_contrast/motif/{motif}', gap, np.ones(count, bool), 0.))
        # Blue-and-white is a decorative appearance, not merely two colors
        # somewhere in the photograph (e.g. a blue-ground polychrome vessel).
        if re.search(r'青花|blue[\s-]*(?:and[\s-]*)?white', parsed.positive_text, re.I):
            positive = self._score_prompt(f'a blue and white {context} with cobalt blue decoration on a white surface')
            negatives = [self._score_prompt(f'a {context} {description}') for description in (
                'with multicolored enamel decoration on a blue ground',
                'with a plain solid blue glaze', 'with a plain white glaze',
                'with a monochrome green celadon glaze')]
            gap = positive - np.max(negatives, axis=0)
            keep &= gap > 0
            evidence.append(('visual_text_contrast/style/blue_white', gap, np.ones(count, bool), 0.))
        if 'narrow_mouth' in parsed.slots.get('shape', []):
            positive = self._score_prompt(f'a narrow-necked {context} with a small mouth')
            negatives = [self._score_prompt(f'a {context} {description}') for description in (
                'with a wide mouth and no narrow neck', 'with a wide flaring trumpet mouth')]
            gap = positive - np.max(negatives, axis=0)
            keep &= gap > 0
            evidence.append(('visual_text_contrast/shape/narrow_mouth', gap, np.ones(count, bool), 0.))
        if re.search(r'embroider|刺绣|刺繡', parsed.positive_text, re.I):
            positive = self._score_prompt('a textile with embroidered flowers, visible embroidery stitches')
            negatives = [self._score_prompt(text) for text in (
                'a woven textile with floral patterns', 'a knotted carpet with floral patterns',
                'a printed textile with floral patterns', 'a painted panel with floral patterns')]
            gap = positive - np.max(negatives, axis=0)
            keep &= gap > 0
            evidence.append(('visual_text_contrast/technique/embroidery', gap, np.ones(count, bool), 0.))
        return keep, evidence

    def _load_auxiliary(self):
        if self._aux_loaded:
            return
        root = self.service.root
        heads = root / 'artifacts/catalogue_set_runtime_v5'
        index = root / 'artifacts/siglip'
        if (heads / 'set_policy.json').exists() and (index / 'image_embeddings.npy').exists():
            ids = json.loads((index / 'item_ids.json').read_text(encoding='utf-8'))
            if set(ids) != set(self.service.by_id):
                raise ValueError('Auxiliary visual heads have a different gallery')
            vectors = np.load(index / 'image_embeddings.npy', allow_pickle=False)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            lookup = {item: i for i, item in enumerate(ids)}
            vectors = vectors[[lookup[str(item['item_id'])] for item in self.service.items]]
            self._heads = ProbeSetService(self.service.items, vectors, probes_path=heads,
                                         set_report_path=heads / 'set_policy.json',
                                         index_sha256=hashlib.sha256((index / 'image_embeddings.npy').read_bytes()).hexdigest())
        directory = root / 'artifacts/visual_parts'
        if (directory / 'manifest.json').exists():
            parts = load_visual_parts(directory)
            if set(parts['item_ids']) != set(self.service.by_id):
                raise ValueError('Auxiliary pixel descriptors have a different gallery')
            lookup = {item: i for i, item in enumerate(parts['item_ids'])}
            order = [lookup[str(item['item_id'])] for item in self.service.items]
            self._parts = {key: values[order] for key, values in parts['features'].items()}
        self._aux_loaded = True

    def _auxiliary_evidence(self, parsed, selected):
        self._load_auxiliary()
        count = len(self.service.items)
        keep = np.ones(count, dtype=bool)
        evidence = []
        # Coarse form/material heads supply contradiction evidence only. Their
        # stored outputs are predictions from image vectors, never item tags.
        if self._heads is not None:
            requested = [('object_type', value) for value in parsed.slots.get('object_type', [])]
            requested += [(self._heads._material_task(value), value) for value in parsed.slots.get('material', [])]
            for task, value in requested:
                classes = self._heads._classes.get(task, [])
                compatible = {'vase': ['vase', 'bottle'], 'bottle': ['vase', 'bottle'],
                              'plate': ['plate', 'dish'], 'dish': ['plate', 'dish']}.get(value, [value])
                columns = [classes.index(candidate) for candidate in compatible if candidate in classes]
                if not columns:
                    continue
                probability = self._heads._probabilities[task][:, columns].sum(axis=1)
                keep &= probability >= .12
                evidence.append((f'visual_head/{task}/{value}', probability, np.ones(count, dtype=bool), .12))
        if self._parts is not None:
            colors = self._parts['color'].reshape(count, 12, 4, 4) ** 2
            hue_ranges = {'red': [0, 11], 'green': [3, 4, 5], 'blue': [6, 7, 8],
                          'yellow': [1, 2], 'purple': [9, 10], 'orange': [0, 1]}
            valid = self._parts['valid_color'].astype(bool)
            for color in parsed.slots.get('colors', []):
                if color in hue_ranges:
                    fraction = colors[:, hue_ranges[color], 1:, 1:].sum(axis=(1, 2, 3))
                    floor = .025 if color == 'blue' else .06
                elif color == 'white':
                    fraction = colors[:, :, 0, 3].sum(axis=1)
                    floor = .08
                elif color == 'black':
                    fraction = colors[:, :, :, 0].sum(axis=(1, 2))
                    floor = .06
                else:
                    continue
                keep &= ~valid | (fraction >= floor)
                evidence.append((f'foreground_HSV/{color}', fraction, valid, floor))
                # An invalid foreground mask must not let a grayscale photograph
                # silently satisfy an explicitly chromatic query.
                fallback_valid = selected & ~valid
                frame_fraction = np.zeros(count)
                for i in np.flatnonzero(fallback_valid):
                    frame_fraction[i] = self._frame_color_presence(int(i), color)
                keep &= ~fallback_valid | (frame_fraction >= .003)
                evidence.append((f'full_frame_color_presence/{color}', frame_fraction, fallback_valid, .003))
            if 'narrow_mouth' in parsed.slots.get('shape', []):
                profile = self._parts['shape'][:, :32]
                ratio = np.median(profile[:, 2:6], axis=1) / np.maximum(profile[:, 6:22].max(axis=1), 1e-6)
                narrow_support = 1 - np.clip(ratio, 0, 1)
                valid = self._parts['valid_shape'].astype(bool)
                keep &= ~valid | (narrow_support >= .25)
                evidence.append(('foreground_silhouette/narrow_mouth', narrow_support, valid, .25))
        compositional_keep, compositional = self._compositional_evidence(parsed)
        keep &= compositional_keep
        evidence.extend(compositional)
        return keep, evidence

    def _graph(self):
        if self._neighbors is None:
            ids = json.loads((self.dino_dir / 'item_ids.json').read_text(encoding='utf-8'))
            vectors = np.load(self.dino_dir / 'image_embeddings.npy', allow_pickle=False)
            if set(ids) != set(self.service.by_id) or len(set(ids)) != len(ids):
                raise ValueError('DINO graph must cover the full active gallery')
            lookup = {item: i for i, item in enumerate(ids)}
            vectors = vectors[[lookup[str(item['item_id'])] for item in self.service.items]]
            with threadpool_limits(limits=1):
                model = NearestNeighbors(n_neighbors=min(21, len(ids)), metric='cosine', n_jobs=1).fit(vectors)
                indices = model.kneighbors(vectors, return_distance=False)
            self._neighbors = np.asarray([[j for j in row if j != i][:20]
                                          for i, row in enumerate(indices)], dtype=int)
        return self._neighbors

    def search(self, query, *, top_k=20, method='adaptive_evidence'):
        development_method=method in {'dev_calibrated','dev_cutoff','dev_color'}
        if not development_method and method not in {'adaptive_margin', 'adaptive_graph', 'adaptive_evidence'}:
            raise ValueError('Unknown visual-set method')
        started = time.perf_counter()
        if self.service.embeddings is None or len(self.service.vector_rows) != len(self.service.items):
            raise RuntimeError('A complete image index is required for whole-gallery set decisions')
        parsed = parse_query(query)
        if not query.strip():
            raise ValueError('query must contain non-whitespace text')
        # Keep negation in the actual encoder input; the parser's positive-only
        # rewrite is inappropriate for this metadata-free decision path.
        encoded_query = (parsed.rewritten if parsed.language == 'zh' and not parsed.negations
                         else query.strip())
        if development_method:
            encoded_query=query.strip()
        scores = self.service._dense(encoded_query)
        if self._generic is None:
            self._generic = self.service._dense('a photograph of a museum object')
        graph = self._graph() if method == 'adaptive_graph' else None
        if development_method:
            from curio.retrieve.calibrated_sets import decide
            yes,posterior,margin,diagnostics,scores=decide(self.service,query,method,self._generic)
        else:
            yes, posterior, margin, diagnostics = visual_boundary(scores, self._generic, neighbors=graph)
        evidence = []
        if method == 'adaptive_evidence':
            aux_keep, evidence = self._auxiliary_evidence(parsed, yes)
            before = int(yes.sum())
            yes &= aux_keep
            diagnostics.update(before_auxiliary=before, after_auxiliary=int(yes.sum()),
                               auxiliary_signals=[name for name, _, _, _ in evidence],
                               auxiliary_policy='require compositional visual support; reject image-derived contradictions; missing pixel descriptors abstain')
        # All matched records are returned. top_k affects only the optional preview.
        accepted = sorted(np.flatnonzero(yes), key=lambda i: (-float(margin[i]), str(self.service.items[i]['item_id'])))
        results = []
        for i in accepted:
            item = self.service.items[int(i)]
            item_id = str(item['item_id'])
            results.append({'item_id': item_id, 'title': item.get('title', ''), 'yes': True,
                            'rank':len(results)+1,'score':float(scores[i]),
                            'image_url': f'/api/images/{item_id}', 'thumb_url': f'/api/images/{item_id}?thumb=true',
                            'cosine': round(float(scores[i]), 6), 'specificity_margin': round(float(margin[i]), 6),
                            'support_mode_posterior': round(float(posterior[i]), 6),
                            'auxiliary_evidence': [
                                {'signal': name, 'status': 'supported' if valid[i] else 'unknown',
                                 'value': round(float(values[i]), 6) if valid[i] else None, 'floor': floor}
                                for name, values, valid, floor in evidence]})
        return {
            'query': query, 'encoded_query': encoded_query, 'variant': 'visual_yes_set',
            'method': method, 'retrieval_mode': 'image_embeddings', 'gallery_count': len(scores),
            'matched_count': len(results), 'rejected_count': len(scores) - len(results),
            'matched_item_ids': [row['item_id'] for row in results], 'results': results,
            'preview': results[:top_k], 'returned': len(results), 'no_match': not bool(results),
            'boundary': diagnostics, 'latency_ms': round((time.perf_counter() - started) * 1000, 2),
            'warnings': [
                'Boundary and support-mode posterior are visual heuristics, not verified match probabilities.',
                'An empty set means insufficient visual support, not proof that the gallery has no relevant image.',
                'Historical date, hidden structure, counts and negation are not certified by this visual boundary.',
            ],
            'provenance': {'index_sha256': self.service.index_sha256,
                           'candidate_inputs': (['image embeddings', 'image-derived attribute predictions',
                                                 'foreground pixel descriptors', 'contrastive visual-text scores']
                                                if method == 'adaptive_evidence' else ['image embeddings']),
                           'candidate_catalogue_used_for_decisions': False},
        }
