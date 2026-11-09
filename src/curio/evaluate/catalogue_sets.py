"""Catalogue-supervised predicates on pixels, with dev-frozen set boundaries.

Missing supervision is unknown. Inference always scores the entire split;
ground-truth eligibility is applied only when computing evaluation metrics.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from curio.evaluate.catalogue_boundary import explicit_metrics, query_dataset, rank_percentile
from curio.evaluate.catalogue_tasks import TASKS, single_label


def linear_probabilities(vectors, weights, bias, class_count):
    logits = np.asarray(vectors) @ np.asarray(weights).T + np.asarray(bias)
    if class_count == 2 and logits.shape[1] == 1:
        positive = expit(logits[:, 0])
        return np.column_stack((1 - positive, positive))
    return softmax(logits, axis=1)


def select_binary_threshold(truth, probabilities):
    choices = []
    for threshold in np.linspace(.05, .95, 37):
        selected = probabilities >= threshold
        tp = int((selected & truth).sum())
        fp = int((selected & ~truth).sum())
        fn = int((~selected & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        choices.append((f1, precision, float(threshold)))
    return max(choices)[-1]


def fit_catalogue_heads(rows, vectors, *, min_train=20, fixed_c=None, fixed_threshold=None):
    heads, reports = {}, {}
    splits = np.asarray([row['split'] for row in rows])
    for domain in TASKS:
        labels = np.asarray([single_label(row, domain) for row in rows], dtype=object)
        known = labels != None  # noqa: E711
        train = (splits == 'train') & known
        classes = sorted(value for value in set(labels[train])
                         if np.sum(labels[train] == value) >= min_train)
        train &= np.isin(labels, classes)
        dev = (splits == 'dev') & np.isin(labels, classes)
        if len(classes) < 2 or (fixed_c is None and not dev.any()):
            reports[domain] = {'status': 'insufficient_support', 'classes': classes}
            continue
        choices = []
        for c in ((fixed_c,) if fixed_c is not None else (.1, 1., 10.)):
            model = LogisticRegression(C=c, class_weight='balanced', max_iter=2000,
                                       random_state=0)
            model.fit(vectors[train], labels[train])
            score = (f1_score(labels[dev], model.predict(vectors[dev]), average='macro',
                              zero_division=0) if fixed_c is None else 0.0)
            choices.append((score, -c, model))
        score, neg_c, model = max(choices, key=lambda choice: choice[:2])
        probabilities = model.predict_proba(vectors)
        calibration = (splits == 'dev') & known
        thresholds = {}
        for j, value in enumerate(model.classes_):
            truth = labels[calibration] == value
            if fixed_threshold is not None:
                thresholds[str(value)] = float(fixed_threshold)
            elif truth.any():
                thresholds[str(value)] = select_binary_threshold(truth, probabilities[calibration, j])
        heads[domain] = {
            'classes': list(model.classes_), 'weights': model.coef_, 'bias': model.intercept_,
            'thresholds': thresholds, 'probabilities': probabilities,
        }
        reports[domain] = {
            'status': 'trained', 'classes': list(model.classes_), 'selected_C': -neg_c,
            'dev_macro_f1': float(score) if fixed_c is None else None, 'thresholds': thresholds,
            'threshold_policy': 'fixed engineering boundary' if fixed_threshold is not None else 'dev-selected binary F1',
            'train_count': int(train.sum()), 'dev_count': int(dev.sum()),
            'threshold_calibration_count': int(calibration.sum()),
        }
    return heads, reports


def atoms_supported(spec, heads):
    return all(domain in heads and value in heads[domain]['thresholds']
               for domain, value in spec.atoms)


def attribute_scores(heads, indices, atoms):
    columns, thresholds = [], []
    for domain, value in atoms:
        head = heads[domain]
        columns.append(head['probabilities'][indices, head['classes'].index(value)])
        thresholds.append(head['thresholds'][value])
    matrix = np.column_stack(columns)
    return {
        'attribute_min': matrix.min(axis=1),
        'attribute_mean': matrix.mean(axis=1),
        'attribute_product': matrix.prod(axis=1),
        'calibrated_min': (matrix / np.asarray(thresholds)).min(axis=1),
    }


def prepare_queries(rows, vectors, heads, text_vectors, specs, split):
    indices = np.asarray([i for i, row in enumerate(rows) if row['split'] == split])
    ids = [rows[int(i)]['item_id'] for i in indices]
    output = []
    for spec in specs:
        truth = query_dataset(rows, spec, split)
        dense = vectors[indices] @ text_vectors[spec.qid]
        scores = attribute_scores(heads, indices, spec.atoms)
        scores.update(dense=dense, dense_percentile=rank_percentile(dense))
        output.append({'spec': spec, 'ids': ids, 'truth': truth, 'scores': scores})
    return output


def score_query_sets(prepared, method, threshold, *, include_ids=False):
    output = []
    for entry in prepared:
        scores = entry['scores'][method]
        full_selected = {item for item, score in zip(entry['ids'], scores) if score >= threshold}
        eligible = set(entry['truth']['ids'])
        selected = full_selected & eligible
        relevant = entry['truth']['relevant']
        row = {
            'qid': entry['spec'].qid, 'family': entry['spec'].family,
            **explicit_metrics(selected, relevant),
            'unknown_excluded': entry['truth']['unknown_excluded'],
            'unknown_selected': len(full_selected - eligible),
            'full_selected_size': len(full_selected),
            'full_gallery_size': len(entry['ids']),
        }
        if include_ids:
            row.update(selected_ids=sorted(selected), full_selected_ids=sorted(full_selected),
                       tp_ids=sorted(selected & relevant), fp_ids=sorted(selected - relevant),
                       fn_ids=sorted(relevant - selected))
        output.append(row)
    return output


def threshold_candidates(prepared, method):
    values = np.concatenate([entry['scores'][method] for entry in prepared])
    # Both empty-all and accept-all are legitimate development controls.
    quantiles = np.quantile(values, np.linspace(0, 1, 81))
    candidates = [*quantiles, float(values.max()) + 1e-6]
    if method == 'calibrated_min':
        candidates.append(1.0)
    return sorted(set(float(value) for value in candidates))
