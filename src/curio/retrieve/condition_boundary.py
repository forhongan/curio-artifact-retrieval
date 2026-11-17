"""Experimental condition evidence; inputs are query specs and image/text vectors only.

The finite dictionary is a disclosed structured-query bridge, not a natural-language
parser. Contrast alternatives need not be exclusive (motifs/colors may coexist).
Scores and sigmoid outputs are support, not calibrated correctness probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


# Each condition has a bilingual description followed by plausible replacements.
CONDITIONS = {
    "material/porcelain": [
        ("made of porcelain", "瓷质"),
        ("made of bronze", "青铜质"),
        ("made of glass", "玻璃质"),
    ],
    "material/bronze": [
        ("made of bronze", "青铜质"),
        ("made of porcelain", "瓷质"),
        ("made of jade", "玉质"),
    ],
    "material/glass": [
        ("made of glass", "玻璃质"),
        ("made of porcelain", "瓷质"),
        ("made of jade", "玉质"),
    ],
    "broad_material/textile": [
        ("a textile or garment", "织物或衣物"),
        ("a ceramic object", "陶瓷器物"),
        ("a metal object", "金属器物"),
    ],
    "broad_material/ceramic": [
        ("made of ceramic", "陶瓷质"),
        ("made of metal", "金属质"),
        ("made of glass", "玻璃质"),
    ],
    "form_family/bottle_vase": [
        ("a bottle or vase", "瓶或花瓶"),
        ("a bowl", "碗"),
        ("a plate or dish", "盘或碟"),
    ],
    "form_family/bowl": [
        ("a bowl or water bowl", "碗或水钵"),
        ("a bottle or vase", "瓶或花瓶"),
        ("a plate or dish", "盘或碟"),
    ],
    "form_family/plate_dish": [
        ("a plate dish or saucer", "盘或碟"),
        ("a bowl", "碗"),
        ("a bottle or vase", "瓶或花瓶"),
    ],
    "form_family/pouring_vessel": [
        ("a pouring vessel", "倾倒器"),
        ("a bowl", "碗"),
        ("a plate", "盘"),
    ],
    "form/jar": [("a jar", "罐"), ("a vase", "花瓶"), ("a bowl", "碗")],
    "vessel/yes": [("a vessel", "容器"), ("a sculpture", "雕塑"), ("a textile", "织物")],
    "tag/green": [
        ("with a green surface", "绿色表面"),
        ("with a red surface", "红色表面"),
        ("with a blue surface", "蓝色表面"),
    ],
    "tag/red": [
        ("with a red surface", "红色表面"),
        ("with a green surface", "绿色表面"),
        ("with a blue surface", "蓝色表面"),
    ],
    "tag/blue_white": [
        ("with blue and white decoration", "蓝白纹饰"),
        ("with a green surface", "绿色表面"),
        ("with red decoration", "红色纹饰"),
    ],
    "tag/long_neck": [("with a long narrow neck", "细长颈"), ("with a short wide neck", "粗短颈")],
    "tag/handles": [
        ("with visible handles", "有可见柄或耳"),
        ("with a smooth handleless body", "器身光滑无柄"),
    ],
    "tag/lidded": [
        ("with a lid covering its mouth", "有盖覆盖器口"),
        ("with an uncovered open mouth", "器口敞开无盖"),
    ],
    "tag/spout": [
        ("with a pouring spout", "有倾倒用的流"),
        ("with a plain round rim", "普通圆形口沿"),
    ],
    "tag/figures": [
        ("depicting human figures", "描绘人物"),
        ("depicting flowers", "描绘花卉"),
        ("depicting animals", "描绘动物"),
    ],
    "tag/floral": [
        ("with flower decoration", "花卉装饰"),
        ("with geometric decoration", "几何纹饰"),
        ("with human figures", "人物纹饰"),
    ],
    "tag/lotus_motif": [
        ("with lotus flowers or leaves", "荷花或荷叶纹"),
        ("with peony flowers", "牡丹纹"),
        ("with geometric decoration", "几何纹饰"),
    ],
    "tag/dragon": [
        ("depicting a dragon", "描绘龙"),
        ("depicting a bird", "描绘鸟"),
        ("with flower decoration", "花卉装饰"),
    ],
}

FEATURE_NAMES = [
    "cosine",
    "generic_margin",
    "gallery_zscore",
    "condition_mean",
    "condition_min",
    "contrast_mean",
    "contrast_min",
    "supported_fraction",
    "condition_count",
]


def prompt_plan(conditions, language):
    if language not in {"en", "zh"}:
        raise ValueError("language must be en or zh")
    index = 0 if language == "en" else 1
    keys = ["/".join(c) for c in conditions]
    supported = [key for key in keys if key in CONDITIONS]
    descriptions = [CONDITIONS[key][0][index] for key in supported]
    prefix = "a museum object: " if language == "en" else "一件博物馆文物："
    joiner = "; " if language == "en" else "；"
    full = prefix + joiner.join(descriptions)
    groups = []
    for i, key in enumerate(supported):
        alternatives = []
        for alternative in CONDITIONS[key][1:]:
            replaced = descriptions.copy()
            replaced[i] = alternative[index]
            alternatives.append(prefix + joiner.join(replaced))
        groups.append({"key": key, "atom": prefix + descriptions[i], "alternatives": alternatives})
    return {
        "positive": full,
        "groups": groups,
        "missing": [k for k in keys if k not in CONDITIONS],
        "condition_count": len(keys),
        "language": language,
    }


def plan_texts(plan):
    if not plan["groups"]:
        return []
    return [plan["positive"]] + [
        text for group in plan["groups"] for text in [group["atom"], *group["alternatives"]]
    ]


def condition_features(dense, generic, plan, prompt_scores):
    """Return N x 9 features. Unsupported conditions are explicit, never fabricated."""
    dense, generic = np.asarray(dense, dtype=float), np.asarray(generic, dtype=float)
    if dense.ndim != 1 or generic.shape != dense.shape or not len(dense):
        raise ValueError("Nonempty aligned gallery vectors required")
    if not np.isfinite(dense).all() or not np.isfinite(generic).all():
        raise ValueError("Scores must be finite")
    atoms, contrasts = [], []
    for group in plan["groups"]:
        atoms.append(prompt_scores[group["atom"]])
        contrasts.append(
            prompt_scores[plan["positive"]]
            - np.max(np.column_stack([prompt_scores[t] for t in group["alternatives"]]), axis=1)
        )
    if atoms:
        a, c = np.column_stack(atoms), np.column_stack(contrasts)
        aggregate = [a.mean(1), a.min(1), c.mean(1), c.min(1)]
    else:
        aggregate = [np.zeros(len(dense)) for _ in range(4)]
    count = plan["condition_count"]
    result = np.column_stack(
        [
            dense,
            dense - generic,
            (dense - dense.mean()) / max(dense.std(), 1e-6),
            *aggregate,
            np.full(len(dense), len(atoms) / max(count, 1)),
            np.full(len(dense), count),
        ]
    )
    if not np.isfinite(result).all():
        raise ValueError("Condition evidence must be finite")
    return result


def support_scores(features, model):
    """Portable inference for a fitted StandardScaler + logistic model."""
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(model["coefficients"]) or not np.isfinite(x).all():
        raise ValueError("Feature schema mismatch or non-finite evidence")
    return expit((x - model["mean"]) / model["scale"] @ model["coefficients"] + model["intercept"])
