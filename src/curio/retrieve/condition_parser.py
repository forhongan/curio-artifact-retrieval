"""Finite, conservative bilingual bridge for the condition boundary.

Only the query is parsed. Unknown text, negation and unsupported Boolean syntax
are retained in the raw encoder input and route to the trained raw-score fallback.
"""

import re

PARSER_VERSION = "condition-query-v2-colors"

# Order is stable and agrees with the structured experiment's supported clauses.
RULES = [
    (("material", "porcelain"), r"\bporcelain\b|(?<!陶)瓷(?:器|质)?"),
    (("material", "bronze"), r"\bbronze\b|青铜(?:器|质)?"),
    (("material", "glass"), r"\bglass\b|玻璃"),
    (("broad_material", "textile"), r"\b(?:textiles?|garments?)\b|织物|衣物"),
    (("broad_material", "ceramic"), r"\bceramics?\b|陶瓷(?:器)?"),
    (("form_family", "bottle_vase"), r"\b(?:bottles?|vases?)\b|花瓶|瓶子?|梅瓶|葫芦瓶"),
    (("form_family", "bowl"), r"\b(?:bowls?|waterpots?)\b|水钵|碗"),
    (("form_family", "plate_dish"), r"\b(?:plates?|dish(?:es)?|saucers?)\b|盘|碟"),
    (("form_family", "pouring_vessel"), r"\bpouring\s+vessels?\b|倾倒器"),
    (("form", "jar"), r"\bjars?\b|罐"),
    (("tag", "green"), r"\bgreen\b|绿色?"),
    (("tag", "red"), r"\bred\b|红色?"),
    (("tag", "blue_white"), r"\bblue\s+(?:and\s+)?white\b|蓝白(?:纹饰)?"),
    (("tag", "blue"), r"\bblue\b|蓝色?"),
    (("tag", "white"), r"\bwhite\b|白色?"),
    (("tag", "black"), r"\bblack\b|黑色?"),
    (("tag", "yellow"), r"\byellow\b|黄色?"),
    (("tag", "brown"), r"\bbrown\b|棕色?|褐色?"),
    (("tag", "pink"), r"\bpink\b|粉色"),
    (
        ("tag", "long_neck"),
        r"\b(?:long\s+)?narrow\s+neck(?:ed)?\b|\blong\s+neck(?:ed)?\b|细长颈|细颈",
    ),
    (("shape", "narrow_mouth"), r"\b(?:narrow|small)\s+mouth(?:ed)?\b|窄口|小口|细口"),
    (("tag", "handles"), r"\bhandles?\b|柄|耳|把手"),
    (("tag", "lidded"), r"\b(?:covered|lidded|lids?)\b|带盖|有盖|盖"),
    (("tag", "figures"), r"\bhuman\s+figures?\b|人物"),
    (("tag", "lotus_motif"), r"\blotus\s+(?:flowers?|leaves)\b|\blotus\b|荷花|荷叶|莲花|莲叶"),
    (("tag", "floral"), r"\b(?:flowers?|floral)\b|花卉"),
    (("tag", "dragon"), r"\bdragons?\b|龙(?:纹)?"),
    (("tag", "spout"), r"\bspouts?\b|流"),
    (("era", "Qianlong"), r"\bqianlong(?:\s+period)?\b|乾隆(?:时期)?"),
    (("era", "ming"), r"\bming(?:\s+dynasty)?\b|明代|明朝"),
    (("blue_white_technique", "yes"), r"青花"),
]


def parse_conditions(query):
    text = query.strip().lower().replace("-", " ")
    if not text:
        raise ValueError("query must contain non-whitespace text")
    language = "zh" if re.search(r"[\u3400-\u9fff]", text) else "en"
    reasons = []
    # Legacy v7 ignored these qualifiers to replay the frozen query plans.
    # Strict verification must expose them instead of certifying them silently.
    unverified_modifiers = []
    if re.search(r"\bopen\b|敞口", text):
        unverified_modifiers.append("shape/open_mouth")
    if re.search(r"\bround\b|圆腹", text):
        unverified_modifiers.append("shape/round_body")
    if re.search(
        r"\b(?:not|no|without|except|excluding|exclude|neither|nor|handleless)\b|不|无|無|没|沒|排除",
        text,
    ):
        reasons.append("negation")
    # These alternatives refer to a supported family/specific motif, not AND.
    boolean_text = re.sub(
        r"\bbottle\s+or\s+vase\b|\bplate\s+dish\s+or\s+saucer\b"
        r"|\bbowl\s+or\s+water\s+bowl\b|\blotus\s+flowers?\s+or\s+lotus\s+leaves\b"
        r"|瓶或花瓶|碗或水钵|盘或碟|荷花或荷叶|柄或耳",
        "",
        text,
    )
    if re.search(r"\bor\b|或|或者", boolean_text):
        reasons.append("boolean_alternative")
    # Preserve specialist names as unsupported, rather than treating a subtype as
    # just a generic vase. The broad rule can still be reported diagnostically.
    if re.search(r"梅瓶|葫芦瓶|\b(?:meiping|gourd)\b", text):
        reasons.append("specialist_form")
    remaining = text
    conditions = []
    # Protect reign names before matching 龙 inside 乾隆 (not the same character,
    # but historical phrases must remain one explicit missing condition).
    for condition, pattern in RULES:
        if re.search(pattern, remaining):
            conditions.append(list(condition))
            remaining = re.sub(pattern, " ", remaining)
    if not any(c[0] in {"form", "form_family"} for c in conditions):
        if re.search(r"\bvessels?\b|容器", remaining):
            conditions.insert(0, ["vessel", "yes"])
    # The existing evaluation's 'round covered jar' is annotated as jar + lid.
    # Do not erase roundness for other forms where it would be an extra clause.
    if ["form", "jar"] in conditions and ["tag", "lidded"] in conditions:
        remaining = re.sub(r"\bround\b|圆腹", " ", remaining)
    remaining = re.sub(
        r"\b(?:a|an|the|and|or|with|its|of|made|museum|objects?|artifacts?|items?|"
        r"vessels?|visible|surface|decoration|decorated|depicting|depicts|water|open|"
        r"period|dynasty)\b",
        " ",
        remaining,
    )
    for word in [
        "博物馆",
        "一件",
        "一个",
        "器物",
        "文物",
        "容器",
        "描绘",
        "装饰",
        "纹饰",
        "敞口",
        "时期",
        "带",
        "有",
        "的",
        "与",
        "和",
        "或",
        "饰",
        "纹",
    ]:
        remaining = remaining.replace(word, " ")
    if any(c[0] == "era" for c in conditions) and ["material", "porcelain"] in conditions:
        conditions = [
            ["blue_white_technique", "yes"] if c == ["tag", "blue_white"] else c for c in conditions
        ]
        conditions.sort(key=lambda c: c[0] == "blue_white_technique")
    remaining = re.sub(r"[\s,.;:，。；：、()（）]+", " ", remaining).strip()
    if remaining:
        reasons.append("unparsed_text")
    if len({c[1] for c in conditions if c[0] in {"form", "form_family"}}) > 1:
        reasons.append("multiple_forms")
    return {
        "parser": PARSER_VERSION,
        "language": language,
        "conditions": conditions,
        "fallback_reasons": reasons,
        "unparsed_text": remaining,
        "unverified_modifiers": unverified_modifiers,
    }
