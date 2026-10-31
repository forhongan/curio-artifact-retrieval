"""Split integrity checks shared by training and held-out evaluation."""

from collections import defaultdict


def validate_object_splits(items):
    ids = [str(x["item_id"]) for x in items]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate item_id in dataset")
    for field in (
        "image_path",
        "sha256",
        "image_sha256",
        "object_group",
        "duplicate_group",
        "phash_group",
        "phash",
    ):
        assignments = defaultdict(set)
        for item in items:
            value = item.get(field)
            if value:
                assignments[str(value)].add(item["split"])
        leaking = [key for key, splits in assignments.items() if len(splits) > 1]
        if leaking:
            raise ValueError(f"Cross-split leakage in {field}: {leaking[:3]}")
    by_id = {x["item_id"]: x for x in items}
    for item in items:
        for field in ("duplicate_sha256_of", "near_duplicate_of"):
            linked = by_id.get(item.get(field))
            if linked and linked["split"] != item["split"]:
                raise ValueError(
                    f"Cross-split near duplicate: {item['item_id']} and {linked['item_id']}"
                )
