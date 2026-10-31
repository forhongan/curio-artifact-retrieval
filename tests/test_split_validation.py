import pytest

from curio.represent.validation import validate_object_splits


def item(item_id, split, **kwargs):
    return {"item_id": item_id, "split": split, **kwargs}


def test_same_bytes_cannot_cross_splits():
    validate_object_splits([item("a", "train", sha256="x"), item("b", "train", sha256="x")])
    with pytest.raises(ValueError, match="sha256"):
        validate_object_splits([item("a", "train", sha256="x"), item("b", "test", sha256="x")])


def test_alias_relationship_cannot_cross_splits():
    with pytest.raises(ValueError, match="near duplicate"):
        validate_object_splits([item("a", "train"), item("b", "test", duplicate_sha256_of="a")])


def test_duplicate_id_rejected():
    with pytest.raises(ValueError, match="Duplicate item_id"):
        validate_object_splits([item("a", "train"), item("a", "test")])


@pytest.mark.parametrize("field", ["image_sha256", "object_group", "phash"])
def test_processed_bytes_and_explicit_object_groups_cannot_leak(field):
    with pytest.raises(ValueError, match=field):
        validate_object_splits(
            [
                item("a", "train", **{field: "same"}),
                item("b", "test", **{field: "same"}),
            ]
        )
    validate_object_splits(
        [
            item("a", "train", **{field: "same"}),
            item("b", "train", **{field: "same"}),
        ]
    )
