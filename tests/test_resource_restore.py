import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("download_resources", Path(__file__).resolve().parents[1] / "scripts/download_resources.py")
resources = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


def archive_fixture(tmp_path, name="data/snapshots/frozen.jsonl", payload=b'{"frozen":true}\r\n'):
    path = tmp_path / "resources.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, payload)
    return path, {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}


def test_restore_preserves_frozen_bytes_and_can_repeat(tmp_path):
    path, expected = archive_fixture(tmp_path)
    root = tmp_path / "restored"
    resources.restore_archive(path, root, expected)
    resources.restore_archive(path, root, expected)
    assert (root / next(iter(expected))).read_bytes() == b'{"frozen":true}\r\n'


def test_conflict_requires_explicit_overwrite(tmp_path):
    path, expected = archive_fixture(tmp_path)
    root = tmp_path / "restored"
    target = root / next(iter(expected))
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local new experiment")
    with pytest.raises(ValueError, match="Existing file differs"):
        resources.restore_archive(path, root, expected)
    assert target.read_bytes() == b"local new experiment"
    resources.restore_archive(path, root, expected, overwrite=True)
    assert resources.digest_file(target) == next(iter(expected.values()))["sha256"]


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/absolute", "data/../../outside", "data\\..\\outside"])
def test_rejects_unsafe_archive_paths(tmp_path, name):
    path, expected = archive_fixture(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe archive path"):
        resources.safe_target(tmp_path / "root", name)
    with pytest.raises(ValueError, match="Unsafe archive path|inventory mismatch"):
        resources.restore_archive(path, tmp_path / "root", expected)


def test_inventory_and_hash_are_required(tmp_path):
    path, expected = archive_fixture(tmp_path)
    with pytest.raises(ValueError, match="inventory mismatch"):
        resources.restore_archive(path, tmp_path / "root", {})
    next(iter(expected.values()))["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        resources.restore_archive(path, tmp_path / "root", expected)


def test_verified_local_archive_needs_no_network(tmp_path, monkeypatch):
    path, _ = archive_fixture(tmp_path)
    entry = {"name": path.name, "bytes": path.stat().st_size, "sha256": resources.digest_file(path)}

    def no_network(*args, **kwargs):
        raise AssertionError("Verified cache must not use the network")

    monkeypatch.setattr(resources, "urlopen", no_network)
    assert resources.download_asset("https://example.invalid", entry, tmp_path) == path
