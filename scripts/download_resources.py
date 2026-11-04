"""Download and verify the pinned GitHub Release resources (Python stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def safe_target(root, name):
    parts = PurePosixPath(name)
    if "\\" in name or ":" in name or parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Archive path escapes destination: {name}")
    return target


def download_asset(base_url, entry, directory):
    name = entry["name"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Invalid release asset name")
    path = directory / name
    if path.is_file() and path.stat().st_size == entry["bytes"] and digest_file(path) == entry["sha256"]:
        print(f"Cached and verified: {name}", flush=True)
        return path
    partial = path.with_name(name + ".partial")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry["bytes"]:
            partial.unlink()
            offset = 0
        if offset == entry["bytes"]:
            if digest_file(partial) == entry["sha256"]:
                partial.replace(path)
                return path
            partial.unlink()
            offset = 0
        headers = {"User-Agent": "Curio-resource-downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(base_url + "/" + name, headers=headers), timeout=90) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected partial download")
                done = offset if append else 0
                with partial.open("ab" if append else "wb") as output:
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        done += len(block)
                        if done // (128 * 1024**2) != (done - len(block)) // (128 * 1024**2):
                            print(f"Downloading {name}: {done / 1024**2:.0f} MiB", flush=True)
            if partial.stat().st_size != entry["bytes"] or digest_file(partial) != entry["sha256"]:
                partial.unlink()
                raise ValueError(f"Download hash/size mismatch: {name}")
            partial.replace(path)
            print(f"Downloaded and verified: {name}", flush=True)
            return path
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == 3:
                raise RuntimeError(f"Download failed for {name}: {type(error).__name__}") from None
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Download failed: {name}")


def restore_archive(path, root, expected, overwrite=False):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError(f"Archive file inventory mismatch: {path.name}")
        targets = []
        for member in members:
            target = safe_target(root, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16) or member.is_dir():
                raise ValueError("Only regular files are accepted in resource archives")
            entry = expected[member.filename]
            if member.file_size != entry["bytes"]:
                raise ValueError(f"Archive member size mismatch: {member.filename}")
            if target.exists():
                matches = target.is_file() and target.stat().st_size == entry["bytes"] and digest_file(target) == entry["sha256"]
                if matches:
                    continue
                if not overwrite:
                    raise ValueError(f"Existing file differs: {target}. Use a clean directory or --overwrite.")
            targets.append((member, target, entry))
        for member, target, entry in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".curio-partial")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 4 * 1024 * 1024)
            if digest_file(temporary) != entry["sha256"]:
                temporary.unlink()
                raise ValueError(f"Restored file hash mismatch: {member.filename}")
            temporary.replace(target)
    print(f"Restored and verified: {path.name} ({len(targets)} written)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/resources_manifest.json")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--archives-dir", type=Path, default=ROOT / "_release/downloads")
    parser.add_argument("--all", action="store_true", help="Also restore originals, CLIP and explicitly invalid history")
    parser.add_argument("--groups", nargs="+", help="Restore only the named manifest groups")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Check installed files using the release inventory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files with different hashes")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    available = {asset["group"] for asset in manifest["assets"]} - {"inventory"}
    groups = available if args.all else set(args.groups or manifest["default_groups"])
    if groups - available:
        parser.error(f"Unknown groups: {sorted(groups - available)}")
    assets = [asset for asset in manifest["assets"] if asset["group"] in groups]
    total = sum(asset["bytes"] for asset in assets)
    print(f"Selected {len(assets)} archives, {total / 1024**3:.2f} GiB: {', '.join(sorted(groups))}")
    if args.list:
        for asset in assets:
            print(f"  {asset['name']}  {asset['bytes'] / 1024**2:.1f} MiB")
        return
    args.archives_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}"
    inventory_asset = next(asset for asset in manifest["assets"] if asset["group"] == "inventory")
    inventory_path = download_asset(url, inventory_asset, args.archives_dir)
    inventory = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.verify_only:
        for entry in inventory:
            if entry["group"] in groups:
                path = safe_target(root, entry["path"])
                if not path.is_file() or path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
                    raise ValueError(f"Installed resource missing or changed: {entry['path']}")
        print("All selected installed resources match the release inventory.")
        return
    for asset in assets:
        path = download_asset(url, asset, args.archives_dir)
        expected = {entry["path"]: entry for entry in inventory if entry["asset"] == asset["name"]}
        if len(expected) != asset["files"]:
            raise ValueError(f"Incomplete inventory for {asset['name']}")
        restore_archive(path, root, expected, args.overwrite)
    print("Resources restored. Install dependencies and build web/, then run scripts/run.ps1.")


if __name__ == "__main__":
    main()
