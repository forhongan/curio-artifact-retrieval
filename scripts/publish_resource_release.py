"""Publish locally prepared assets; Git Credential Manager keeps credentials off disk."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--publish", action="store_true", help="Make the completed draft public after all checks")
    args = parser.parse_args()
    manifest_path = args.directory / "resources_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repository, tag = manifest["repository"], manifest["tag"]
    if repository != "forhongan/curio-artifact-retrieval":
        raise ValueError("Unexpected repository")
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\nusername=forhongan\n\n",
                            text=True, capture_output=True, check=True)
    credential = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = credential.get("password")
    if not token:
        raise RuntimeError("No GitHub credential available")
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "Curio-resource-release"}
    api = f"https://api.github.com/repos/{repository}"

    def request(method, url, **kwargs):
        response = requests.request(method, url, headers=headers, timeout=(30, 120), **kwargs)
        if not response.ok:
            raise RuntimeError(f"GitHub API {method}: HTTP {response.status_code}")
        return response.json()

    releases = request("GET", api + "/releases?per_page=100")
    release = next((item for item in releases if item["tag_name"] == tag), None)
    if release is None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        body = (ROOT / "docs/RESOURCE_RELEASE_NOTES.md").read_text(encoding="utf-8")
        release = request("POST", api + "/releases", json={"tag_name": tag, "target_commitish": commit,
                          "name": "Curio v2 complete resources / 完整数据、模型与实验记录", "body": body, "draft": True})
    assets = list(manifest["assets"])
    for name in ("resources_manifest.json", "SHA256SUMS.txt"):
        path = args.directory / name
        assets.append({"name": name, "bytes": path.stat().st_size, "sha256": file_digest(path)})
    for entry in assets:
        path = args.directory / entry["name"]
        if path.stat().st_size != entry["bytes"] or file_digest(path) != entry["sha256"]:
            raise ValueError(f"Local asset changed: {entry['name']}")
    upload_url = release["upload_url"].split("{", 1)[0]
    if not upload_url.startswith("https://uploads.github.com/repos/" + repository + "/"):
        raise ValueError("Unexpected upload destination")
    existing = {entry["name"]: entry for entry in request("GET", release["assets_url"] + "?per_page=100")}

    def upload(entry):
        name = entry["name"]
        remote = existing.get(name)
        if remote and remote["state"] == "uploaded":
            if remote["size"] != entry["bytes"] or remote.get("digest") != "sha256:" + entry["sha256"]:
                raise ValueError(f"Existing release asset hash differs: {name}")
            print(f"Already uploaded and verified: {name}", flush=True)
            return
        if remote:
            if not release["draft"]:
                raise ValueError(f"Incomplete asset on published release: {name}")
            response = requests.delete(remote["url"], headers=headers, timeout=30)
            if response.status_code != 204:
                raise RuntimeError(f"Could not remove incomplete draft asset: {name}")
        for attempt in range(4):
            print(f"Uploading {name}: {entry['bytes'] / 1024**2:.1f} MiB (attempt {attempt + 1})", flush=True)
            started = time.monotonic()
            try:
                with (args.directory / name).open("rb") as data:
                    response = requests.post(upload_url, params={"name": name}, data=data,
                                             headers={**headers, "Content-Type": "application/octet-stream",
                                                      "Content-Length": str(entry["bytes"])}, timeout=(30, 300))
                if response.status_code != 201:
                    raise RuntimeError(f"Upload HTTP {response.status_code}: {name}")
                remote = response.json()
                if remote["size"] != entry["bytes"] or remote.get("digest") != "sha256:" + entry["sha256"]:
                    raise ValueError(f"Uploaded asset digest mismatch: {name}")
                elapsed = time.monotonic() - started
                print(f"Verified {name}: {elapsed:.1f}s, {entry['bytes'] / max(elapsed, .01) / 1024**2:.2f} MiB/s", flush=True)
                return
            except (requests.RequestException, RuntimeError):
                if attempt == 3:
                    raise RuntimeError(f"Upload failed after retries: {name}") from None
                # A lost response can leave an uploaded asset. Reconcile before retrying.
                current = request("GET", release["assets_url"] + "?per_page=100")
                remote = next((item for item in current if item["name"] == name), None)
                if remote and remote["state"] == "uploaded" and remote.get("digest") == "sha256:" + entry["sha256"]:
                    print(f"Recovered successful upload: {name}", flush=True)
                    return
                if remote and release["draft"]:
                    deleted = requests.delete(remote["url"], headers=headers, timeout=30)
                    if deleted.status_code != 204:
                        raise RuntimeError(f"Cannot clean incomplete upload: {name}") from None
                time.sleep(3 * (attempt + 1))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(upload, assets))
    remote_assets = {item["name"]: item for item in request("GET", release["assets_url"] + "?per_page=100")}
    for entry in assets:
        remote = remote_assets.get(entry["name"], {})
        if remote.get("state") != "uploaded" or remote.get("size") != entry["bytes"] or remote.get("digest") != "sha256:" + entry["sha256"]:
            raise ValueError(f"Final remote verification failed: {entry['name']}")
    if args.publish and release["draft"]:
        release = request("PATCH", release["url"], json={"draft": False, "make_latest": "true"})
    receipt = {"url": release["html_url"], "draft": release["draft"], "tag": tag,
               "assets": [{"name": item["name"], "bytes": item["size"], "digest": item.get("digest")}
                          for item in remote_assets.values()]}
    (args.directory / "upload_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"url": release["html_url"], "draft": release["draft"], "verified_assets": len(assets)}), flush=True)


if __name__ == "__main__":
    main()
