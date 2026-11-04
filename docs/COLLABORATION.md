# Collaboration workflow

The repository is the shared source of truth for code, tests, documentation, and small reproducibility manifests. Images, model weights, indexes and experiment records are shared as versioned **GitHub Release assets in this repository** rather than Git blobs. See [RESOURCES.md](RESOURCES.md) for restoration and SHA256 verification.

## Recommended GitHub setup

The repository is public following the owner's authorization. Anyone can read, clone, fork and open pull requests. Invite group members as collaborators only if they need direct write access. Keep course source documents, credentials and local conversation snapshots outside Git and Release assets. Use pull requests and a reviewer for changes to `src/`, `scripts/`, `tests/`, or `web/`.

Protect the default branch. Work on feature branches, open a pull request, run the checks in `CONTRIBUTING.md`, and merge only after review. Use Issues for bugs and experiments; label them `backend`, `frontend`, `data`, `evaluation`, or `docs`.

## What belongs in Git

Commit source code, tests, documentation, small protocol files, and lockfiles. The current `.gitignore` keeps data, indexes, checkpoints, logs and environments out of Git history. Download shared resources with `python scripts/download_resources.py` or `--all`. New resource versions should use a new Release tag and manifest; do not replace frozen assets to hide changed experimental results.

## Suggested roles for the next phase

- Backend: API, retrieval methods, model loading, latency and security tests.
- Data and provenance: source downloads, licenses, deduplication, split isolation, and version manifests.
- Evaluation: query sets, blinded relevance labels, metrics, confidence calibration, and failure analysis.
- Frontend and integration: search modes, evidence panels, screenshots, and end-to-end checks.

These roles are work areas rather than permanent ownership. Every merged change should still include a clear author and reviewer.
