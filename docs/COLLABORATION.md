# Collaboration workflow

The repository is the shared source of truth for code, tests, documentation, and small reproducibility manifests. Large image collections, model weights, indexes, and generated reports stay outside Git and are identified by SHA256 in the documentation.

## Recommended GitHub setup

Create a **private** repository first and invite every group member with write access. Change it to public only after checking course-material rights, image redistribution terms, credentials, and generated artifacts. Enable pull requests and require one review for changes to `src/`, `scripts/`, `tests/`, or `web/`.

Protect the default branch. Work on feature branches, open a pull request, run the checks in `CONTRIBUTING.md`, and merge only after review. Use Issues for bugs and experiments; label them `backend`, `frontend`, `data`, `evaluation`, or `docs`.

## What belongs in Git

Commit source code, tests, documentation, small protocol files, and lockfiles. Keep downloaded data, indexes, checkpoints, logs, and virtual environments local. The current `.gitignore` enforces this boundary. If a large asset must be shared, publish a separately versioned release artifact or use an approved storage service and record its URL, license, size, and SHA256 in `docs/`.

## Suggested roles for the next phase

- Backend: API, retrieval methods, model loading, latency and security tests.
- Data and provenance: source downloads, licenses, deduplication, split isolation, and version manifests.
- Evaluation: query sets, blinded relevance labels, metrics, confidence calibration, and failure analysis.
- Frontend and integration: search modes, evidence panels, screenshots, and end-to-end checks.

These roles are work areas rather than permanent ownership. Every merged change should still include a clear author and reviewer.
