# Contributing to Curio

## Working agreement

- Keep the default `image_only` route purely visual. Metadata-assisted or research methods must remain explicitly named and opt-in.
- Treat `data/`, `artifacts/`, `models/`, `runs/`, and `logs/` as local or release-managed assets. Do not commit them to the source repository.
- Add tests for backend behavior and update the relevant documentation for any new retrieval method, dataset version, or evaluation result.
- Do not report AI-generated labels as human annotations. Record provenance and limitations in the same change that adds the result.

## Branches and pull requests

Use short branches such as `feature/reference-ranking`, `fix/query-parser`, or `docs/evaluation-notes`. Keep one coherent change per pull request.

Before opening a pull request, run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
pnpm --dir web build
```

The pull request should state the user-visible change, the evaluation or tests run, and any known limitation. Include a screenshot for visible UI changes.

## Review priorities

Reviewers should check data leakage and split isolation first, then ranking behavior, provenance, failure handling, and UI claims. New weights or thresholds must be selected on development data and frozen before test evaluation.
