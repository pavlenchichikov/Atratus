# Contributing

Thanks for your interest in Atratus. This is a personal research project licensed under the
**PolyForm Noncommercial License 1.0.0** (non-commercial) - please keep that in mind for any reuse or contribution.

## Development setup

```bash
pip install -r requirements.txt
pip install ruff pytest
```

The web UI and the test suite do **not** need TensorFlow-heavy training. Only
`train_hybrid.py` (and anything that retrains) is CPU-bound and slow.

## Before opening a pull request

- `pytest -q` - the test suite passes
- `ruff check .` - lint is clean

Both of those can pass here and still fail in CI, for the same reason twice: your
checkout is not CI's. It has the git-ignored local tools, and it has whatever ruff
you installed once.

- **Lint with the pinned version.** CI installs the version named in
  `.github/workflows/ci.yml`; the commit hook is pinned to the same one, so
  `pre-commit install` once is the cheapest way to stop guessing. Different ruff
  versions disagree about real code, so a green run on another version proves
  nothing.
- **Run the suite the way CI runs it**, which is the whole directory in
  alphabetical order, over tracked files only:

  ```bash
  tmp=$(mktemp -d) && git checkout-index -a -f --prefix="$tmp/" && (cd "$tmp" && pytest tests/ -q)
  ```

  A hand-picked list of test files hides two things a full run finds: a committed
  test that imports a git-ignored module (CI fails collection for the WHOLE suite),
  and state one test file leaks into a later one.
- **ASCII-only** source (no smart quotes, em-dashes, or arrows); match the style,
  naming and comment density of the file you are editing
- Keep each PR focused on **one topic**; write a clear description of the problem solved
- **Do not commit** secrets, `.env`, model artifacts, `market.db`, logs, or the local
  research journals (`_ar_*.json`, `_ar_wiki/`) - these are git-ignored on purpose

## Good to know

- `ab_labeling.py` (and its test) is a **local, git-ignored** experiment script - edit it
  locally, but it is not part of the tracked repository.
- Heavy retrains are best run in chunks (`GTRADE_ASSETS`, ~15 assets per process) - see
  [Training](README.md#training) in the README.
- The project follows a design-first, test-driven workflow: a short spec and plan come
  before the code, and changes are small, well-tested, and additive behind env flags
  (default-off and byte-identical) wherever possible.
