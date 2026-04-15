# Contributing to mosaicraft

Thanks for your interest in improving mosaicraft! This document describes how
to set up a development environment, run the tests, and submit changes.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to abide by its terms.

## Ways to contribute

* **Bug reports** &mdash; open an issue with steps to reproduce and the smallest
  possible example image.
* **Feature requests** &mdash; describe the use case before the implementation.
* **Pull requests** &mdash; bug fixes, new color transfer methods, new placement
  algorithms, performance work, docs, examples, tests.
* **Real-world examples** &mdash; if you create something cool, we'd love to
  feature it.

## Development setup

```bash
git clone https://github.com/hinanohart/mosaicraft.git
cd mosaicraft
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running the test suite

```bash
pytest                            # full suite
pytest tests/test_color.py        # one file
pytest -k "oklab"                 # match by keyword
pytest --cov=mosaicraft           # with coverage
```

The synthetic fixtures in `tests/conftest.py` mean the suite never needs
external image assets.

## Code style

We use [`ruff`](https://github.com/astral-sh/ruff) for both linting and
import sorting. Run it before committing:

```bash
ruff check src tests
ruff check --fix src tests        # auto-fix
```

Other quality gates:

```bash
bandit -r src -ll                 # security scan
mypy src/mosaicraft               # type checking (best-effort)
```

## Pull request checklist

Before opening a pull request, please make sure:

- [ ] `pytest` passes locally
- [ ] `ruff check src tests` is clean
- [ ] New behavior has tests (use the `synthetic_*` fixtures where possible)
- [ ] Public APIs have docstrings (NumPy style)
- [ ] You added an entry to `CHANGELOG.md` under `[Unreleased]`
- [ ] Commit messages explain the *why* in the body

## Commit message style

```
short summary in imperative mood (<= 60 chars)

Longer paragraph explaining what changed and why, wrapped at 72 chars.
Reference issues with #123. Multiple paragraphs are fine.
```

## Reporting security issues

Please follow [SECURITY.md](SECURITY.md). Do not file public issues for
vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0 (see [LICENSE](LICENSE)).
