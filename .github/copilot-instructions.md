# Repository Guidance

- This is a Python 3.13+ project managed with `uv`. Project metadata and dependencies live in `pyproject.toml`; keep `uv.lock` in sync when dependency declarations change.
- Application code uses the `src/proxigen_rag` package layout. The `proxigen-rag` command is configured in `pyproject.toml` to call `proxigen_rag:main`.
- The text-cleaning pipeline is in `src/proxigen_rag/data/cleaning.py`. It reads UTF-8 `.txt` files from `data/raw` and writes cleaned files to `data/cleaned`; preserve this input/output behavior unless the task calls for changing it.
- Keep cleaning transformations explicit and narrowly scoped so they do not remove useful product content.
- There is no configured test framework or established test suite yet. The root `test.py` is incomplete scratch code, not a runnable test suite.