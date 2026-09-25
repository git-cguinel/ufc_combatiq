# Refactor record — 25 September 2026

Original checkout: `/Users/clementguinel/code/git-cguinel/ufc_combatiq`.
Original Git revision: `040840c`.

## Changes

- Moved application code into `src/combat_iq`, separated HTTP, feature construction,
  artifact loading, inference, and training, and retained public legacy imports.
- Moved fighter data from the API package to `data/processed/fighters.csv` and
  training data to `data/raw/fights.csv`. Replaced the root copy with a symlink and
  removed the identical copy from the unused interface package.
- Moved backup implementations and the unused stylesheet to `archive/legacy`.
  Moved `cd_test.ipynb` into `notebooks` and adjusted its model path.
- Removed tracked build output, package metadata, bytecode, notebook checkpoints,
  and CatBoost logs. These are reproducible artifacts, not source code.
- Replaced duplicate prediction implementations with one cached service, removed
  the broken `umpy` import, corrected paths, and normalized outcome whitespace.
- Corrected training's ambiguous DataFrame truth check and unsafe preprocessing.
  New training uses explicit feature names and native CatBoost handling of numeric
  zeros/missing values; it saves separately from the original model.
- Matched scikit-learn to the saved model's recorded version, added packaging and
  tests, repaired the Docker build context and Makefile syntax.

Original models, research notebooks (apart from the noted move), local raw data,
credentials, and the three uncommitted images are preserved. No Git commit,
publication, cloud command, or full production training run was performed.

## Verification

- 23 automated tests pass, including ten original-model prediction fixtures.
- Lint and formatting checks pass.
- Editable installation and wheel build succeed.
- Training smoke test uses 100 historical rows, two iterations, and two CV folds;
  it saves to a temporary directory and verifies reloaded predictions.
- Docker build/run could not be verified because the local Docker daemon is stopped.
- FastAPI's test client emits an upstream httpx deprecation warning; tests pass.

## Rollback

The local `.refactor-backup/` directory contains only the files changed or removed
by this refactor, their original copies, and a manifest. It excludes credentials
and is ignored by Git. Run `python .refactor-backup/undo.py` from the project root
to restore those files and remove files introduced by the refactor. The script
refuses rollback if affected files have been edited since the refactor; preserve
or reconcile those edits first. Untouched files are not changed by rollback.
