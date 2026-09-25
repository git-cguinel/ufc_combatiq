# UFC Combat IQ

A FastAPI service that predicts a UFC fight outcome from two fighters' historical statistics.
The bundled data is historical (through March 2021); predictions are not current fight advice.

## Run locally

Use Python 3.10 or 3.11. The ML dependency versions are pinned for the bundled model;
its scikit-learn serialization requires **1.4.1.post1**.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
uvicorn combat_iq.api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation. Example:

```sh
curl --get http://127.0.0.1:8000/predict \
  --data-urlencode 'red_fighter=Gustavo Lopez' \
  --data-urlencode 'blue_fighter=Mark Kerr'
```

The response provides `model_pick`, `model_confidence`, and `predicted_outcome`.
The target distinguishes `red_win` from `non_red_win`, which includes draws. A
non-red prediction is displayed as a blue fighter pick, not a proven blue win.
Confidence is the uncalibrated score for that target class, rounded to three decimals.

Legacy `winner`, `win_probability`, `fight_outcome`, and `confidence_rate` fields
remain for compatibility and are deprecated in the API schema. In particular,
`win_probability` is a historical misnomer, not a calibrated probability of winning.

Portfolio validation: regression tests verify software consistency, not model
accuracy. Original evaluation, pre-fight feature timing, and later unseen-fight
performance have not been independently audited in this version. No validated
accuracy or betting performance is claimed.

Unknown, empty, or identical fighters return HTTP 422. Missing artifacts return HTTP 503.

## Structure

```text
src/combat_iq/
  api/app.py          FastAPI routes and response schema
  config.py           Paths and environment overrides
  data.py             Validated fighter data and cache
  features.py         Shared feature names, order, and matchup validation
  prediction.py       Saved-model loading and inference
  training.py         Explicit model training and evaluation
  ml_logic/           Compatibility wrappers for existing notebooks
data/
  processed/fighters.csv
  raw/fights.csv
models/               Original fitted models; unchanged
notebooks/            Research and exploration
archive/legacy/       Historical alternate implementations and unused stylesheet
tests/               Regression, API, cache, and training tests
```

The root `data.csv` is a compatibility symlink. Existing `raw_data/` files are left
untouched for research notebooks. Some historical notebooks need external datasets;
notebooks are preserved as research records, not production entry points.

## Configuration

Defaults resolve relative to the source checkout, independent of the launch directory.
For a wheel installation, provide absolute paths to separately deployed artifacts:

| Variable | Default in checkout |
| --- | --- |
| `COMBAT_IQ_FIGHTERS_PATH` | `data/processed/fighters.csv` |
| `COMBAT_IQ_MODEL_PATH` | `models/of_model3_acc079468.pkl` |
| `COMBAT_IQ_TRAINING_PATH` | `data/raw/fights.csv` |

Fighter data and models are cached by path, modification time, and size. Replacing an
artifact refreshes the cache. Only use trusted pickle models. Python wheels contain
code; datasets and fitted models are deployed separately.

## Train a new model

```python
from combat_iq.training import train_pipeline

model = train_pipeline()
```

Training uses the shared 30-column feature schema, a stratified holdout, reproducible
splits, and five-fold cross-validation. CatBoost handles numeric zeros and missing
values directly. Results are logged using Python logging. Training saves to
`models/trained_model.pkl`, leaving the historical model untouched. Set
`COMBAT_IQ_MODEL_PATH` to the new file to use it. The new workflow corrects the old
training implementation; it does not claim to reproduce its historical accuracy.

Existing imports of `preprocessed_df`, `get_data`, `predict`, and the misspelled
`train_pipline` continue to work through `combat_iq.ml_logic`. The old Uvicorn entry
point `combat_iq.api.api_file:app` is also retained.

## Checks

```sh
make test
make lint
python -m pip wheel --no-deps . -w dist
```

Regression fixtures capture ten predictions from the original preprocessing and
saved model, covering both labels. Tests also exercise feature order/dtypes, path
independence, cache isolation/refresh, HTTP responses, and a small real training run.

## Docker

```sh
docker build -t ufc-combat-iq .
docker run --rm -p 8000:8000 ufc-combat-iq
```

The image contains the default fighter data and historical model. `PORT` defaults to
8000. Cloud deployment targets remain in the Makefile and require your existing
cloud environment settings. No deployment is performed by installation or testing.

See [the refactor record](docs/refactor.md) for migration details and local rollback.

## Streamlit demo

```sh
source .venv/bin/activate
python -m pip install -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. The frontend calls the cached prediction service directly;
no separate API server is needed. See [deployment](docs/streamlit-deployment.md).
