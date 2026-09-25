# Combat IQ

[Open the live demo](https://combat-iq-clement-guinel.streamlit.app/)

An interactive portfolio demo of a historical UFC classification model. The app
shows a model pick and an uncalibrated confidence score, with explicit limitations.
This is an independent educational personal project, not affiliated with UFC.

## Run

Use Python 3.10 or 3.11:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]' -r frontend/requirements.txt
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501. No separate API server is needed for Streamlit.
To run the API separately: `uvicorn combat_iq.api.app:app --reload`.
Interactive API documentation is at http://localhost:8000/docs.

## Production structure

```text
frontend/                 Streamlit interface, styles, deployment dependencies
src/combat_iq/            Inference, feature preparation, artifact cache, API
models/                   The one model used by the app
data/processed/           Fighter statistics used by inference
tests/                    Regression and app behavior checks
.streamlit/config.toml    UI theme
docs/                     Deployment instructions
```

Research notebooks, training code, alternative models, duplicate datasets and
unused images were removed from the serving checkout. They remain in Git history
and in the ignored local `.local-archive/production-cleanup/` folder. Its manifest
records their original paths and SHA-256 checksums; move files back to those paths
to restore them. No training runs occur during app startup or deployment.

## Model and interpretation

The saved CatBoost classifier has 2,500 trees, depth 5, and learning rate 0.04.
Inputs comprise two fighter names and 14 historical statistics per corner. The
fighter archive ends in March 2021; it does not reflect current records or odds.

The original target distinguishes **red wins** from **non-red outcomes**, including
draws. The latter is presented as a blue-corner pick with an explicit explanation.
Confidence refers to that predicted class, not a calibrated chance of winning.
Feature timing, original evaluation and performance on later unseen fights have
not been independently audited in this portfolio version. No validated accuracy
score is claimed; the number in the historical artifact filename is not evidence.

## API contract

`GET /predict?red_fighter=Max%20Griffin&blue_fighter=Mark%20Schultz`
returns `model_pick`, `model_confidence` (0–1), and `predicted_outcome`
(`red_win` or `non_red_win`). Unknown, blank or identical fighters return HTTP 422;
missing artifacts return HTTP 503.

Deprecated `winner`, `win_probability`, `fight_outcome`, and `confidence_rate`
fields remain for existing clients. `win_probability` is a historical misnomer.
The old Uvicorn entry point `combat_iq.api.api_file:app` still works.

## Artifact configuration

Defaults resolve from the checkout, independently of the launch directory.
Override with absolute `COMBAT_IQ_FIGHTERS_PATH` or `COMBAT_IQ_MODEL_PATH` values.
Wheels contain code; data/model artifacts are deployed separately. Only load
trusted pickle files. ML versions are pinned to the saved model's environment.

## Validation and deployment

```sh
make test
make lint
```

Tests check prediction consistency, class-score mapping, input validation, cache
behavior and frontend interaction. These are software checks, not accuracy claims.

The Dockerfile serves the API with its production artifacts; `PORT` defaults to
8000. Streamlit deployment uses `frontend/streamlit_app.py` and Python 3.11.
See [deployment](docs/streamlit-deployment.md).
