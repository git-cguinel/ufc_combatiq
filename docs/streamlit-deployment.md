# Streamlit deployment and LinkedIn handoff

Deployed on 25 September 2026: https://combat-iq-clement-guinel.streamlit.app/

Runtime: Python 3.11. Verified the hosted Max Griffin / Mark Schultz matchup:
Mark Schultz, 61.9% model confidence, with the target-class caveat displayed.
The LinkedIn profile has not been changed.

## Streamlit Community Cloud

1. Publish intended code changes to the GitHub repository. `.env` and `.envrc`
   are excluded from the current tracked files. Do not publish `.refactor-backup`,
   local environments, `.local-archive`, or credentials.
2. Sign in at https://share.streamlit.io/ with the GitHub-connected account.
3. Create an app using `git-cguinel/ufc_combatiq`, the published branch, and
   `frontend/streamlit_app.py` as the entry point.
4. In Advanced settings select Python 3.11 to match the supported model dependencies.
5. Deploy. Streamlit supplies the final HTTPS link. No API service or API key is needed.

Required files: `frontend/`, `.streamlit/config.toml`, `src/`, `pyproject.toml`,
`requirements.txt`, `README.md`, `data/processed/fighters.csv`, and
`models/of_model3_acc079468.pkl`. The requirements file beside the entry point
installs the root package in editable mode so artifact paths resolve to the checkout.

Reference: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Suggested LinkedIn Featured entry

Title: Combat IQ — an interactive machine-learning demo

Description: Explore a historical UFC matchup, compare fighters’ records, and see
a CatBoost model’s pick and uncalibrated confidence score. A team project
connecting data preparation, machine learning, an API, and an interactive application,
with explicit model limitations and validation status.
Based on historical data through March 2021; an educational demo, not live fight odds.

Use the verified deployed URL for the Featured link. Do not claim a validated
accuracy score from the model filename. The methodology panel explains the original
non-red target grouping and the lack of probability calibration.
