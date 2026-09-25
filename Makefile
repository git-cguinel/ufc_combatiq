.PHONY: install test lint serve frontend

install:
	python -m pip install -e ".[dev]" -r frontend/requirements.txt

test:
	python -m pytest

lint:
	ruff check src tests frontend
	ruff format --check src tests frontend

serve:
	uvicorn combat_iq.api.app:app --reload

frontend:
	streamlit run frontend/streamlit_app.py
