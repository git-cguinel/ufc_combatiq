FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml setup.py requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY data/processed ./data/processed
COPY models/of_model3_acc079468.pkl ./models/of_model3_acc079468.pkl
ENV COMBAT_IQ_FIGHTERS_PATH=/app/data/processed/fighters.csv \
    COMBAT_IQ_MODEL_PATH=/app/models/of_model3_acc079468.pkl
CMD ["sh", "-c", "exec uvicorn combat_iq.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
