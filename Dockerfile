FROM python:3.8.12-slim

COPY combat_iq /combat_iq
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -e .

# RUN CONTAINER LOCALLY
CMD uvicorn combat_iq.api.api_file:app --host 0.0.0.0

# RUN CONTAINER ON THE CLOUD
# CMD uvicorn combat_iq.api.api_file:app --host 0.0.0.0 --port $PORT
