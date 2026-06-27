FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI=file:/app/mlruns

EXPOSE 8000

WORKDIR /app/src
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
