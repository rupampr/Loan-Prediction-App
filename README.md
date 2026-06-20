# Loan Prediction Model — Deployment Guide

## Overview

This repository contains a Jupyter notebook that trains a loan default prediction model and a small web app to serve predictions. Use this guide to run the notebook locally, export the trained model, and deploy the application.

## Key files

- `LoanPredictionModel.ipynb` — model training and evaluation notebook
- `loan_approval_dataset.csv` — dataset used for training
- `app.py` — Flask app to serve predictions

## Prerequisites

- Python 3.8 or newer
- pip
- (Optional) Docker for containerized deployment

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install required packages (common ones used by the notebook):

```bash
pip install pandas scikit-learn matplotlib seaborn jupyterlab flask joblib
```

Create a `requirements.txt` if you need reproducible installs:

```bash
pip freeze > requirements.txt
```

## Run the Notebook

1. Start JupyterLab or Jupyter Notebook:

```bash
jupyter lab
```

2. Open `LoanPredictionModel.ipynb` and run cells in order. The notebook includes sections for data exploration, preprocessing, model training, evaluation, and saving the trained model (commonly saved using `joblib` or `pickle`).

## Export / Save the Model

If the notebook saves the trained model to a file (e.g., `model.joblib`), ensure the web app expects the same filename and path. Typical pattern in notebook:

```python
import joblib
joblib.dump(trained_model, 'model.joblib')
```

## Run the Web App Locally

1. Ensure the exported model file is present in the repository root (or update the path in `app.py`).

2. Run the Flask app:

```bash
python app.py
```

3. By default the app will bind to `localhost:5000`. Use the provided endpoints to send test requests (see `app.py` for exact routes and expected payloads).

## Docker Deployment (Optional)

Create a `Dockerfile` like this minimal example:

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:

```bash
docker build -t loan-predict-app .
docker run -p 5000:5000 loan-predict-app
```

## Platform Notes (Heroku / Cloud Providers)

- Ensure the web server binds to the port provided by the environment variable (e.g., `PORT`).
- For production use Gunicorn or another WSGI server instead of Flask's built-in server.

Example `Procfile` for Heroku:

```
web: gunicorn app:app
```

## Environment & Configuration

- If `app.py` reads configuration or secrets (API keys, model path), provide them via environment variables or a config file outside the repo.
- Confirm model input feature ordering matches the preprocessing used during training.

## Troubleshooting

- If predictions are wrong, confirm the same preprocessing (scaling, encoding) is applied at inference time.
- If model file not found, ensure the path in `app.py` matches the saved location.

## Next Steps

- Optionally create a lightweight `requirements.txt` and `Dockerfile` in this repo.
- Add a CI step to run the notebook or tests before deploying.

---

