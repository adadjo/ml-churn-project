"""API de prédiction du churn client.

Parties 7 (API), 19 (logging), 20 (logs vers ELK) et 21 (métriques Prometheus) du TP.
"""
import json
import os
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from api.logging_config import setup_logging

# --------------------------------------------------------------------------- #
# Configuration et chargement du modèle
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.pkl"))
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
FRONTEND_DIR = BASE_DIR / "frontend"

logger = setup_logging()

model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)  # ordre exact des colonnes vu à l'entraînement

metadata = (
    json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
)
MODEL_NAME = metadata.get("model_name", "Random Forest")
MODEL_VERSION = os.getenv("MODEL_VERSION", metadata.get("model_version", "1.0.0"))

# --------------------------------------------------------------------------- #
# Métriques métier Prometheus (en plus des métriques HTTP de l'Instrumentator)
# --------------------------------------------------------------------------- #
PREDICTIONS = Counter(
    "churn_predictions_total",
    "Nombre de prédictions de churn",
    ["prediction", "model_version"],
)
PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total",
    "Nombre d'erreurs rencontrées lors des prédictions",
    ["error_type"],
)
PREDICTION_DURATION = Histogram(
    "churn_prediction_duration_seconds",
    "Durée de l'inférence du modèle (secondes)",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
PREDICTION_PROBABILITY = Histogram(
    "churn_prediction_probability",
    "Distribution des probabilités de churn prédites",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
MODEL_INFO = Gauge(
    "churn_model_info",
    "Modèle actuellement servi (valeur constante : 1)",
    ["model", "version"],
)

MODEL_INFO.labels(model=MODEL_NAME, version=MODEL_VERSION).set(1)
for _label in ("CHURN", "NO_CHURN"):  # séries présentes dès le démarrage (valeur 0)
    PREDICTIONS.labels(prediction=_label, model_version=MODEL_VERSION)

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="Prédit le risque de départ (churn) d'un client d'un opérateur télécom.",
)


class ClientData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 35,
                "anciennete": 5,
                "revenu": 450000,
                "nb_reclamations": 2,
                "montant_mensuel": 25000,
                "retards_paiement": 2,
            }
        }
    )

    # Bornes larges mais finies : elles protègent le modèle des valeurs aberrantes
    # (dépassement float32, prédictions dénuées de sens) et l'API des entrées malveillantes.
    age: int = Field(..., ge=18, le=120, description="Âge du client (années)")
    anciennete: int = Field(..., ge=0, le=1200, description="Ancienneté (mois)")
    revenu: float = Field(..., ge=0, le=1e9, allow_inf_nan=False, description="Revenu du client")
    nb_reclamations: int = Field(..., ge=0, le=1000, description="Nombre de réclamations")
    montant_mensuel: float = Field(..., ge=0, le=1e8, allow_inf_nan=False,
                                   description="Montant de la facture mensuelle")
    retards_paiement: int = Field(..., ge=0, le=1000, description="Nombre de retards de paiement")


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # On ne renvoie ni la valeur saisie ni le contexte : la valeur fautive peut être inf/NaN,
    # non sérialisable en JSON, ce qui transformerait une erreur 422 en erreur 500.
    errors = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
    PREDICTION_ERRORS.labels(error_type="ValidationError").inc()
    logger.warning(
        "Invalid request",
        extra={"fields": {
            "event": "validation_error",
            "operation": request.url.path,
            "client_ip": _client_ip(request),
            "error": errors[:3],
        }},
    )
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/")
def root():
    return {
        "application": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "UP"}


@app.get("/model/info")
def model_info():
    return {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "features": FEATURES,
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics"),
        "dataset_sha256": metadata.get("dataset_sha256"),
    }


@app.post("/predict")
def predict(data: ClientData, request: Request):
    start = time.perf_counter()
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    client_ip = _client_ip(request)

    try:
        # DataFrame aux noms de colonnes d'entraînement (évite l'avertissement sklearn
        # "X does not have valid feature names")
        features = pd.DataFrame([data.model_dump()], columns=FEATURES)
        probability = float(model.predict_proba(features)[0][1])
        # Équivalent exact de model.predict() pour 2 classes, sans second parcours des 200 arbres
        prediction = 1 if probability > 0.5 else 0
    except Exception as exc:
        duration = time.perf_counter() - start
        PREDICTION_ERRORS.labels(error_type=type(exc).__name__).inc()
        logger.error(
            "Prediction error",
            extra={"fields": {
                "event": "prediction_error",
                "operation": "predict",
                "request_id": request_id,
                "client_ip": client_ip,
                "model": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "response_time_ms": round(duration * 1000, 2),
            }},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction") from exc

    duration = time.perf_counter() - start
    label = "CHURN" if prediction == 1 else "NO_CHURN"

    PREDICTIONS.labels(prediction=label, model_version=MODEL_VERSION).inc()
    PREDICTION_DURATION.observe(duration)
    PREDICTION_PROBABILITY.observe(probability)

    logger.info(
        "Prediction request",
        extra={"fields": {
            "event": "prediction",
            "operation": "predict",
            "request_id": request_id,
            "client_ip": client_ip,
            "model": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "prediction": label,
            "probability": round(probability, 4),
            "response_time_ms": round(duration * 1000, 2),
        }},
    )

    return {
        "prediction": label,
        "probability": round(probability, 4),
        "model_version": MODEL_VERSION,
        "duration_ms": round(duration * 1000, 2),
        "request_id": request_id,
    }


logger.info(
    "Application started",
    extra={"fields": {
        "event": "startup",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
    }},
)

# Interface Web (frontend/index.html) servie sur /ui/
app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

# Exposition des métriques Prometheus sur /metrics
Instrumentator().instrument(app).expose(app)
