"""Tests complémentaires : validation des entrées, cohérence du modèle,
métriques Prometheus, journalisation structurée, gestion des erreurs."""
import json
import logging

from fastapi.testclient import TestClient

import api.main as main
from api.logging_config import JsonFormatter
from api.main import app

client = TestClient(app)

VALID = {
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2,
}


class CaptureHandler(logging.Handler):
    """Récupère les enregistrements émis par le logger de l'application."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def capture_logs():
    handler = CaptureHandler()
    logging.getLogger("churn_api").addHandler(handler)
    return handler


def test_predict_refuse_un_age_invalide():
    response = client.post("/predict", json={**VALID, "age": 5})
    assert response.status_code == 422


def test_predict_refuse_un_champ_manquant():
    payload = {k: v for k, v in VALID.items() if k != "revenu"}
    assert client.post("/predict", json=payload).status_code == 422


def test_les_valeurs_aberrantes_sont_refusees():
    """Sans bornes, 1e39 dépasse la capacité d'un float32 et 1e30 donne une prédiction absurde."""
    for valeur in (1e30, 1e39):
        assert client.post("/predict", json={**VALID, "revenu": valeur}).status_code == 422
    assert client.post("/predict", json={**VALID, "anciennete": 99999}).status_code == 422


def test_infini_et_nan_sont_refuses_sans_erreur_500():
    """Le JSON Python accepte les littéraux Infinity/NaN : ils doivent donner 422, pas 500."""
    corps = '{"age": 35, "anciennete": 5, "revenu": %s, "nb_reclamations": 2, "montant_mensuel": 25000, "retards_paiement": 2}'
    for litteral in ("Infinity", "-Infinity", "NaN"):
        response = client.post("/predict", content=corps % litteral,
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 422, litteral


def test_probabilite_et_classe_sont_coherentes():
    data = client.post("/predict", json=VALID).json()
    assert 0.0 <= data["probability"] <= 1.0
    assert (data["prediction"] == "CHURN") == (data["probability"] > 0.5)


def test_une_forte_anciennete_reduit_le_risque():
    """Test de comportement du modèle : l'ancienneté est la variable la plus protectrice."""
    fidele = client.post("/predict", json={**VALID, "anciennete": 110}).json()["probability"]
    recent = client.post("/predict", json={**VALID, "anciennete": 1}).json()["probability"]
    assert recent > fidele


def test_model_info_expose_les_metadonnees():
    data = client.get("/model/info").json()
    assert data["model"] == "Random Forest"
    assert data["version"] == "1.0.0"
    assert data["features"] == list(VALID.keys())


def test_metriques_http_et_metier_sont_exposees():
    client.post("/predict", json=VALID)
    text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert "churn_predictions_total" in text
    assert "churn_prediction_duration_seconds_bucket" in text


def test_interface_web_est_servie():
    response = client.get("/ui/")
    assert response.status_code == 200
    assert "Prédire" in response.text


def test_le_formatter_produit_une_ligne_json_valide():
    record = logging.LogRecord("churn_api", logging.INFO, __file__, 1, "Prediction request", None, None)
    record.fields = {"model_version": "1.0.0"}
    line = JsonFormatter().format(record)
    parsed = json.loads(line)
    assert "\n" not in line
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "Prediction request"
    assert parsed["model_version"] == "1.0.0"
    assert "timestamp" in parsed


def test_une_prediction_est_journalisee_avec_les_champs_attendus():
    handler = capture_logs()
    try:
        client.post("/predict", json=VALID, headers={"x-request-id": "test-123"})
    finally:
        logging.getLogger("churn_api").removeHandler(handler)

    fields = [r.fields for r in handler.records if getattr(r, "fields", {}).get("event") == "prediction"][0]
    for key in ("operation", "client_ip", "model", "model_version", "prediction",
                "probability", "response_time_ms", "request_id"):
        assert key in fields
    assert fields["request_id"] == "test-123"


def test_une_erreur_interne_retourne_500_et_est_journalisee(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("modèle indisponible")

    monkeypatch.setattr(main.model, "predict_proba", boom)
    handler = capture_logs()
    try:
        response = client.post("/predict", json=VALID)
    finally:
        logging.getLogger("churn_api").removeHandler(handler)

    assert response.status_code == 500
    errors = [r for r in handler.records if r.levelname == "ERROR"]
    assert errors and errors[0].fields["error_type"] == "RuntimeError"
