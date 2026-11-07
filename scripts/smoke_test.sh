#!/bin/sh
# Test de fumée : vérifie qu'une instance déployée répond correctement.
# Usage : ./scripts/smoke_test.sh [URL]      (défaut : http://localhost:8000)
set -eu
URL="${1:-http://localhost:8000}"

echo "-> GET  $URL/health"
curl -fsS "$URL/health"; echo

echo "-> GET  $URL/model/info"
curl -fsS "$URL/model/info"; echo

echo "-> POST $URL/predict"
RESPONSE=$(curl -fsS -X POST "$URL/predict" -H "Content-Type: application/json" \
  -d '{"age": 35, "anciennete": 5, "revenu": 450000, "nb_reclamations": 2, "montant_mensuel": 25000, "retards_paiement": 2}')
echo "$RESPONSE"
echo "$RESPONSE" | grep -q '"prediction"'

echo "-> GET  $URL/metrics"
curl -fsS "$URL/metrics" | grep -q "churn_predictions_total"

echo "SMOKE TEST OK"
