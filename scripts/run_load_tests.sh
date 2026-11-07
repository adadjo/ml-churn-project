#!/bin/sh
# Rejoue les 3 scénarios de charge du TP (100 / 500 / 1000 utilisateurs, ramp-up 10 s, 10 boucles)
# avec le générateur Python équivalent au plan JMeter.
# Usage : ./scripts/run_load_tests.sh [URL]       (défaut : http://localhost:8000)
# Sortie : $RESULTS_DIR/load_<N>.json  (défaut : results/)
set -u
URL="${1:-http://localhost:8000}"
RESULTS_DIR="${RESULTS_DIR:-results}"
mkdir -p "$RESULTS_DIR"

# Échauffement (chargement des caches Python, ouverture des premières connexions)
python tests/load/load_runner.py --url "$URL" --users 2 --loops 10 --ramp-up 0 --label "échauffement" >/dev/null 2>&1

for USERS in 100 500 1000; do
  python tests/load/load_runner.py --url "$URL" --users "$USERS" --ramp-up 10 --loops 10 \
      --label "$USERS utilisateurs" --json "$RESULTS_DIR/load_${USERS}.json"
  sleep 5
done
