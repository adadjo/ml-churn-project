"""Configuration commune des tests : les fichiers de log vont dans un dossier temporaire
(on évite de polluer le dépôt avec logs/api.log). Ce fichier est chargé avant les tests."""
import os
import tempfile

os.environ.setdefault("LOG_DIR", tempfile.mkdtemp(prefix="churn-test-logs-"))
