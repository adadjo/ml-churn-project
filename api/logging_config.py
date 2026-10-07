"""Journalisation structurée (JSON) - Parties 19 et 20 du TP.

Chaque événement est écrit sur UNE ligne JSON, à la fois sur la sortie standard
(visible avec `docker logs`) et dans un fichier (`LOG_DIR/api.log`) que Logstash
lit pour alimenter Elasticsearch / Kibana.

Variables d'environnement :
    LOG_LEVEL  niveau de log (défaut : INFO)
    LOG_DIR    dossier du fichier de log (défaut : logs)
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "churn_api"


class JsonFormatter(logging.Formatter):
    """Formate un enregistrement en une ligne JSON.

    Les champs métier sont passés via `extra={"fields": {...}}` et fusionnés
    à la racine du document : ils deviennent des champs indexables dans Kibana.
    """

    def format(self, record):
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging():
    """Configure (une seule fois) et retourne le logger de l'application."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / "api.log", maxBytes=10_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        # L'API doit démarrer même si le fichier de log est inaccessible
        logger.warning(
            "Fichier de log indisponible : journalisation sur la sortie standard uniquement",
            extra={"fields": {"event": "log_file_unavailable", "error": str(exc)}},
        )
    return logger
