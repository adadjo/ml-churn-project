# Image de base : même version de Python que l'environnement de développement (3.12).
# (L'énoncé propose 3.11, mais le requirements.txt gelé sous Python 3.12 contient
#  NumPy >= 2.5 et SciPy >= 1.18 qui exigent Python >= 3.12.)
FROM python:3.12-slim

# Pas de fichiers .pyc ; sortie non bufferisée (logs visibles immédiatement dans `docker logs`)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Utilisateur non-root (sécurité) et dossier de logs applicatifs
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown appuser:appuser /app/logs

# 1) Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Puis le code et le modèle (voir .dockerignore pour ce qui est exclu)
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# L'image slim n'a pas curl : on utilise Python pour sonder /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
