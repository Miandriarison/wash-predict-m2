# Image de l'API WASH Predict (FastAPI + front HTML/Bootstrap/Chart.js)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Les dépendances d'abord : cette couche est réutilisée tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Utilisateur non administrateur (bonne pratique de sécurité)
RUN useradd --create-home --uid 1000 app
COPY --chown=app:app . .
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/openapi.json', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
