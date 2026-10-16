# WASH Predict : lancer l'application

L'application finale est une page web (HTML, Bootstrap 5, JavaScript, Chart.js) servie par l'API FastAPI.
Le prototype Streamlit (`src/affichage/app.py`) est conservé comme maquette de validation.

## 1. Avec Docker (recommandé : une seule commande lance tout)

```powershell
# Ton fichier .env existant convient (DB_NAME, DB_USER, DB_PASS). Sinon : copy .env.example .env
docker compose up -d db         # PostgreSQL (les tables sont créées au premier démarrage)
docker compose run --rm pipeline   # ETL -> nettoyage -> entraînement du MLP (quelques minutes)
docker compose up -d api        # l'API et la page web
```

- Page web : http://localhost:8000
- Documentation interactive de l'API : http://localhost:8000/docs
- Les fichiers Excel `data/SEAM_2015.xlsx` … `SEAM_2025.xlsx` doivent être dans le dossier `data/`.
- Dans Docker, `DB_HOST` devient automatiquement `db` ; ton `src/database.py` n'est pas modifié.
- Le mot de passe `DB_PASS` sert à créer la base au premier démarrage : si tu le changes ensuite, fais `docker compose down -v`.
- Si `mlp_model.joblib` a été créé hors de Docker (autre version de scikit-learn), relance `docker compose run --rm pipeline`.
- Tout réinitialiser (base comprise) : `docker compose down -v`.

## 2. Sans Docker

```powershell
pip install -r requirements.txt
python src/etl_pipeline.py ; python src/cleaning.py ; python src/optimize.py
uvicorn src.api:app --reload
```

Le prototype Streamlit : `pip install -r requirements-streamlit.txt` puis `streamlit run src/affichage/app.py`.

## 3. Tests

```powershell
python -m unittest discover -s tests -v      # ou : pytest tests
```

- `tests/test_simulation.py` : calculs (taux entre 0 et 100, scénarios, besoin en latrines, erreurs). Sans base ni serveur.
- `tests/test_api.py` : routes de l'API (codes 200, 404, 422, 503), avec un jeu de données synthétique.

## 4. Architecture

```
Navigateur (HTML + Bootstrap 5 + Chart.js)
      │  JSON (POST /api/simulation, GET /api/meta, ...)
      ▼
FastAPI  src/api.py            validation des entrées (Pydantic), routes, erreurs HTTP
      ▼
src/simulation.py              assemble une simulation (aucun framework web)
      ▼
src/forecast.py + MLP          prévision récursive, scénarios, besoin en latrines
      ▼
PostgreSQL (wash_cleaned_data) panel commune × année
```

## 5. Sécurité (à citer dans le mémoire)

- Entrées validées par Pydantic (bornes sur chaque paramètre) ; les erreurs renvoient 422, 404 ou 503, jamais le détail interne.
- Mot de passe de la base (`DB_PASS`) dans `.env` (hors du code et hors de Git) ; la base n'est pas exposée hors de Docker.
- Images servies par liste blanche (aucun chemin fourni par l'utilisateur).
- Textes venant de l'API insérés avec `textContent` côté navigateur (pas d'injection de code).
- Conteneur exécuté par un utilisateur non administrateur.
- Pas d'authentification : l'application est prévue pour un usage local ou interne, à ajouter avant toute mise en ligne.
