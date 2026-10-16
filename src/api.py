"""API FastAPI de la plateforme WASH Predict (M2 BIHAR).

Lancement (depuis la racine du projet) :  uvicorn src.api:app --reload
Documentation interactive automatique :   http://localhost:8000/docs

Routes
  GET  /                              page web (HTML/Bootstrap/Chart.js)
  GET  /sante                         vérification de l'état (modèle, base)
  GET  /api/meta                      régions, scénarios, valeurs par défaut
  GET  /api/communes?region=...       communes d'une région
  POST /api/simulation                simulation complète (courbes, indicateurs, besoin en latrines, tableau)
  GET  /api/performance               métriques d'évaluation du modèle (metrics.json)
  GET  /api/performance/image/{nom}   graphiques d'évaluation (png)
  POST /predict, /forecast            prédiction brute d'une commune (routes de la première version)
"""

import sys
import threading
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Permet de lancer aussi bien `uvicorn src.api:app` (depuis la racine) que `uvicorn api:app` (depuis src/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import simulation as sim  # noqa: E402
from forecast import PERSONNES_PAR_LATRINE, charger_modele, effet_scenario, prevoir  # noqa: E402
from prepa_data import ROOT_DIR, construire_features  # noqa: E402

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="WASH Prediction API",
    description="Prévision du taux d'accès à l'assainissement par commune (MLP) - M2 BIHAR",
    version="3.0.0",
)

# ---------------------------------------------------------------------------
# Contexte (données + modèle), chargé à la première requête puis gardé en mémoire
# ---------------------------------------------------------------------------
_ctx: Optional[sim.Contexte] = None
_verrou = threading.Lock()


def set_contexte(ctx: Optional[sim.Contexte]) -> None:
    """Remplace le contexte (utilisé par les tests pour éviter PostgreSQL)."""
    global _ctx
    _ctx = ctx


def contexte() -> sim.Contexte:
    global _ctx
    if _ctx is None:
        with _verrou:
            if _ctx is None:
                try:
                    _ctx = sim.Contexte.depuis_base()
                except Exception as e:  # base arrêtée, modèle absent...
                    raise HTTPException(
                        status_code=503,
                        detail="Données ou modèle indisponibles : vérifie que PostgreSQL est démarré et que le pipeline a été "
                               "exécuté (`docker compose run --rm pipeline`, ou `python src/optimize.py` sans Docker). "
                               f"Détail : {e}",
                    )
    return _ctx


# ---------------------------------------------------------------------------
# Schémas d'entrée
# ---------------------------------------------------------------------------
class SimulationInput(BaseModel):
    type_territoire: Literal["national", "region", "commune"] = "national"
    valeur: Optional[Union[str, int]] = Field(default=None, description="Nom de la région ou identifiant de la commune")
    nb_annees: int = Field(default=4, ge=1, le=10)
    objectif_final: float = Field(default=90.0, ge=0, le=100, description="Taux cible à l'année finale (%)")
    sensibilisation: float = Field(default=0.0, ge=0, le=1)
    education: float = Field(default=0.0, ge=0, le=1)
    campagnes_sante: float = Field(default=0.0, ge=0, le=1)
    croissance_pct: float = Field(default=2.5, ge=0, le=10, description="Croissance annuelle de la population (%)")
    personnes_par_latrine: float = Field(default=PERSONNES_PAR_LATRINE, ge=1, le=50)
    latrines_par_an: float = Field(default=0.0, ge=0, le=10_000_000, description="Simulation : latrines construites par an")


class PredictInput(BaseModel):
    taux_lag1: float = Field(ge=0, le=100, description="Taux de l'année N-1 (%)")
    taux_lag2: float = Field(ge=0, le=100, description="Taux de l'année N-2 (%)")
    population: float = Field(gt=0, description="Population de l'année N")
    milieu_URBAIN: int = Field(ge=0, le=1)
    delta_moy: Optional[float] = Field(default=None, description="Progression annuelle moyenne passée (points/an), facultatif")


class ForecastInput(BaseModel):
    taux_depart: float = Field(ge=0, le=100, description="Taux de la dernière année connue (%)")
    delta_depart: float = Field(ge=0, description="Progression annuelle habituelle (points/an)")
    population: float = Field(gt=0, description="Population de la dernière année connue")
    milieu_URBAIN: int = Field(ge=0, le=1)
    annee_depart: int = 2025
    nb_annees: int = Field(default=4, ge=1, le=10)
    sensibilisation: float = Field(default=0.0, ge=0, le=1)
    education: float = Field(default=0.0, ge=0, le=1)
    campagnes_sante: float = Field(default=0.0, ge=0, le=1)


class PointPrevision(BaseModel):
    annee: int
    taux: float
    population: int


# ---------------------------------------------------------------------------
# Routes de l'application web
# ---------------------------------------------------------------------------
@app.get("/sante")
def sante():
    """État de l'API : le contexte est chargé à la demande, on le teste ici."""
    ctx = contexte()
    return {"statut": "ok", "modele": type(ctx.model).__name__, "annee_depart": ctx.annee_depart,
            "communes": int(ctx.panel["commune_id"].nunique())}


@app.get("/api/meta")
def api_meta():
    return sim.meta(contexte())


@app.get("/api/communes")
def api_communes(region: str):
    try:
        return sim.communes(contexte(), region)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


@app.post("/api/simulation")
def api_simulation(data: SimulationInput):
    try:
        return sim.simuler(
            contexte(), data.type_territoire, data.valeur, data.nb_annees, data.objectif_final,
            data.sensibilisation, data.education, data.campagnes_sante, data.croissance_pct,
            data.personnes_par_latrine, data.latrines_par_an,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/performance")
def api_performance():
    res = sim.performance()
    if res is None:
        raise HTTPException(status_code=404, detail="metrics.json introuvable : lance le pipeline (`docker compose run --rm pipeline` ou `python src/optimize.py`).")
    return res


@app.get("/api/performance/image/{nom}")
def api_performance_image(nom: str):
    # liste blanche : jamais de chemin fourni par l'utilisateur
    fichier = sim.IMAGES_PERFORMANCE.get(nom)
    if fichier is None or not (ROOT_DIR / fichier).exists():
        raise HTTPException(status_code=404, detail="Image introuvable")
    return FileResponse(ROOT_DIR / fichier, media_type="image/png")


# ---------------------------------------------------------------------------
# Routes de la première version (prédiction brute)
# ---------------------------------------------------------------------------
_modele_brut = None


def _modele():
    global _modele_brut
    if _modele_brut is None:
        try:
            _modele_brut = charger_modele()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Erreur lors du chargement de mlp_model.joblib / scaler.joblib : {e}")
    return _modele_brut


@app.post("/predict")
def predict_next_year(data: PredictInput):
    """Taux prévu pour l'année suivante d'une commune."""
    model, scaler = _modele()
    try:
        X = construire_features(data.taux_lag1, data.taux_lag2, data.population, data.milieu_URBAIN, delta_moy=data.delta_moy)
        delta = float(max(model.predict(scaler.transform(X))[0], 0.0))
        taux = float(np.clip(data.taux_lag1 + delta, 0.0, 100.0))
        return {"taux_assainissement_predit": round(taux, 2), "unite": "%", "statut": "Succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")


@app.post("/forecast", response_model=List[PointPrevision])
def forecast(data: ForecastInput):
    """Trajectoire sur plusieurs années pour une commune, avec scénario optionnel."""
    model, scaler = _modele()
    try:
        effet = effet_scenario(
            {"sensibilisation": data.sensibilisation, "education": data.education,
             "campagnes_sante": data.campagnes_sante}
        )
        taux, pops = prevoir(
            model, scaler, [data.taux_depart], [data.delta_depart], [data.population],
            [data.milieu_URBAIN], data.nb_annees, effet,
        )
        return [
            PointPrevision(annee=data.annee_depart + k + 1, taux=round(float(taux[0, k]), 2),
                           population=int(round(pops[0, k])))
            for k in range(taux.shape[1])
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prévision : {e}")


# ---------------------------------------------------------------------------
# Front-end (fichiers statiques)
# ---------------------------------------------------------------------------
if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def accueil():
    page = WEB_DIR / "index.html"
    if page.exists():
        return FileResponse(page)
    return JSONResponse({"statut": "online", "message": "API WASH Predict opérationnelle (front absent)"})
