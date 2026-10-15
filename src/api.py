import sys
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Permet de lancer aussi bien `uvicorn src.api:app` (depuis la racine) que `uvicorn api:app` (depuis src/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast import charger_modele, effet_scenario, prevoir
from prepa_data import construire_features

app = FastAPI(
    title="WASH Prediction API",
    description="Prévision du taux d'accès à l'assainissement par commune (MLP) - M2 BIHAR",
    version="2.0.0",
)

try:
    model, scaler = charger_modele()
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement de mlp_model.joblib / scaler.joblib : {e}")


class PredictInput(BaseModel):
    taux_lag1: float = Field(ge=0, le=100, description="Taux de l'année N-1 (%)")
    taux_lag2: float = Field(ge=0, le=100, description="Taux de l'année N-2 (%)")
    population: float = Field(gt=0, description="Population de l'année N")
    milieu_URBAIN: int = Field(ge=0, le=1)


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


@app.get("/")
def root():
    return {"status": "online", "message": "API WASH Predict opérationnelle", "modele": type(model).__name__}


@app.post("/predict")
def predict_next_year(data: PredictInput):
    """Taux prévu pour l'année suivante d'une commune."""
    try:
        X = construire_features(data.taux_lag1, data.taux_lag2, data.population, data.milieu_URBAIN)
        delta = float(max(model.predict(scaler.transform(X))[0], 0.0))
        taux = float(np.clip(data.taux_lag1 + delta, 0.0, 100.0))
        return {"taux_assainissement_predit": round(taux, 2), "unite": "%", "statut": "Succès"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")


@app.post("/forecast", response_model=List[PointPrevision])
def forecast(data: ForecastInput):
    """Trajectoire sur plusieurs années pour une commune, avec scénario optionnel."""
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