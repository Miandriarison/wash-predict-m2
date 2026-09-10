import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="WASH Prediction API",
    description="API de prédiction du taux d'accès aux services d'assainissement (M2 BIHAR)",
    version="1.0.0",
)

# Chargement du modèle et du scaler
try:
    model = joblib.load("mlp_model.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    raise RuntimeError(
        f"Erreur lors du chargement des fichiers .joblib : {e}"
    )


# Structure des données d'entrée
class WashInput(BaseModel):
    population: float
    beneficiaires_latrine_basique_total: float
    menages_latrine_amelioree_partagee: float
    menages_latrine_amelioree_non_partagee: float
    milieu_URBAIN: int  # 1 pour Urbain, 0 pour Rural


@app.get("/")
def root():
    return {"status": "online", "message": "API WASH Predict opérationnelle"}


@app.post("/predict")
def predict_wash(data: WashInput):
    try:
        # Conversion de la requête en DataFrame avec les mêmes colonnes que lors de l'entraînement
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])

        # Standardisation des données
        scaled_data = scaler.transform(input_df)

        # Prédiction avec le modèle MLP
        prediction = model.predict(scaled_data)[0]

        # Bornage logique entre 0% et 100%
        taux_predit = float(np.clip(prediction, 0.0, 100.0))

        return {
            "taux_assainissement_predit": round(taux_predit, 2),
            "unite": "%",
            "statut": "Succès",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur de prédiction : {str(e)}"
        )