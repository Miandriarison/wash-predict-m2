import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

# Resolution du chemin vers la racine du projet (wash-predict-m2)
# .parent (affichage) -> .parent (src) -> .parent (racine)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configuration de la page Streamlit
st.set_page_config(
    page_title="WASH Predict - M2 BIHAR",
    page_icon="💧",
    layout="wide",
)

st.title("💧 Plateforme de Prédiction du Taux d'Assainissement (WASH)")
st.markdown(
    "Application de simulation et de prédiction du taux d'accès à l'assainissement basée sur un réseau de neurones (MLP)."
)

st.sidebar.header("⚙️ Paramètres d'entrée")

# Formulaire d'entrée dans la barre latérale
population = st.sidebar.number_input(
    "Population totale", min_value=1, value=3000, step=100
)
beneficiaires_latrine = st.sidebar.number_input(
    "Bénéficiaires latrines basiques", min_value=0, value=100, step=10
)
latrine_partagee = st.sidebar.number_input(
    "Ménages avec latrines améliorées (Partagées)",
    min_value=0,
    value=30,
    step=5,
)
latrine_non_partagee = st.sidebar.number_input(
    "Ménages avec latrines améliorées (Non partagées)",
    min_value=0,
    value=20,
    step=5,
)
milieu = st.sidebar.selectbox(
    "Milieu de résidence", ["Rural", "Urbain"], index=0
)

milieu_urbain_val = 1 if milieu == "Urbain" else 0

# Bouton de prédiction
if st.button("🚀 Calculer la prédiction", type="primary"):
    payload = {
        "population": population,
        "beneficiaires_latrine_basique_total": beneficiaires_latrine,
        "menages_latrine_amelioree_partagee": latrine_partagee,
        "menages_latrine_amelioree_non_partagee": latrine_non_partagee,
        "milieu_URBAIN": milieu_urbain_val,
    }

    # 1. Tentative de requête via l'API FastAPI
    api_url = "http://127.0.0.1:8000/predict"
    prediction_reussie = False
    taux_resultat = 0.0

    try:
        response = requests.post(api_url, json=payload, timeout=3)
        if response.status_code == 200:
            taux_resultat = response.json().get(
                "taux_assainissement_predit", 0.0
            )
            prediction_reussie = True
            st.success("Prédiction obtenue via l'API FastAPI (http://127.0.0.1:8000)")
    except Exception:
        # 2. Mode secours : Inférence locale directe via les fichiers .joblib
        try:
            model = joblib.load(BASE_DIR / "mlp_model.joblib")
            scaler = joblib.load(BASE_DIR / "scaler.joblib")
            df_input = pd.DataFrame([payload])
            scaled_data = scaler.transform(df_input)
            raw_pred = model.predict(scaled_data)[0]
            taux_resultat = float(np.clip(raw_pred, 0.0, 100.0))
            prediction_reussie = True
            st.info("API indisponible - Prédiction calculée via le modèle local (.joblib).")
        except Exception as e:
            st.error(
                f"Impossible d'effectuer la prédiction : {e}"
            )

    # Affichage des résultats
    if prediction_reussie:
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(
                label="Taux d'assainissement prédit",
                value=f"{taux_resultat:.2f} %",
            )
        with col2:
            st.subheader("Niveau d'accès")
            st.progress(min(int(taux_resultat), 100))