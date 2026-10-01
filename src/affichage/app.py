from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Resolution du chemin vers la racine du projet (wash-predict-m2)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Configuration de la page Streamlit
st.set_page_config(
    page_title="WASH Predict - M2 BIHAR",
    page_icon="💧",
    layout="wide",
)

st.title("💧 Plateforme de Prédiction et Suivi du Taux d'Assainissement (WASH)")
st.markdown(
    "Simulation interactive du taux d'accès à l'assainissement et suivi des objectifs régionaoux."
)

st.sidebar.header("⚙️ Paramètres d'entrée")

# Formulaire d'entrée interactif (mis à jour en temps réel)
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

st.sidebar.divider()
st.sidebar.header("🎯 Cible & Objectif")
objectif_taux = st.sidebar.number_input(
    "Taux cible à atteindre (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0
)

milieu_urbain_val = 1 if milieu == "Urbain" else 0

# Payload de la requête
payload = {
    "population": population,
    "beneficiaires_latrine_basique_total": beneficiaires_latrine,
    "menages_latrine_amelioree_partagee": latrine_partagee,
    "menages_latrine_amelioree_non_partagee": latrine_non_partagee,
    "milieu_URBAIN": milieu_urbain_val,
}

# Inférence dynamique via FastAPI ou Modèle Local
api_url = "http://127.0.0.1:8000/predict"
prediction_reussie = False
taux_resultat = 0.0

try:
    response = requests.post(api_url, json=payload, timeout=2)
    if response.status_code == 200:
        taux_resultat = response.json().get("taux_assainissement_predit", 0.0)
        prediction_reussie = True
except Exception:
    try:
        model = joblib.load(BASE_DIR / "mlp_model.joblib")
        scaler = joblib.load(BASE_DIR / "scaler.joblib")
        df_input = pd.DataFrame([payload])
        scaled_data = scaler.transform(df_input)
        raw_pred = model.predict(scaled_data)[0]
        taux_resultat = float(np.clip(raw_pred, 0.0, 100.0))
        prediction_reussie = True
    except Exception as e:
        st.error(f"Erreur d'inférence : {e}")

# Affichage comparatif et Graphique
if prediction_reussie:
    ecart = taux_resultat - objectif_taux

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="🎯 Objectif fixé",
            value=f"{objectif_taux:.2f} %",
        )
    with col2:
        st.metric(
            label="🔴 Réalisation (Prédiction)",
            value=f"{taux_resultat:.2f} %",
        )
    with col3:
        st.metric(
            label="📊 Écart (Réalisation - Objectif)",
            value=f"{ecart:.2f} %",
            delta=f"{ecart:.2f} %",
            delta_color="normal",
        )

    st.divider()

    # Histogramme comparatif : Vert (Objectif) vs Rouge (Réalisation)
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Objectif (Cible)"],
            y=[objectif_taux],
            name="Objectif",
            marker_color="#2ecc71",  # Vert
            text=[f"{objectif_taux:.2f}%"],
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            x=["Réalisation (Prédite)"],
            y=[taux_resultat],
            name="Réalisation",
            marker_color="#e74c3c",  # Rouge
            text=[f"{taux_resultat:.2f}%"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Comparaison entre l'Objectif à atteindre et la Réalisation prédite</b>",
        yaxis=dict(title="Taux d'assainissement (%)", range=[0, 100]),
        height=450,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)