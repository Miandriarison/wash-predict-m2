import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(
    page_title="WASH Access Rate Predictor - Madagascar",
    page_icon="💧",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"

st.title("💧 Dashboard de Prédiction du Taux d'Accès à l'Assainissement (M2 BIHAR)")
st.markdown("---")

col_form, col_visu = st.columns([1, 2])

with col_form:
    st.header("📋 Données de la Commune")
    pop = st.number_input("Population totale", min_value=100, value=12500, step=500)
    benef_lat = st.number_input("Bénéficiaires latrines basiques", min_value=0, value=3200, step=100)
    lat_part = st.number_input("Ménages latrines améliorées partagées", min_value=0, value=450, step=50)
    lat_non_part = st.number_input("Ménages latrines améliorées non partagées", min_value=0, value=1200, step=50)
    milieu = st.radio("Milieu de résidence", ["Rural", "Urbain"])

    btn_predict = st.button("🚀 Prédire le Taux d'Accès", use_container_width=True)

with col_visu:
    st.header("📊 Résultat de la Prédiction")

    if btn_predict:
        payload = {
            "population": float(pop),
            "beneficiaires_latrine_basique_total": float(benef_lat),
            "menages_latrine_amelioree_partagee": float(lat_part),
            "menages_latrine_amelioree_non_partagee": float(lat_non_part),
            "milieu_URBAIN": 1 if milieu == "Urbain" else 0
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                res = response.json()
                taux = res["valeur_numerique"]

                # KPI
                st.metric(label="Taux d'Accès Prédit", value=f"{taux}%")

                # Jauge Plotly
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=taux,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Taux d'Accès Prévisionnel (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#1f77b4"},
                        'steps': [
                            {'range': [0, 30], 'color': "#ffcccc"},
                            {'range': [30, 70], 'color': "#fff2cc"},
                            {'range': [70, 100], 'color': "#d9ead3"}
                        ]
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error("Erreur de communication avec l'API FastAPI.")
        except Exception as e:
            st.error(f"Vérifie que FastAPI est bien démarré sur http://127.0.0.1:8000. Erreur : {e}")
    else:
        st.info("Renseigne les paramètres à gauche et clique sur 'Prédire'.")