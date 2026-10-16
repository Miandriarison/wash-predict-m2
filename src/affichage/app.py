import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Racine du projet : <racine>/src/affichage/app.py
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from database import obtenir_donnees_wash  # noqa: E402
from forecast import (  # noqa: E402
    EFFETS_MAX,
    PERSONNES_PAR_LATRINE,
    LIBELLES_SCENARIO,
    besoin_terrain,
    taux_avec_latrines,
    charger_modele,
    effet_scenario,
    preparer_depart,
    prevision_lineaire_nationale,
    prevoir_communes,
    retro_prediction,
    taux_pondere,
    trajectoire_objectif,
)

st.set_page_config(page_title="WASH Predict - M2 BIHAR", page_icon="💧", layout="wide")

# Couleurs (comme le tableau de bord de référence)
C_REEL, C_PREV, C_SCEN, C_OBJ, C_LIN = "#e8604c", "#3b4a8a", "#f39c12", "#00b39f", "#95a5a6"


# ---------- Données et modèle (mis en cache) ----------
@st.cache_data
def charger_panel():
    return obtenir_donnees_wash()


@st.cache_resource
def charger_modele_cache():
    return charger_modele()


try:
    panel = charger_panel()
    model, scaler = charger_modele_cache()
except Exception as e:
    st.error(
        "Impossible de charger les données ou le modèle. Vérifie que PostgreSQL est démarré et que "
        f"`python src/model.py` (ou optimize.py) a été exécuté.\n\nDétail : {e}"
    )
    st.stop()

dep, annee_depart = preparer_depart(panel)

st.title("💧 Plateforme de Prédiction et Suivi du Taux d'Assainissement (WASH)")
st.markdown(
    "Taux d'accès à l'assainissement de base : **réalisation**, **prévision** par réseau de neurones (MLP) "
    "et **objectif**, avec scénarios de sensibilisation."
)

# ---------- Barre latérale ----------
st.sidebar.header("⚙️ Paramètres")
nb_annees = st.sidebar.slider("Années à prévoir", 1, 5, 4)
annee_cible = annee_depart + nb_annees
objectif_final = st.sidebar.number_input(
    f"🎯 Taux cible en {annee_cible} (%)", min_value=0.0, max_value=100.0, value=90.0, step=5.0
)

st.sidebar.divider()
st.sidebar.header("🧪 Scénario")
st.sidebar.caption("Hypothèses de progression : 0 % = aucune action, 100 % = effet maximal.")
niveaux = {}
for cle in EFFETS_MAX:
    niveaux[cle] = st.sidebar.slider(LIBELLES_SCENARIO[cle], 0, 100, 0, step=10) / 100
effet = effet_scenario(niveaux)
st.sidebar.metric("Effet sur la progression annuelle", f"+{effet * 100:.0f} %")

st.sidebar.divider()
st.sidebar.header("🏗️ Terrain")
pers_latrine = st.sidebar.number_input(
    "Personnes servies par latrine (hypothèse)", min_value=1, max_value=50, value=PERSONNES_PAR_LATRINE, step=1
)

with st.sidebar.expander("Hypothèses avancées"):
    croissance_pct = st.number_input(
        "Croissance de la population (%/an)", min_value=0.0, max_value=10.0, value=2.5, step=0.1,
        help="2,5 % = valeur donnée par l'encadreur.",
    )
croissance = croissance_pct / 100

# Prévisions (sans scénario et avec scénario) pour toutes les communes
modele_base = prevoir_communes(dep, annee_depart, model, scaler, nb_annees, 0.0, croissance)
modele_scen = prevoir_communes(dep, annee_depart, model, scaler, nb_annees, effet, croissance)


def tracer(hist, prev_base, prev_scen, objectif, lineaire=None, retro=None, titre="", lat=None):
    """hist/prev_* : Series indexées par année. Les prévisions démarrent à la dernière année réelle."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines+markers", name="Réalisation",
                             line=dict(color=C_REEL, width=3)))
    if retro is not None and len(retro):
        fig.add_trace(go.Scatter(x=retro["annee"], y=retro["taux_predit"], mode="markers",
                                 name="Modèle sur le passé (prédiction à 1 an)",
                                 marker=dict(color=C_PREV, symbol="x", size=9)))

    def prolonger(serie):
        return [annee_depart] + list(serie.index), [float(hist.loc[annee_depart])] + list(serie.values)

    x, y = prolonger(prev_base)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Prévision (MLP)",
                             line=dict(color=C_PREV, dash="dash")))
    if effet > 0:
        x, y = prolonger(prev_scen)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Prévision avec scénario",
                                 line=dict(color=C_SCEN, dash="dash")))
    if lineaire is not None:
        x, y = prolonger(lineaire)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Prévision linéaire (méthode actuelle)",
                                 line=dict(color=C_LIN, dash="dot")))
    if lat is not None:
        fig.add_trace(go.Scatter(x=lat.index, y=lat.values, mode="lines+markers", name="Avec latrines construites",
                                 line=dict(color="#8e44ad", dash="dashdot")))
    fig.add_trace(go.Scatter(x=objectif.index, y=objectif.values, mode="lines+markers", name="Objectif",
                             line=dict(color=C_OBJ, dash="dash")))
    fig.update_layout(title=f"<b>{titre}</b>", yaxis=dict(title="Taux en (%)", range=[0, 100]),
                      xaxis=dict(dtick=1), height=470, legend=dict(orientation="h", y=-0.3, yanchor="top"))
    return fig



def saisir_latrines(dep_sel, prev_b, cle):
    """Simulation : nombre de latrines construites par an -> (nombre, série du taux obtenu ou None)."""
    nb = st.number_input("🏗️ Simulation : latrines construites par an (0 = aucune)", min_value=0, value=0, step=10, key=f"lat_{cle}")
    return nb, (taux_avec_latrines(dep_sel, prev_b, nb, annee_depart, pers_latrine) if nb > 0 else None)


def afficher_besoin(dep_sel, prev_b, prev_s, objectif, nb_lat, lat):
    """Bloc « Besoin sur le terrain » : personnes à équiper et latrines à construire (cumul depuis l'année de départ)."""
    st.subheader("🏗️ Besoin sur le terrain")
    t = besoin_terrain(dep_sel, prev_b, prev_s, objectif, annee_depart, pers_latrine)
    f = t.iloc[-1]
    pop_f = int(f["Population"])
    st.markdown(
        f"D'ici **{annee_cible}**, la population passe à environ **{pop_f:,}** habitants (+{croissance_pct:g} %/an). "
        f"Par rapport à {annee_depart}, il faut équiper en plus :".replace(",", " ")
    )
    a, b, c, d = st.columns(4)
    for col, nom in zip((a, b, c, d), ["Maintenir le taux actuel", "Tendance (MLP)", "Avec scénario", "Atteindre l'objectif"]):
        col.metric(nom, f"{int(f[nom + ' — latrines']):,} latrines".replace(",", " "),
                   delta=f"{int(f[nom + ' — personnes']):,} personnes".replace(",", " "), delta_color="off")
    with st.expander("Détail année par année"):
        st.dataframe(t)
    if lat is not None:
        st.metric(f"Avec {nb_lat} latrines/an : taux {annee_cible}", f"{lat.iloc[-1]:.1f} %",
                  delta=f"{lat.iloc[-1] - float(lat.iloc[0]):+.1f} pts vs {annee_depart}")
    st.caption(
        f"Estimation : 1 latrine pour {pers_latrine} personnes (hypothèse modifiable à gauche). Même sans progresser, "
        "il faut construire pour suivre la croissance de la population. Ces chiffres sont une simulation, "
        "pas un budget ni une mesure de terrain."
    )


tab_commune, tab_nat, tab_perf = st.tabs(["🏘️ Simulation par commune", "🇲🇬 Vue régionale / nationale", "📈 Performance du modèle"])

# ---------- Onglet 1 : commune ----------
with tab_commune:
    c1, c2 = st.columns(2)
    region = c1.selectbox("Région", sorted(panel["nom_region"].unique()))
    sub = panel[panel["nom_region"] == region].drop_duplicates("commune_id")[["commune_id", "nom_commune", "nom_district"]]
    sub = sub.assign(label=sub["nom_commune"] + " — " + sub["nom_district"]).sort_values("label")
    label = c2.selectbox("Commune", sub["label"].tolist())
    cid = int(sub.loc[sub["label"] == label, "commune_id"].iloc[0])

    h = panel[panel["commune_id"] == cid].sort_values("annee")
    hist = h.set_index("annee")["taux_assainissement"]
    pb = modele_base[modele_base["commune_id"] == cid].set_index("annee")["taux"]
    ps = modele_scen[modele_scen["commune_id"] == cid].set_index("annee")["taux"]
    obj = trajectoire_objectif(float(hist.loc[annee_depart]), objectif_final, annee_depart, nb_annees)
    retro = retro_prediction(model, scaler, h)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Réalisation {annee_depart}", f"{hist.loc[annee_depart]:.1f} %")
    m2.metric(f"Prévision {annee_cible} (MLP)", f"{pb.iloc[-1]:.1f} %")
    m3.metric(f"Avec scénario {annee_cible}", f"{ps.iloc[-1]:.1f} %", delta=f"{ps.iloc[-1] - pb.iloc[-1]:+.1f} pts")
    m4.metric(f"Écart à l'objectif {annee_cible}", f"{ps.iloc[-1] - objectif_final:+.1f} pts")

    dep_c, base_c = dep[dep["commune_id"] == cid], modele_base[modele_base["commune_id"] == cid]
    nb_lat, lat = saisir_latrines(dep_c, base_c, "commune")
    st.plotly_chart(tracer(hist, pb, ps, obj, retro=retro, lat=lat, titre=f"Taux d'accès à l'assainissement de base — {label}"))
    afficher_besoin(dep_c, base_c, modele_scen[modele_scen["commune_id"] == cid], obj, nb_lat, lat)
    st.caption(
        "Les croix montrent la prédiction à un an du modèle sur les années passées (années connues du modèle de "
        "production). La vraie évaluation sur une année jamais vue est dans l'onglet Performance."
    )

# ---------- Onglet 2 : région / national ----------
with tab_nat:
    territoire = st.selectbox("Territoire", ["Madagascar (toutes les communes)"] + sorted(panel["nom_region"].unique()))
    ids = panel["commune_id"].unique() if territoire.startswith("Madagascar") else \
        panel.loc[panel["nom_region"] == territoire, "commune_id"].unique()

    hist_n = taux_pondere(panel[panel["commune_id"].isin(ids)].assign(taux=lambda d: d["taux_assainissement"]), "taux")
    pb_n = taux_pondere(modele_base[modele_base["commune_id"].isin(ids)], "taux")
    ps_n = taux_pondere(modele_scen[modele_scen["commune_id"].isin(ids)], "taux")
    lin_n = prevision_lineaire_nationale(hist_n, annee_depart, nb_annees)
    obj_n = trajectoire_objectif(float(hist_n.loc[annee_depart]), objectif_final, annee_depart, nb_annees)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Réalisation {annee_depart}", f"{hist_n.loc[annee_depart]:.1f} %")
    k2.metric(f"Prévision {annee_cible} (MLP)", f"{pb_n.iloc[-1]:.1f} %")
    k3.metric(f"Avec scénario {annee_cible}", f"{ps_n.iloc[-1]:.1f} %", delta=f"{ps_n.iloc[-1] - pb_n.iloc[-1]:+.1f} pts")
    k4.metric(f"Écart à l'objectif {annee_cible}", f"{ps_n.iloc[-1] - objectif_final:+.1f} pts")

    dep_n, base_n = dep[dep["commune_id"].isin(ids)], modele_base[modele_base["commune_id"].isin(ids)]
    nb_lat_n, lat_n = saisir_latrines(dep_n, base_n, "nat")
    st.plotly_chart(tracer(hist_n, pb_n, ps_n, obj_n, lineaire=lin_n, lat=lat_n,
                           titre=f"TAUX D'ACCÈS À L'ASSAINISSEMENT DE BASE — {territoire}"))
    afficher_besoin(dep_n, base_n, modele_scen[modele_scen["commune_id"].isin(ids)], obj_n, nb_lat_n, lat_n)

    tableau = pd.DataFrame({"Réalisation": hist_n, "Prévision MLP": pb_n, "Avec scénario": ps_n,
                            "Linéaire (actuel)": lin_n, "Objectif": obj_n}).round(2)
    tableau.index.name = "Année"
    # l'objectif démarre en 2025 (point de départ du graphique) : pas affiché dans le tableau
    tableau.loc[tableau.index <= annee_depart, "Objectif"] = float("nan")
    # texte : « — » pour les cases vides (sinon Streamlit affiche « None »)
    tableau_txt = tableau.apply(lambda col: col.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"))
    st.dataframe(tableau_txt)
    st.caption(
        f"Communes plafonnées à 100 % en {annee_depart} : elles ne progressent plus, ce qui ralentit la hausse "
        "du taux national."
    )

# ---------- Onglet 3 : performance ----------
with tab_perf:
    chemin = BASE_DIR / "metrics.json"
    if chemin.exists():
        data = json.loads(chemin.read_text(encoding="utf-8"))
        infos = data["infos"]
        st.subheader(f"Évaluation sur {infos.get('annee_test', '?')} (année jamais vue à l'entraînement)")
        st.dataframe(pd.DataFrame(data["modeles"]).T)
        st.markdown(
            f"**Architecture du réseau** : {len(infos.get('variables', []))} variables d'entrée → "
            f"{' → '.join(str(n) for n in infos.get('hidden_layer_sizes', []))} neurones "
            f"({infos.get('nb_couches_cachees', '?')} couches cachées, activation {infos.get('activation', '?')}) → 1 sortie."
        )
        with st.expander("Détails de l'entraînement"):
            st.json(infos)
        a, b = st.columns(2)
        for col, nom, legende in ((a, "predit_vs_reel.png", "Prédit vs réel"), (b, "courbes_perte.png", "Courbe de perte")):
            if (BASE_DIR / nom).exists():
                col.image(str(BASE_DIR / nom), caption=legende)
        if (BASE_DIR / "comparaison_modeles.png").exists():
            st.image(str(BASE_DIR / "comparaison_modeles.png"), caption="Comparaison des méthodes (erreur absolue moyenne)")
    else:
        st.info("Lance `python src/model.py` (ou `optimize.py`) pour générer metrics.json et les graphiques.")

    with st.expander("⚠️ Hypothèses et limites (à citer dans le mémoire)"):
        st.markdown(
            "- Les années 2015-2023 et 2025 sont **simulées** à partir des données réelles de 2024 (population ≈ +2,5 %/an, progression propre à chaque commune avec une part aléatoire) ; seules les données 2024 sont réelles.\n"
            "- 2025 est une **rupture** (+19 points) : elle est exclue de l'entraînement et sert de point de départ des prévisions.\n"
            "- Sur ces données, le MLP fait à peu près aussi bien que les méthodes simples (persistance, tendance nationale) : la progression annuelle y est en grande partie aléatoire, aucune méthode ne la prédit mieux qu'une moyenne.\n"
            "- L'effet de la sensibilisation, de l'éducation et des campagnes de santé est une **hypothèse** paramétrable : ces variables n'existent pas dans la base.\n"
            "- Population : +2,5 % par an dans les prévisions."
        )
