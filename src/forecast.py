"""Prévision multi-annuelle du taux d'assainissement (récursive) et scénarios.

Principe : le MLP prédit l'évolution annuelle d'une commune. Pour prévoir plusieurs années, on
réinjecte chaque prédiction comme historique de l'année suivante.

  taux N = taux N-1 + (1 + effet du scénario) x évolution prédite, borné entre 0 et 100

Point de départ : la dernière année disponible (ex. 2025). Comme 2025 est une rupture (+19 points),
l'évolution de départ n'est PAS le saut 2024->2025 mais la pente habituelle de la commune sur la
période régulière (années <= ANNEE_MAX_MODELE).

ATTENTION : l'effet des scénarios (sensibilisation, éducation, campagnes de santé) est une
HYPOTHÈSE paramétrable, pas un effet appris des données (aucune de ces variables n'existe dans la base).
"""

import joblib
import numpy as np
import pandas as pd

from prepa_data import ANNEE_MAX_MODELE, ROOT_DIR, construire_features, construire_paires

CROISSANCE_POPULATION = 0.025  # +2,5 % par an
PERSONNES_PAR_LATRINE = 5  # HYPOTHÈSE : nombre de personnes servies par une latrine (modifiable dans l'application)

# Hausse maximale de la progression annuelle quand le paramètre est à 100 % (HYPOTHÈSES à valider)
EFFETS_MAX = {
    "sensibilisation": 0.30,
    "education": 0.15,
    "campagnes_sante": 0.15,
}
LIBELLES_SCENARIO = {
    "sensibilisation": "Sensibilisation à l'assainissement",
    "education": "Niveau d'éducation",
    "campagnes_sante": "Campagnes de santé",
}


def charger_modele():
    model = joblib.load(ROOT_DIR / "mlp_model.joblib")
    scaler = joblib.load(ROOT_DIR / "scaler.joblib")
    return model, scaler


def effet_scenario(niveaux: dict) -> float:
    """Hausse relative de la progression annuelle (ex. 0.12 = +12 %). niveaux : valeurs entre 0 et 1."""
    return float(
        sum(EFFETS_MAX[k] * min(max(float(niveaux.get(k, 0.0)), 0.0), 1.0) for k in EFFETS_MAX)
    )


def preparer_depart(panel: pd.DataFrame):
    """Une ligne par commune : situation de la dernière année + pente habituelle (période régulière)."""
    d = panel.sort_values(["commune_id", "annee"]).copy()
    annee_depart = int(d["annee"].max())

    reg = d[d["annee"] <= ANNEE_MAX_MODELE].copy()
    reg["delta"] = reg.groupby("commune_id")["taux_assainissement"].diff()
    pente = reg.groupby("commune_id")["delta"].mean().rename("delta_depart")

    dep = d[d["annee"] == annee_depart][
        ["commune_id", "nom_region", "nom_district", "nom_commune", "milieu", "population", "taux_assainissement"]
    ].rename(columns={"population": "pop_depart", "taux_assainissement": "taux_depart"})
    dep = dep.merge(pente, on="commune_id", how="left")
    dep["delta_depart"] = dep["delta_depart"].fillna(0.0)
    # nombre de progressions annuelles observées sur la période régulière (ex. 2020-2024 -> 4)
    dep["n_hist"] = int(reg["annee"].max() - d["annee"].min())
    dep["milieu_URBAIN"] = (dep["milieu"].astype(str).str.strip().str.upper() == "URBAIN").astype(int)
    return dep.reset_index(drop=True), annee_depart


def prevoir(model, scaler, taux_depart, delta_depart, pop_depart, milieu_urbain, nb_annees=4, effet=0.0,
            croissance_pop=CROISSANCE_POPULATION, n_hist=4):
    """Prévision récursive pour un ensemble de communes (tableaux numpy). Retourne (taux, population), forme (n, nb_annees)."""
    taux_prev = np.asarray(taux_depart, dtype=float).copy()
    delta_prev = np.asarray(delta_depart, dtype=float).copy()
    pop = np.asarray(pop_depart, dtype=float).copy()
    urb = np.asarray(milieu_urbain, dtype=int)

    delta_moy = delta_prev.copy()  # progression moyenne passée (départ = pente habituelle)
    n = float(n_hist)
    taux_out, pop_out = [], []
    for _ in range(int(nb_annees)):
        pop = pop * (1 + croissance_pop)
        X = construire_features(taux_prev, taux_prev - delta_prev, pop, urb, delta_moy=delta_moy)
        # progression jamais négative ; l'effet du scénario s'applique à la progression, sans se cumuler
        delta_hat = np.maximum(model.predict(scaler.transform(X)), 0.0)
        taux_prev = np.clip(taux_prev + (1 + effet) * delta_hat, 0.0, 100.0)
        delta_prev = delta_hat
        delta_moy = (delta_moy * n + delta_hat) / (n + 1)
        n += 1
        taux_out.append(taux_prev.copy())
        pop_out.append(pop.copy())
    return np.column_stack(taux_out), np.column_stack(pop_out)


def prevoir_communes(dep: pd.DataFrame, annee_depart: int, model, scaler, nb_annees=4, effet=0.0,
                     croissance_pop=CROISSANCE_POPULATION) -> pd.DataFrame:
    """Prévision pour toutes les communes de `dep`, format long (commune_id, annee, taux, population)."""
    taux, pops = prevoir(
        model, scaler, dep["taux_depart"], dep["delta_depart"], dep["pop_depart"],
        dep["milieu_URBAIN"], nb_annees, effet, croissance_pop,
        int(dep["n_hist"].iloc[0]) if "n_hist" in dep else 4,
    )
    lignes = []
    for k in range(taux.shape[1]):
        lignes.append(
            pd.DataFrame(
                {
                    "commune_id": dep["commune_id"].to_numpy(),
                    "annee": annee_depart + k + 1,
                    "taux": taux[:, k],
                    "population": pops[:, k],
                }
            )
        )
    return pd.concat(lignes, ignore_index=True)


def taux_pondere(df: pd.DataFrame, col_taux: str, col_pop: str = "population") -> pd.Series:
    """Taux national (ou régional) par année, pondéré par la population."""
    g = df.assign(_b=df[col_taux] * df[col_pop]).groupby("annee")
    return (g["_b"].sum() / g[col_pop].sum()).rename("taux")


def prevision_lineaire_nationale(serie_reelle: pd.Series, annee_depart: int, nb_annees=4) -> pd.Series:
    """Méthode actuelle : droite ajustée sur la période régulière, prolongée depuis la dernière valeur réelle."""
    reg = serie_reelle[serie_reelle.index <= ANNEE_MAX_MODELE]
    pente = float(np.polyfit(reg.index.astype(float), reg.to_numpy(), 1)[0]) if len(reg) >= 2 else 0.0
    annees = [annee_depart + k for k in range(1, nb_annees + 1)]
    base = float(serie_reelle.loc[annee_depart])
    return pd.Series([min(base + pente * k, 100.0) for k in range(1, nb_annees + 1)], index=annees, name="lineaire")


def trajectoire_objectif(valeur_depart: float, objectif_final: float, annee_depart: int, nb_annees=4) -> pd.Series:
    """Droite de la valeur de départ (dernière année réelle) jusqu'à l'objectif à l'année finale."""
    annees = [annee_depart + k for k in range(nb_annees + 1)]
    valeurs = np.linspace(valeur_depart, objectif_final, nb_annees + 1)
    return pd.Series(valeurs, index=annees, name="objectif")


def retro_prediction(model, scaler, hist_commune: pd.DataFrame) -> pd.DataFrame:
    """Prédiction à un an du modèle sur les années passées d'une commune (annee, taux_predit)."""
    paires = construire_paires(hist_commune)
    paires = paires[paires["annee"] <= ANNEE_MAX_MODELE]
    if paires.empty:
        return pd.DataFrame(columns=["annee", "taux_predit"])
    from prepa_data import FEATURES

    delta_hat = model.predict(scaler.transform(paires[FEATURES]))
    taux = np.clip(paires["taux_lag1"].to_numpy() + delta_hat, 0.0, 100.0)
    return pd.DataFrame({"annee": paires["annee"].to_numpy(), "taux_predit": taux})


def besoin_terrain(dep_sel: pd.DataFrame, prev_base: pd.DataFrame, prev_scen: pd.DataFrame,
                   objectif: pd.Series, annee_depart: int, personnes_par_latrine: float = PERSONNES_PAR_LATRINE) -> pd.DataFrame:
    """Besoin sur le terrain, cumulé depuis l'année de départ, pour un ensemble de communes.

    dep_sel : lignes de preparer_depart() des communes retenues ; prev_base / prev_scen : sorties de
    prevoir_communes() pour ces communes ; objectif : taux cible (%) par année (index = années, y compris le départ).

    Pour chaque année : personnes à équiper EN PLUS de celles qui ont déjà accès en annee_depart,
    et latrines correspondantes (personnes / personnes_par_latrine, arrondi au-dessus).
    4 cas : maintenir le taux actuel, tendance MLP, avec scénario, atteindre l'objectif.
    """
    pop0 = float(dep_sel["pop_depart"].sum())
    acces0 = float((dep_sel["pop_depart"] * dep_sel["taux_depart"] / 100).sum())
    taux0 = acces0 / pop0 if pop0 else 0.0

    def acces(prev):
        g = prev.assign(_a=prev["population"] * prev["taux"] / 100).groupby("annee")
        return g["_a"].sum(), g["population"].sum()

    acc_b, pop_t = acces(prev_base)
    acc_s, _ = acces(prev_scen)

    lignes = []
    for annee in pop_t.index:
        p = float(pop_t.loc[annee])
        cas = {
            "Maintenir le taux actuel": taux0 * p,
            "Tendance (MLP)": float(acc_b.loc[annee]),
            "Avec scénario": float(acc_s.loc[annee]),
            "Atteindre l'objectif": float(objectif.loc[annee]) / 100 * p,
        }
        ligne = {"Année": int(annee), "Population": round(p)}
        for nom, a in cas.items():
            pers = max(a - acces0, 0.0)
            ligne[f"{nom} — personnes"] = round(pers)
            ligne[f"{nom} — latrines"] = int(np.ceil(pers / personnes_par_latrine))
        lignes.append(ligne)
    return pd.DataFrame(lignes).set_index("Année")


def taux_avec_latrines(dep_sel: pd.DataFrame, prev_base: pd.DataFrame, latrines_par_an: float, annee_depart: int,
                       personnes_par_latrine: float = PERSONNES_PAR_LATRINE) -> pd.Series:
    """Taux (%) par année si l'on construit `latrines_par_an` latrines chaque année (calcul direct, sans le MLP).

    Personnes avec accès = celles de l'année de départ + latrines cumulées x personnes par latrine ;
    la population, elle, continue de croître, donc le taux peut baisser si l'on construit trop peu.
    """
    acces0 = float((dep_sel["pop_depart"] * dep_sel["taux_depart"] / 100).sum())
    pop0 = float(dep_sel["pop_depart"].sum())
    pop_t = prev_base.groupby("annee")["population"].sum()
    valeurs = {annee_depart: acces0 / pop0 * 100 if pop0 else 0.0}
    for annee in pop_t.index:
        k = int(annee) - annee_depart
        acces = acces0 + latrines_par_an * k * personnes_par_latrine
        valeurs[int(annee)] = min(acces / float(pop_t.loc[annee]) * 100, 100.0)
    return pd.Series(valeurs, name="latrines").sort_index()