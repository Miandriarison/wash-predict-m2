"""Préparation des données pour la prédiction du taux d'assainissement dans le temps.

Principe : on prédit l'ÉVOLUTION annuelle (delta = taux N - taux N-1) d'une commune à partir de
son historique (taux N-1, évolution précédente), de sa population et de son milieu.
Taux prédit = taux N-1 + delta prédit, borné entre 0 et 100.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
SCALER_PATH = ROOT_DIR / "scaler.joblib"

# L'ORDRE de cette liste doit être identique à l'entraînement, à l'API et à l'application
# delta_moy = progression annuelle MOYENNE passée de la commune (depuis sa première année connue) :
# elle lisse le bruit d'une année à l'autre et capte la vitesse propre de chaque commune.
FEATURES = ["taux_lag1", "delta_lag1", "delta_moy", "log_population", "milieu_URBAIN"]
# Les années postérieures à ANNEE_MAX_MODELE (ex. 2025 : saut de +19 points) ne servent ni à
# l'entraînement ni au test : elles sont le point de départ des prévisions, et la rupture est
# expliquée dans le mémoire plutôt qu'apprise comme une tendance régulière.
ANNEE_MAX_MODELE = 2024


def construire_features(taux_lag1, taux_lag2, population, milieu_urbain, index=None, delta_moy=None) -> pd.DataFrame:
    """Construit les variables d'entrée (utilisé à l'entraînement ET en prévision).

    delta_moy : progression moyenne passée ; si absente, on la remplace par la dernière progression.
    """
    lag1 = np.atleast_1d(np.asarray(taux_lag1, dtype=float))
    lag2 = np.atleast_1d(np.asarray(taux_lag2, dtype=float))
    pop = np.atleast_1d(np.asarray(population, dtype=float))
    urb = np.atleast_1d(np.asarray(milieu_urbain, dtype=int))
    delta1 = lag1 - lag2
    moy = delta1 if delta_moy is None else np.atleast_1d(np.asarray(delta_moy, dtype=float))
    return pd.DataFrame(
        {
            "taux_lag1": lag1,
            "delta_lag1": delta1,
            "delta_moy": moy,
            "log_population": np.log1p(pop),
            "milieu_URBAIN": urb,
        },
        index=index,
    )[FEATURES]


def charger_panel() -> pd.DataFrame:
    from database import obtenir_donnees_wash

    df = obtenir_donnees_wash()
    print(f"📊 {len(df)} lignes ({df['commune_id'].nunique()} communes, "
          f"années {df['annee'].min()}-{df['annee'].max()}) chargées depuis wash_cleaned_data.")
    return df


def construire_paires(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne = une commune pour une année N (N >= 3e année), avec son historique."""
    d = df.sort_values(["commune_id", "annee"]).copy()

    manquants = d["milieu"].isna().sum()
    if manquants:
        print(f"⚠️ {manquants} lignes sans milieu : considérées RURAL.")
    d["milieu_URBAIN"] = (d["milieu"].astype(str).str.strip().str.upper() == "URBAIN").astype(int)

    g = d.groupby("commune_id")
    d["taux_lag1"] = g["taux_assainissement"].shift(1)
    d["taux_lag2"] = g["taux_assainissement"].shift(2)
    d["annee_lag2"] = g["annee"].shift(2)
    d["taux_debut"] = g["taux_assainissement"].transform("first")
    d["annee_debut"] = g["annee"].transform("first")
    # progression moyenne passée = (taux N-1 - taux de la 1re année) / nombre d'années écoulées
    d["delta_moy"] = (d["taux_lag1"] - d["taux_debut"]) / (d["annee"] - 1 - d["annee_debut"]).replace(0, np.nan)

    # On ne garde que les années réellement consécutives
    paires = d[(d["annee"] - d["annee_lag2"]) == 2].copy()
    X = construire_features(
        paires["taux_lag1"], paires["taux_lag2"], paires["population"],
        paires["milieu_URBAIN"], index=paires.index, delta_moy=paires["delta_moy"],
    )
    out = paires[["commune_id", "nom_region", "nom_district", "nom_commune", "annee", "population"]].copy()
    out = out.join(X)
    out["taux"] = paires["taux_assainissement"]
    out["delta"] = out["taux"] - out["taux_lag1"]          # cible
    return out.reset_index(drop=True)


def prepare_matrices(annee_test=None, panel=None):
    """Découpage TEMPOREL : entraînement = années avant annee_test, test = annee_test.

    Le scaler est ajusté sur l'entraînement seulement (pas de fuite).
    Retourne un dictionnaire de matrices et de métadonnées.
    panel : tableau déjà chargé (sinon lecture dans PostgreSQL).
    """
    paires = construire_paires(charger_panel() if panel is None else panel)
    paires = paires[paires["annee"] <= ANNEE_MAX_MODELE].reset_index(drop=True)
    annees = sorted(paires["annee"].unique())
    annee_test = annee_test or annees[-1]
    if len(annees) < 2:
        raise ValueError("Il faut au moins 2 années de paires pour entraîner et tester.")

    train = paires[paires["annee"] < annee_test].reset_index(drop=True)
    test = paires[paires["annee"] == annee_test].reset_index(drop=True)
    print(f"🔹 Entraînement : années {sorted(train['annee'].unique())} ({len(train)} lignes) | "
          f"Test : {annee_test} ({len(test)} lignes)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_test = scaler.transform(test[FEATURES])

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": train["delta"].to_numpy(), "y_test": test["delta"].to_numpy(),
        "annees_train": train["annee"].to_numpy(),
        "annee_test": int(annee_test),
        "lag1_test": test["taux_lag1"].to_numpy(),
        "delta_lag1_test": test["delta_lag1"].to_numpy(),
        "delta_moy_test": test["delta_moy"].to_numpy(),
        "taux_test": test["taux"].to_numpy(),
        "scaler": scaler,
        "paires": paires,
    }


def matrices_finales(paires: pd.DataFrame):
    """Modèle de production : toutes les paires disponibles. Sauvegarde le scaler."""
    scaler = StandardScaler()
    X = scaler.fit_transform(paires[FEATURES])
    joblib.dump(scaler, SCALER_PATH)
    return X, paires["delta"].to_numpy(), scaler