"""Analyse de sensibilité : les résultats dépendent-ils du niveau de bruit des années simulées ?

Pour chaque niveau de bruit (écart-type de la variation annuelle propre à chaque commune), on
regénère 2020-2023 à partir de la vraie année 2024 (plusieurs graines), on entraîne le MLP et les
méthodes de comparaison sur 2022-2023, et on les teste sur 2024. On rapporte la MOYENNE et l'écart-type
des erreurs sur les graines. Aucun réglage n'est fait pour favoriser une méthode.

Utilisation (depuis la racine du projet) :
    python src/sensibilite_bruit.py data/SEAM_2024.xlsx
Sorties : sensibilite_bruit.csv et sensibilite_bruit.png à la racine du projet.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from generer_historique import generer
from model import evaluer_panel
from prepa_data import ROOT_DIR

warnings.filterwarnings("ignore", category=ConvergenceWarning)

NIVEAUX_BRUIT = [0.0, 0.004, 0.008, 0.016, 0.03]   # en fraction : 0,4 point = 0.004
GRAINES = [1, 2, 3, 4, 5]


def construire_panel(d24: pd.DataFrame, annees_simulees: dict) -> pd.DataFrame:
    """Tableau 2020-2024 au format attendu par prepa_data (taux en %)."""
    tous = pd.concat([*annees_simulees.values(), d24], ignore_index=True)
    ids = tous.groupby(["REGION", "DISTRICT", "COMMUNE"], sort=False).ngroup()
    return pd.DataFrame({
        "commune_id": ids, "nom_region": tous["REGION"], "nom_district": tous["DISTRICT"],
        "nom_commune": tous["COMMUNE"], "milieu": tous["MILIEU"], "annee": tous["ANNEE"].astype(int),
        "population": tous["POPULATION"], "taux_assainissement": tous["TAUX"] * 100,
    })


def main(chemin_2024):
    d24 = pd.read_excel(chemin_2024)
    lignes = []
    for bruit in NIVEAUX_BRUIT:
        for graine in GRAINES:
            panel = construire_panel(d24, generer(d24, bruit_sd=bruit, seed=graine))
            for methode, m in evaluer_panel(panel).items():
                lignes.append({"bruit_points": bruit * 100, "graine": graine, "methode": methode,
                               "MAE": m["MAE"], "R2_evolution": m["R2_evolution"]})
        print(f"✅ bruit {bruit * 100:.1f} point(s) terminé")

    df = pd.DataFrame(lignes)
    res = df.groupby(["bruit_points", "methode"]).agg(
        MAE=("MAE", "mean"), MAE_ecart_type=("MAE", "std"),
        R2_evolution=("R2_evolution", "mean")).round(3).reset_index()
    res.to_csv(ROOT_DIR / "sensibilite_bruit.csv", index=False, encoding="utf-8-sig")

    print("\n📊 MAE moyenne (points de %) selon le bruit annuel par commune :")
    print(res.pivot(index="methode", columns="bruit_points", values="MAE").to_string())
    print("\n📊 R² de l'évolution annuelle (moyenne) :")
    print(res.pivot(index="methode", columns="bruit_points", values="R2_evolution").to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for methode, g in res.groupby("methode"):
        style = dict(lw=3, color="#e74c3c") if "MLP" in methode else dict(lw=1.4)
        axes[0].plot(g["bruit_points"], g["MAE"], marker="o", label=methode, **style)
        axes[1].plot(g["bruit_points"], g["R2_evolution"], marker="o", label=methode, **style)
    axes[0].set_ylabel("MAE sur 2024 (points de %)")
    axes[1].set_ylabel("R² de l'évolution annuelle")
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    for ax in axes:
        ax.set_xlabel("Bruit annuel par commune (points de %)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sensibilité au bruit des années simulées (moyenne sur 5 graines)")
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "sensibilite_bruit.png", dpi=120)
    print(f"\n💾 sensibilite_bruit.csv et sensibilite_bruit.png enregistrés dans {ROOT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1])
