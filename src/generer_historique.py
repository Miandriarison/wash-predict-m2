"""Génère les années 2015-2023 (DONNÉES SIMULÉES) à partir de la vraie année 2024.

Méthode (à documenter dans le mémoire) :
- Population : on remonte le temps depuis 2024 avec une croissance de 2,5 %/an en moyenne, variable
  par commune et par année (écart-type 0,3 point, bornée entre 2,0 % et 3,0 %).
- Taux d'assainissement : on remonte le temps commune par commune. L'écart d'une année à l'autre
  = pas national moyen x facteur propre à la commune + bruit aléatoire. Un coefficient commun est
  ajusté pour que le taux national (pondéré par la population) colle aux valeurs cibles lues sur
  le graphique de référence (années 2022-2024 quasi plates, puis saut en 2025).
- Bénéficiaires = taux x population (arrondi) ; le taux est recalculé à partir des bénéficiaires.
- Graine aléatoire fixe : résultats reproductibles.

Utilisation : python src/generer_historique.py data/SEAM_2024.xlsx data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CROISSANCE_MOY, CROISSANCE_SD = 0.025, 0.003
# Taux national cible (en %, pondéré par la population).
# 2022-2023 : lus sur le graphique de référence. 2015-2021 : HYPOTHÈSES (trajectoire plausible, pas des données).
CIBLES = {2023: 39.0, 2022: 38.0, 2021: 37.5, 2020: 36.8,
          2019: 35.8, 2018: 34.9, 2017: 34.0, 2016: 33.0, 2015: 32.0}
ANNEES = sorted(CIBLES, reverse=True)  # 2023 -> 2015 : on remonte le temps depuis 2024
BRUIT_SD = 0.008        # bruit annuel par commune (en fraction : 0,8 point)
HETEROGENEITE_SD = 0.6  # dispersion du facteur propre à chaque commune
TAUX_MAX = 0.99


def taux_national(taux, pop):
    return float((taux * pop).sum() / pop.sum())


def generer(d24: pd.DataFrame, bruit_sd: float = BRUIT_SD, seed: int = SEED) -> dict:
    """Retourne {annee: tableau} pour 2023 à 2015 à partir du tableau réel de 2024."""
    rng = np.random.default_rng(seed)
    cles = ["REGION", "DISTRICT", "COMMUNE", "MILIEU"]
    n = len(d24)
    facteur = rng.lognormal(0.0, HETEROGENEITE_SD, n)  # propre à chaque commune, fixe dans le temps

    pop = d24["POPULATION"].to_numpy(float)
    taux = d24["TAUX"].to_numpy(float)
    resultat = {}

    for annee in ANNEES:
        g = np.clip(rng.normal(CROISSANCE_MOY, CROISSANCE_SD, n), 0.02, 0.03)
        pop_prec = np.round(pop / (1 + g))
        bruit = rng.normal(0.0, bruit_sd, n)
        cible = CIBLES[annee] / 100

        def candidat(k):
            return np.clip(taux - k * facteur + bruit, 0.0, TAUX_MAX)

        # Recherche (dichotomie) du coefficient k qui donne le taux national voulu
        bas, haut = -0.2, 0.2
        for _ in range(60):
            k = (bas + haut) / 2
            if taux_national(candidat(k), pop_prec) > cible:
                bas = k
            else:
                haut = k
        taux_prec = candidat((bas + haut) / 2)

        benef = np.round(taux_prec * pop_prec).astype(int)
        taux_prec = benef / pop_prec
        out = d24[cles].copy()
        out["ANNEE"] = annee
        out["POPULATION"] = pop_prec
        out["BENEFICIAIRES_TOTAL"] = benef
        out["TAUX"] = taux_prec
        resultat[annee] = out
        pop, taux = pop_prec, taux_prec
    return resultat


def main(chemin_2024, dossier_sortie):
    d24 = pd.read_excel(chemin_2024)
    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    for annee, out in sorted(generer(d24).items()):
        with pd.ExcelWriter(sortie / f"SEAM_{annee}.xlsx") as w:
            out.to_excel(w, sheet_name=f"DONNEES_{annee}", index=False)
        print(annee, "taux national", round(100 * taux_national(out["TAUX"].to_numpy(), out["POPULATION"].to_numpy()), 2),
              "| population", int(out["POPULATION"].sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])