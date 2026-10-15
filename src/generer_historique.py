"""Génère les années 2020-2023 (DONNÉES SIMULÉES) à partir de la vraie année 2024.

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
# Taux national cible (en %, pondéré par la population). 2020-2021 : choisis, pas lus sur le graphique.
CIBLES = {2023: 39.0, 2022: 38.0, 2021: 37.5, 2020: 36.8}
BRUIT_SD = 0.008        # bruit annuel par commune (en fraction : 0,8 point)
HETEROGENEITE_SD = 0.6  # dispersion du facteur propre à chaque commune
TAUX_MAX = 0.99


def taux_national(taux, pop):
    return float((taux * pop).sum() / pop.sum())


def main(chemin_2024, dossier_sortie):
    rng = np.random.default_rng(SEED)
    d24 = pd.read_excel(chemin_2024)
    cles = ["REGION", "DISTRICT", "COMMUNE", "MILIEU"]
    n = len(d24)
    facteur = rng.lognormal(0.0, HETEROGENEITE_SD, n)  # propre à chaque commune, fixe dans le temps

    pop = d24["POPULATION"].to_numpy(float)
    taux = d24["TAUX"].to_numpy(float)
    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    for annee in (2023, 2022, 2021, 2020):
        g = np.clip(rng.normal(CROISSANCE_MOY, CROISSANCE_SD, n), 0.02, 0.03)
        pop_prec = np.round(pop / (1 + g))
        bruit = rng.normal(0.0, BRUIT_SD, n)
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
        with pd.ExcelWriter(sortie / f"SEAM_{annee}.xlsx") as w:
            out.to_excel(w, sheet_name=f"DONNEES_{annee}", index=False)
        print(annee, "taux national", round(100 * taux_national(taux_prec, pop_prec), 2),
              "| population", int(pop_prec.sum()))
        pop, taux = pop_prec, taux_prec


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
