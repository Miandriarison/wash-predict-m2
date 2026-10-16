"""Outils communs aux tests : un petit jeu de données synthétique et un MLP entraîné dessus (sans PostgreSQL)."""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import prepa_data  # noqa: E402
from simulation import Contexte  # noqa: E402

_cache = {}


def panel_synthetique(n_communes=60, seed=0) -> pd.DataFrame:
    """Communes réparties en 3 régions, années 2015-2025 ; 2025 = saut de +19 points (comme dans le projet)."""
    rng = np.random.default_rng(seed)
    lignes = []
    for c in range(1, n_communes + 1):
        region = f"REGION {1 + c % 3}"
        taux = rng.uniform(10, 95)
        pente = rng.uniform(0.3, 1.2)
        pop = rng.uniform(5_000, 60_000)
        for annee in range(2015, 2026):
            if annee > 2015:
                taux += pente + rng.normal(0, 0.5) + (19 if annee == 2025 else 0)
                pop *= 1.025
            taux = float(np.clip(taux, 0, 100))
            lignes.append({
                "commune_id": c, "nom_region": region, "nom_district": f"DISTRICT {c % 5}",
                "nom_commune": f"COMMUNE {c}", "milieu": "URBAIN" if c % 4 == 0 else "RURAL",
                "annee": annee, "population": round(pop), "taux_assainissement": round(taux, 2),
            })
    return pd.DataFrame(lignes)


def contexte_test() -> Contexte:
    """Contexte (panel + MLP entraîné) construit une seule fois pour tous les tests."""
    if "ctx" not in _cache:
        panel = panel_synthetique()
        with redirect_stdout(io.StringIO()):  # prepare_matrices est bavard
            d = prepa_data.prepare_matrices(panel=panel)
        paires = d["paires"]
        scaler = StandardScaler()
        X = scaler.fit_transform(paires[prepa_data.FEATURES])
        mlp = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=300, random_state=0).fit(X, paires["delta"])
        _cache["ctx"] = Contexte.depuis_panel(panel, mlp, scaler)
    return _cache["ctx"]
