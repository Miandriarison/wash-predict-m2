"""Recherche des meilleurs hyperparamètres du MLP (validation temporelle), puis évaluation finale."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

from model import executer
from prepa_data import prepare_matrices

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def optimiser():
    print("🚀 Optimisation du MLPRegressor (GridSearchCV, validation temporelle)...")
    d = prepare_matrices()

    # Validation temporelle : on entraîne sur les années anciennes, on valide sur la dernière
    # année d'entraînement (l'année de test reste totalement invisible).
    annees = np.unique(d["annees_train"])
    if len(annees) < 2:
        raise ValueError("Pas assez d'années pour une validation temporelle.")
    idx_train = np.where(d["annees_train"] < annees[-1])[0]
    idx_val = np.where(d["annees_train"] == annees[-1])[0]
    print(f"   Validation : entraîne sur {list(annees[:-1])}, valide sur {annees[-1]}")

    param_grid = {
        "hidden_layer_sizes": [(32, 16), (64, 32), (64, 32, 16)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [1000],
        "tol": [1e-4],
        "early_stopping": [True],
        "n_iter_no_change": [20],
    }
    gs = GridSearchCV(
        MLPRegressor(random_state=42),
        param_grid,
        cv=[(idx_train, idx_val)],
        scoring="r2",
        n_jobs=-1,
        refit=False,
    )
    gs.fit(d["X_train"], d["y_train"])

    best = dict(gs.best_params_)
    print(f"✅ Meilleurs hyperparamètres : {best}")
    print(f"   R² de validation (sur l'évolution annuelle) : {gs.best_score_:.4f}")

    best["random_state"] = 42
    infos = {
        "meilleurs_parametres": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()},
        "r2_validation_temporelle": round(float(gs.best_score_), 4),
    }
    return executer(best, "optimize.py (GridSearchCV)", infos)


if __name__ == "__main__":
    optimiser()