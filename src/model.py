"""Entraînement, évaluation et comparaison du MLP pour la prédiction du taux (dans le temps)."""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

from prepa_data import FEATURES, ROOT_DIR, matrices_finales, prepare_matrices

MODEL_PATH = ROOT_DIR / "mlp_model.joblib"
METRICS_PATH = ROOT_DIR / "metrics.json"
LOSS_CURVE_PATH = ROOT_DIR / "courbes_perte.png"
PRED_PLOT_PATH = ROOT_DIR / "predit_vs_reel.png"
COMPARE_PLOT_PATH = ROOT_DIR / "comparaison_modeles.png"

PARAMS_PAR_DEFAUT = dict(
    hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
    max_iter=500, early_stopping=True, n_iter_no_change=20, random_state=42,
)


def predire_taux(lag1, delta_pred):
    """Taux prédit = taux N-1 + évolution prédite, borné entre 0 et 100."""
    return np.clip(np.asarray(lag1) + np.asarray(delta_pred), 0.0, 100.0)


def calculer_metriques(d, delta_pred) -> dict:
    taux_pred = predire_taux(d["lag1_test"], delta_pred)
    mse = mean_squared_error(d["taux_test"], taux_pred)
    return {
        "MSE": round(float(mse), 4),
        "RMSE": round(float(np.sqrt(mse)), 4),
        "MAE": round(float(mean_absolute_error(d["taux_test"], taux_pred)), 4),
        "R2": round(float(r2_score(d["taux_test"], taux_pred)), 4),
        # R² sur l'ÉVOLUTION annuelle : mesure plus exigeante (le niveau du taux est déjà connu)
        "R2_evolution": round(float(r2_score(d["y_test"], delta_pred)), 4),
    }


def comparer_modeles(mlp, d) -> dict:
    """MLP face aux méthodes classiques, sur l'année de test."""
    n = len(d["y_test"])
    lr = LinearRegression().fit(d["X_train"], d["y_train"])
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(d["X_train"], d["y_train"])

    predictions = {
        "Tendance nationale (méthode actuelle)": np.full(n, d["y_train"].mean()),
        "Persistance (taux N = taux N-1)": np.zeros(n),
        "Tendance linéaire par commune": d["delta_lag1_test"],
        "Régression linéaire (apprise)": lr.predict(d["X_test"]),
        "Random Forest": rf.predict(d["X_test"]),
        "MLP (RNA)": mlp.predict(d["X_test"]),
    }
    return {nom: calculer_metriques(d, p) for nom, p in predictions.items()}


def tracer_graphiques(mlp, d, resultats):
    # 1. Courbes de perte
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mlp.loss_curve_, label="Perte (train)")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Perte (MSE / 2)")
    ax.set_title("Courbe de perte du MLP")
    if getattr(mlp, "validation_scores_", None) is not None:
        ax2 = ax.twinx()
        ax2.plot(mlp.validation_scores_, color="orange", label="R² validation")
        ax2.set_ylabel("R² validation")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(LOSS_CURVE_PATH, dpi=120)
    plt.close(fig)

    # 2. Prédit vs réel (taux de l'année de test)
    pred = predire_taux(d["lag1_test"], mlp.predict(d["X_test"]))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d["taux_test"], pred, s=8, alpha=0.4)
    ax.plot([0, 100], [0, 100], "r--", label="Prédiction parfaite")
    ax.set_xlabel(f"Taux réel {d['annee_test']} (%)")
    ax.set_ylabel("Taux prédit (%)")
    ax.set_title(f"MLP : prédit vs réel ({d['annee_test']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PRED_PLOT_PATH, dpi=120)
    plt.close(fig)

    # 3. Comparaison des méthodes (MAE)
    noms = list(resultats)
    maes = [resultats[n]["MAE"] for n in noms]
    fig, ax = plt.subplots(figsize=(8, 4))
    couleurs = ["#e74c3c" if "MLP" in n else "#95a5a6" for n in noms]
    ax.barh(noms, maes, color=couleurs)
    ax.set_xlabel("Erreur absolue moyenne (points de %)")
    ax.set_title(f"Comparaison des méthodes ({d['annee_test']})")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(COMPARE_PLOT_PATH, dpi=120)
    plt.close(fig)


def executer(params: dict, source: str, infos_supplementaires: dict = None):
    """Évalue sur l'année de test, puis réentraîne sur toutes les données (modèle de production)."""
    d = prepare_matrices()

    print("\n🧠 Entraînement du MLPRegressor (évaluation)...")
    mlp = MLPRegressor(**params).fit(d["X_train"], d["y_train"])
    print(f"✅ {mlp.n_iter_} itérations (perte finale : {mlp.loss_:.4f})")

    resultats = comparer_modeles(mlp, d)
    print(f"\n📊 Performances sur {d['annee_test']} (jamais vue à l'entraînement) :")
    print(f"{'Méthode':42s} {'MAE':>7s} {'RMSE':>7s} {'R²':>8s} {'R² évol.':>9s}")
    for nom, m in resultats.items():
        print(f"{nom:42s} {m['MAE']:7.2f} {m['RMSE']:7.2f} {m['R2']:8.4f} {m['R2_evolution']:9.4f}")

    meilleur = min(resultats, key=lambda k: resultats[k]["MAE"])
    if meilleur != "MLP (RNA)":
        print(f"\nℹ️ Meilleure méthode (MAE) : {meilleur}. À discuter honnêtement dans le mémoire.")

    tracer_graphiques(mlp, d, resultats)

    # Modèle de production : toutes les paires (dont la dernière année)
    X_all, y_all, _ = matrices_finales(d["paires"])
    mlp_final = MLPRegressor(**params).fit(X_all, y_all)
    joblib.dump(mlp_final, MODEL_PATH)

    infos = {
        "source": source,
        "annee_test": d["annee_test"],
        "variables": FEATURES,
        "hidden_layer_sizes": list(params["hidden_layer_sizes"]),
        "nb_couches_cachees": len(params["hidden_layer_sizes"]),
        "activation": params["activation"],
        "n_train": int(len(d["y_train"])),
        "n_test": int(len(d["y_test"])),
        "modele_de_production": "réentraîné sur toutes les années disponibles",
    }
    infos.update(infos_supplementaires or {})
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump({"modeles": resultats, "infos": infos}, f, ensure_ascii=False, indent=2)

    print(f"\n💾 {MODEL_PATH.name}, scaler.joblib, {METRICS_PATH.name} et 3 graphiques (.png) enregistrés.")
    return mlp_final, resultats


def train_mlp_model():
    return executer(PARAMS_PAR_DEFAUT, "model.py (baseline)")


if __name__ == "__main__":
    train_mlp_model()