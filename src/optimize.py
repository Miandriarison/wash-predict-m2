import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

# Filtrer les avertissements de convergence pour garder une console propre
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def entrainer_et_evaluer_mlp(X_train, X_test, y_train, y_test):
    print("🚀 Début de l'optimisation du MLPRegressor (GridSearchCV)...")

    param_grid = {
        "hidden_layer_sizes": [(64, 32), (100, 50), (64, 32, 16)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [1000],
        "tol": [1e-3],  # Tolérance réaliste pour valider la convergence
        "early_stopping": [True],
        "n_iter_no_change": [15],
    }

    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(
        mlp, param_grid, cv=5, scoring="r2", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"✅ Meilleurs hyperparamètres : {grid_search.best_params_}")

    # Prédictions sur le jeu de test
    y_pred = best_model.predict(X_test)

    # Métriques (Section 5.6)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 --- PERFORMANCES DU MLP REGRESSOR ---")
    print(f"🎯 MAE  (Erreur Absolue Moyenne) : {mae:.2f}%")
    print(f"🎯 RMSE (Racine Erreur Quadratique) : {rmse:.2f}%")
    print(f"🎯 R²   (Coefficient de Détermination) : {r2:.4f}")

    # Baseline Régression Linéaire
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    print("\n📈 --- COMPARAISON BASELINE (Régression Linéaire) ---")
    print(f"🔹 MAE Régression Linéaire : {mae_lr:.2f}%")
    print(f"🔹 R²  Régression Linéaire : {r2_lr:.4f}")

    # Sauvegarde du modèle
    joblib.dump(best_model, "mlp_model.joblib")
    print("\n💾 Modèle MLPRegressor sauvegardé sous 'mlp_model.joblib'.")

    return best_model