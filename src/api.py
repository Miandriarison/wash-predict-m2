import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def entrainer_et_evaluer_mlp(X_train, X_test, y_train, y_test):
    print("🚀 Début de l'optimisation du MLPRegressor (GridSearchCV)...")
    
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (100, 50), (64, 32, 16)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }

    mlp = MLPRegressor(random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"✅ Meilleurs hyperparamètres : {grid_search.best_params_}")

    # Prédictions sur le jeu de test
    y_pred = best_model.predict(X_test)

    # Métriques exigées par le Cahier des Charges (Section 5.6)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 --- PERFORMANCES DU MLP REGRESSOR ---")
    print(f"🎯 MAE  (Erreur Absolue Moyenne) : {mae:.2f}%")
    print(f"🎯 RMSE (Racine Erreur Quadratique) : {rmse:.2f}%")
    print(f"🎯 R²   (Coefficient de Détermination) : {r2:.4f}")

    # Comparaison avec la Régression Linéaire (Baseline - Section 5.6)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    print("\n📈 --- COMPARAISON BASELINE (Régression Linéaire) ---")
    print(f"🔹 MAE Régression Linéaire : {mae_lr:.2f}%")
    print(f"🔹 R²  Régression Linéaire : {r2_lr:.4f}")

    # Sauvegarde du modèle
    joblib.dump(best_model, 'mlp_model.joblib')
    print("\n💾 Modèle MLPRegressor sauvegardé sous 'mlp_model.joblib'.")

    return best_model