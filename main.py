# main.py
from src.database import obtenir_donnees_wash
from src.prepa_data import charger_et_preparer_donnees
from src.optimize import entrainer_et_evaluer_mlp

def main():
    print("=== DÉMARRAGE DU PIPELINE ML - REGRESSION TAUX D'ACCÈS WASH ===")

    # 1. Chargement de la base consolidée depuis PostgreSQL
    df = obtenir_donnees_wash()

    # 2. Préparation des matrices (Train/Test + Scaling)
    X_train, X_test, y_train, y_test, feature_names = charger_et_preparer_donnees(df)

    # 3. Entraînement MLPRegressor & Évaluation
    best_model = entrainer_et_evaluer_mlp(X_train, X_test, y_train, y_test)

    print("\n✅ PIPELINE TERMINÉ AVEC SUCCÈS !")

if __name__ == "__main__":
    main()