import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.database import get_db_engine


def load_cleaned_data(engine) -> pd.DataFrame:
    """Charge les données nettoyées depuis PostgreSQL."""
    query = "SELECT * FROM wash_cleaned_data;"
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def prepare_matrices():
    """Prépare les matrices X, y, applique le One-Hot Encoding et le Scaling."""
    engine = get_db_engine()
    print("📥 Extraction des données depuis 'wash_cleaned_data'...")
    df = load_cleaned_data(engine)

    # 1. Sélection de la cible (y) et des features (X)
    # y = odf_2024 (converti en entier : True -> 1, False -> 0)
    y = df["odf_2024"].astype(int)

    # Features à utiliser
    feature_cols = [
        "population",
        "beneficiaires_latrine_basique_total",
        "menages_latrine_amelioree_partagee",
        "menages_latrine_amelioree_non_partagee",
        "taux_assainissement",
        "milieu",
    ]

    X = df[feature_cols].copy()

    # 2. Encodage de la colonne catégorielle 'milieu' (One-Hot Encoding)
    X = pd.get_dummies(X, columns=["milieu"], drop_first=True)

    # Convertir les booléens créés par get_dummies en entiers (0/1)
    X = X.astype(float)

    print(f"📊 Features sélectionnées ({X.shape[1]} colonnes) : {list(X.columns)}")

    # 3. Train / Test Split (80% train, 20% test) avec stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Normalisation (Scaling) avec StandardScaler
    scaler = StandardScaler()

    # On ajuste le scaler UNIQUEMENT sur X_train pour éviter le Data Leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Sauvegarde du Scaler pour la future API (Phase 4)
    joblib.dump(scaler, "scaler.joblib")
    print("💾 Scaler sauvegardé sous 'scaler.joblib'.")

    # Résumé des shapes
    print("\n✅ Matrice de données prête avec succès !")
    print(f"🔹 X_train : {X_train_scaled.shape} | y_train : {y_train.shape}")
    print(f"🔹 X_test  : {X_test_scaled.shape}  | y_test  : {y_test.shape}")
    print(
        f"🔹 Ratio ODF dans Train : {np.mean(y_train):.2%} | Test : {np.mean(y_test):.2%}"
    )

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


if __name__ == "__main__":
    prepare_matrices()