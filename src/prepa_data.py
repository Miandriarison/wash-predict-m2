import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def charger_et_preparer_donnees(df: pd.DataFrame):
    """Prépare la matrice de données X et la variable cible y (Taux d'accès %)."""
    df_prep = df.copy()

    # Encodage de la colonne 'milieu' si présente
    if "milieu" in df_prep.columns and "milieu_URBAIN" not in df_prep.columns:
        df_prep = pd.get_dummies(
            df_prep, columns=["milieu"], drop_first=False, dtype=int
        )

    # Vérification de l'existence de la colonne 'milieu_URBAIN'
    if "milieu_URBAIN" not in df_prep.columns:
        df_prep["milieu_URBAIN"] = 0

    # Features (variables explicatives)
    features = [
        "population",
        "beneficiaires_latrine_basique_total",
        "menages_latrine_amelioree_partagee",
        "menages_latrine_amelioree_non_partagee",
        "milieu_URBAIN",
    ]

    X = df_prep[features]
    y = df_prep["taux_assainissement"]

    # Séparation Train (80%) / Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisation Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sauvegarde du scaler
    joblib.dump(scaler, "scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns