import numpy as np
import pandas as pd
from sqlalchemy import text

from src.database import get_db_engine


def load_raw_wash_data(engine) -> pd.DataFrame:
    """Extrait les données combinées depuis PostgreSQL."""
    query = """
        SELECT 
            c.commune_id,
            r.nom_region,
            r.nom_district,
            c.nom_commune,
            c.milieu,
            iw.annee,
            iw.population,
            iw.menages_latrine_amelioree_partagee,
            iw.menages_latrine_amelioree_non_partagee,
            iw.beneficiaires_latrine_basique_partage,
            iw.beneficiaires_latrine_basique_non_partage,
            iw.beneficiaires_latrine_basique_total,
            iw.odf_2023,
            iw.odf_2024,
            iw.taux_assainissement
        FROM indicateurs_wash iw
        JOIN communes c ON c.commune_id = iw.commune_id
        JOIN regions r ON r.region_id = c.region_id;
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def clean_and_impute_wash_data(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les algorithmes de nettoyage et d'imputation des données."""
    print("⏳ [1/3] Traitement des incohérences et valeurs aberrantes...")
    df_clean = df.copy()

    # 1. Traitement de la population (remplacer <= 0 par NaN pour imputation)
    df_clean["population"] = df_clean["population"].apply(
        lambda x: np.nan if pd.isna(x) or x <= 0 else x
    )

    # 2. Corriger les taux d'assainissement hors limites [0, 100]
    if "taux_assainissement" in df_clean.columns:
        df_clean["taux_assainissement"] = df_clean[
            "taux_assainissement"
        ].apply(lambda x: np.nan if pd.isna(x) or x < 0 or x > 100 else x)

    print("⏳ [2/3] Imputation des valeurs manquantes par médiane (District)...")

    # Imputation de la POPULATION par la médiane du District, puis de la Région
    df_clean["population"] = df_clean["population"].fillna(
        df_clean.groupby("nom_district")["population"].transform("median")
    )
    df_clean["population"] = df_clean["population"].fillna(
        df_clean.groupby("nom_region")["population"].transform("median")
    )

    # Imputation des bénéficiaires totaux si manquants
    df_clean["beneficiaires_latrine_basique_total"] = df_clean[
        "beneficiaires_latrine_basique_total"
    ].fillna(
        df_clean.groupby("nom_district")[
            "beneficiaires_latrine_basique_total"
        ].transform("median")
    )

    # Correction des bénéficiaires s'ils dépassent la population
    df_clean["beneficiaires_latrine_basique_total"] = np.minimum(
        df_clean["beneficiaires_latrine_basique_total"], df_clean["population"]
    )

    # Recalcul ou Imputation du TAUX D'ASSAINISSEMENT (%)
    # Si le taux est manquant, on le calcule : (Bénéficiaires / Population) * 100
    calc_taux = (
        df_clean["beneficiaires_latrine_basique_total"]
        / df_clean["population"]
    ) * 100
    df_clean["taux_assainissement"] = df_clean["taux_assainissement"].fillna(
        calc_taux
    )

    # Si encore des NaN dans le taux, impute par la médiane du District
    df_clean["taux_assainissement"] = df_clean["taux_assainissement"].fillna(
        df_clean.groupby("nom_district")["taux_assainissement"].transform(
            "median"
        )
    )

    # Remplacer d'éventuels NaN restants par 0 pour les effectifs
    cols_zero = [
        "menages_latrine_amelioree_partagee",
        "menages_latrine_amelioree_non_partagee",
        "beneficiaires_latrine_basique_partage",
        "beneficiaires_latrine_basique_non_partage",
    ]
    df_clean[cols_zero] = df_clean[cols_zero].fillna(0)

    # Arrondir les colonnes d'effectifs
    int_cols = [
        "population",
        "beneficiaires_latrine_basique_total",
        "menages_latrine_amelioree_partagee",
        "menages_latrine_amelioree_non_partagee",
    ]
    df_clean[int_cols] = df_clean[int_cols].round().astype(int)
    df_clean["taux_assainissement"] = df_clean["taux_assainissement"].round(2)

    print("✅ Nettoyage et imputation terminés avec succès !")
    return df_clean


def save_cleaned_data(df: pd.DataFrame, engine):
    """Sauvegarde les données nettoyées dans la table PostgreSQL 'wash_cleaned_data'."""
    print("⏳ [3/3] Sauvegarde dans la table 'wash_cleaned_data'...")
    df.to_sql(
        name="wash_cleaned_data",
        con=engine,
        if_exists="replace",
        index=False,
    )
    print("🚀 Données nettoyées enregistrées en base !")


def run_cleaning_pipeline():
    engine = get_db_engine()
    print("📥 Extraction des données brutes depuis PostgreSQL...")
    df_raw = load_raw_wash_data(engine)
    print(f"📊 Données chargées : {len(df_raw)} communes.")

    df_clean = clean_and_impute_wash_data(df_raw)
    save_cleaned_data(df_clean, engine)


if __name__ == "__main__":
    run_cleaning_pipeline()