import re
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Import de la connexion centralisée
from src.database import get_db_engine


# ==========================================
# 1. FONCTIONS DE NETTOYAGE (TEXTE & TAUX)
# ==========================================
def clean_text(series: pd.Series) -> pd.Series:
    """Standardise les noms géographiques (MAJUSCULES, sans espaces superflus)."""
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )


def clean_percentage(val) -> float:
    """Convertit '45,5588707826486%' ou 45.5588 en float (ex: 0.46)."""
    if pd.isna(val):
        return None
    val_str = str(val).replace(",", ".").replace("%", "").strip()
    try:
        return round(float(val_str), 2)
    except ValueError:
        return None


# ==========================================
# 2. DISPATCH VERS LES TABLES RELATIONNELLES
# ==========================================
def dispatch_to_relational_tables(engine: Engine):
    """Exécute les requêtes d'insertion depuis staging vers le schéma relationnel final."""
    print(
        "⏳ [4b/4] Dispatching des données depuis staging vers les 3 tables finales..."
    )

    query_dispatch = text("""
        -- A. Insertion Régions & Districts
        INSERT INTO regions (nom_region, nom_district)
        SELECT DISTINCT "REGION", "DISTRICT"
        FROM staging_wash_data
        WHERE "REGION" IS NOT NULL AND "DISTRICT" IS NOT NULL
        ON CONFLICT (nom_region, nom_district) DO NOTHING;

        -- B. Insertion Communes
        INSERT INTO communes (region_id, nom_commune, milieu)
        SELECT DISTINCT 
            r.region_id, 
            s."COMMUNE", 
            s."MILIEU"
        FROM staging_wash_data s
        JOIN regions r ON r.nom_region = s."REGION" AND r.nom_district = s."DISTRICT"
        WHERE s."COMMUNE" IS NOT NULL
        ON CONFLICT (region_id, nom_commune) DO NOTHING;

        -- C. Insertion Indicateurs WASH
        INSERT INTO indicateurs_wash (
            commune_id, 
            annee, 
            population, 
            menages_latrine_amelioree_partagee, 
            menages_latrine_amelioree_non_partagee, 
            beneficiaires_latrine_basique_partage, 
            beneficiaires_latrine_basique_non_partage, 
            beneficiaires_latrine_basique_total, 
            odf_2023, 
            odf_2024, 
            taux_assainissement
        )
        SELECT 
            c.commune_id,
            2024 AS annee,
            s."POPULATION"::INT,
            COALESCE(s."MENAGES_LATRINE_PARTAGEE"::INT, 0),
            COALESCE(s."MENAGES_LATRINE_NON_PARTAGEE"::INT, 0),
            COALESCE(s."BENEFICIAIRES_PARTAGE"::INT, 0),
            COALESCE(s."BENEFICIAIRES_NON_PARTAGE"::INT, 0),
            COALESCE(s."BENEFICIAIRES_TOTAL"::INT, 0),
            CASE WHEN s."ODF_2023" = 'OUI' THEN TRUE ELSE FALSE END,
            CASE WHEN s."ODF_2024" = 'OUI' THEN TRUE ELSE FALSE END,
            s."TAUX_ASSAINISSEMENT"::NUMERIC(5,2)
        FROM staging_wash_data s
        JOIN regions r ON r.nom_region = s."REGION" AND r.nom_district = s."DISTRICT"
        JOIN communes c ON c.region_id = r.region_id AND c.nom_commune = s."COMMUNE"
        ON CONFLICT (commune_id, annee) DO NOTHING;
    """)

    with engine.begin() as conn:
        conn.execute(query_dispatch)

    print("✨ Transfert réussi ! Les tables finales sont remplies.")


# ==========================================
# 3. PIPELINE ETL PRINCIPAL
# ==========================================
def run_etl_pipeline(
    path_taux_2024: str, path_odf: str, db_engine: Engine = None
):
    if db_engine is None:
        db_engine = get_db_engine()

    print("⏳ [1/4] Chargement des fichiers Excel SE&AM...")
    df_taux = pd.read_excel(path_taux_2024)
    df_odf = pd.read_excel(path_odf)

    # Filtrer les lignes vides
    df_taux = df_taux.dropna(subset=["REGION", "COMMUNE"])
    df_odf = df_odf.dropna(subset=["REGION", "COMMUNE"])

    # Nettoyage des colonnes
    df_taux.columns = [
        re.sub(r"\s+", " ", col).strip() for col in df_taux.columns
    ]
    df_odf.columns = [re.sub(r"\s+", " ", col).strip() for col in df_odf.columns]

    print("⏳ [2/4] Mapping et standardisation des colonnes Taux et ODF...")
    rename_taux = {
        "POPULATION": "POPULATION",
        "Nombre de ménage disposant de latrine améliorée partagée 2024": "MENAGES_LATRINE_PARTAGEE",
        "Nombre de ménage disposant de latrine améliorée non partagée 2024": "MENAGES_LATRINE_NON_PARTAGEE",
        "NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE PARTAGE": "BENEFICIAIRES_PARTAGE",
        "NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE NON PARTAGE": "BENEFICIAIRES_NON_PARTAGE",
        "NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE": "BENEFICIAIRES_TOTAL",
        "Taux": "TAUX_ASSAINISSEMENT",
    }
    df_taux.rename(columns=rename_taux, inplace=True)

    rename_odf = {
        "CODE COMMUNE": "CODE_COMMUNE",
        "MILIEU": "MILIEU",
        "ODF_2023": "ODF_2023",
        "ODF_2024": "ODF_2024",
    }
    df_odf.rename(columns=rename_odf, inplace=True)

    for df in [df_taux, df_odf]:
        df["REGION"] = clean_text(df["REGION"])
        df["DISTRICT"] = clean_text(df["DISTRICT"])
        df["COMMUNE"] = clean_text(df["COMMUNE"])

    if "MILIEU" in df_odf.columns:
        df_odf["MILIEU"] = clean_text(df_odf["MILIEU"])

    if "TAUX_ASSAINISSEMENT" in df_taux.columns:
        df_taux["TAUX_ASSAINISSEMENT"] = df_taux["TAUX_ASSAINISSEMENT"].apply(
            clean_percentage
        )

    print("⏳ [3/4] Fusion (Merge) des datasets...")
    cols_odf_select = [
        c
        for c in [
            "CODE_COMMUNE",
            "REGION",
            "DISTRICT",
            "COMMUNE",
            "MILIEU",
            "ODF_2023",
            "ODF_2024",
        ]
        if c in df_odf.columns
    ]

    df_merged = pd.merge(
        df_taux,
        df_odf[cols_odf_select],
        on=["REGION", "DISTRICT", "COMMUNE"],
        how="left",
    )

    print(f"✅ Fusion réussie ! Lignes fusionnées : {len(df_merged)}")

    print("⏳ [4/4] Insertion dans PostgreSQL (staging_wash_data)...")
    df_merged.to_sql(
        name="staging_wash_data",
        con=db_engine,
        if_exists="replace",
        index=False,
    )

    # Dispatching avec le moteur passé en paramètre
    dispatch_to_relational_tables(db_engine)

    print("🚀 Pipeline ETL exécuté avec succès !")
    return df_merged


# ==========================================
# EXÉCUTION
# ==========================================
if __name__ == "__main__":
    PATH_TAUX = "./data/SEAM_Taux_2024.xlsx"
    PATH_ODF = "./data/SEAM_ODF_2023_2024.xlsx"

    # Récupération de l'engine centralisé
    engine = get_db_engine()
    df_resultat = run_etl_pipeline(PATH_TAUX, PATH_ODF, db_engine=engine)
    import re
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

# Import de la connexion DB centralisée
from src.database import get_db_engine


# ==========================================
# 1. FONCTIONS DE NETTOYAGE (TEXTE & TAUX)
# ==========================================
def clean_text(series: pd.Series) -> pd.Series:
    """Standardise les noms géographiques (MAJUSCULES, sans espaces superflus)."""
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )


def clean_percentage(val) -> float:
    """Convertit '45,5588707826486%' ou 45.5588 en float (ex: 0.46)."""
    if pd.isna(val):
        return None
    val_str = str(val).replace(",", ".").replace("%", "").strip()
    try:
        return round(float(val_str), 2)
    except ValueError:
        return None


# ==========================================
# 2. DISPATCH VERS LES TABLES RELATIONNELLES
# ==========================================
def dispatch_to_relational_tables(engine: Engine):
    """Exécute les requêtes d'insertion depuis staging vers le schéma relationnel final."""
    print(
        "⏳ [4b/4] Dispatching des données depuis staging vers les 3 tables"
        " finales..."
    )

    query_dispatch = text("""
        -- A. Insertion Régions & Districts
        INSERT INTO regions (nom_region, nom_district)
        SELECT DISTINCT "REGION", "DISTRICT"
        FROM staging_wash_data
        WHERE "REGION" IS NOT NULL AND "DISTRICT" IS NOT NULL
        ON CONFLICT (nom_region, nom_district) DO NOTHING;

        -- B. Insertion Communes
        INSERT INTO communes (region_id, nom_commune, milieu)
        SELECT DISTINCT 
            r.region_id, 
            s."COMMUNE", 
            s."MILIEU"
        FROM staging_wash_data s
        JOIN regions r ON r.nom_region = s."REGION" AND r.nom_district = s."DISTRICT"
        WHERE s."COMMUNE" IS NOT NULL
        ON CONFLICT (region_id, nom_commune) DO NOTHING;

        -- C. Insertion Indicateurs WASH
        INSERT INTO indicateurs_wash (
            commune_id, 
            annee, 
            population, 
            menages_latrine_amelioree_partagee, 
            menages_latrine_amelioree_non_partagee, 
            beneficiaires_latrine_basique_partage, 
            beneficiaires_latrine_basique_non_partage, 
            beneficiaires_latrine_basique_total, 
            odf_2023, 
            odf_2024, 
            taux_assainissement
        )
        SELECT 
            c.commune_id,
            2024 AS annee,
            s."POPULATION"::INT,
            COALESCE(s."MENAGES_LATRINE_PARTAGEE"::INT, 0),
            COALESCE(s."MENAGES_LATRINE_NON_PARTAGEE"::INT, 0),
            COALESCE(s."BENEFICIAIRES_PARTAGE"::INT, 0),
            COALESCE(s."BENEFICIAIRES_NON_PARTAGE"::INT, 0),
            COALESCE(s."BENEFICIAIRES_TOTAL"::INT, 0),
            CASE WHEN s."ODF_2023" = 'OUI' THEN TRUE ELSE FALSE END,
            CASE WHEN s."ODF_2024" = 'OUI' THEN TRUE ELSE FALSE END,
            s."TAUX_ASSAINISSEMENT"::NUMERIC(5,2)
        FROM staging_wash_data s
        JOIN regions r ON r.nom_region = s."REGION" AND r.nom_district = s."DISTRICT"
        JOIN communes c ON c.region_id = r.region_id AND c.nom_commune = s."COMMUNE"
        ON CONFLICT (commune_id, annee) DO NOTHING;
    """)

    with engine.begin() as conn:
        conn.execute(query_dispatch)

    print("✨ Transfert réussi ! Les tables finales sont remplies.")


# ==========================================
# 3. PIPELINE ETL PRINCIPAL
# ==========================================
def run_etl_pipeline(
    path_taux_2024: str, path_odf: str, db_engine: Engine = None
):
    if db_engine is None:
        db_engine = get_db_engine()

    print("⏳ [1/4] Chargement des fichiers Excel SE&AM...")
    df_taux = pd.read_excel(path_taux_2024)
    df_odf = pd.read_excel(path_odf)

    # Filtrer les lignes vides
    df_taux = df_taux.dropna(subset=["REGION", "COMMUNE"])
    df_odf = df_odf.dropna(subset=["REGION", "COMMUNE"])

    # Nettoyage des espaces dans les noms de colonnes
    df_taux.columns = [
        re.sub(r"\s+", " ", col).strip() for col in df_taux.columns
    ]
    df_odf.columns = [re.sub(r"\s+", " ", col).strip() for col in df_odf.columns]

    print("⏳ [2/4] Mapping et standardisation des colonnes Taux et ODF...")
    rename_taux = {
        "POPULATION": "POPULATION",
        "Nombre de ménage disposant de latrine améliorée partagée 2024": (
            "MENAGES_LATRINE_PARTAGEE"
        ),
        "Nombre de ménage disposant de latrine améliorée non partagée 2024": (
            "MENAGES_LATRINE_NON_PARTAGEE"
        ),
        "NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE PARTAGE": (
            "BENEFICIAIRES_PARTAGE"
        ),
        "NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE NON PARTAGE": (
            "BENEFICIAIRES_NON_PARTAGE"
        ),
        "NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE": (
            "BENEFICIAIRES_TOTAL"
        ),
        "Taux": "TAUX_ASSAINISSEMENT",
    }
    df_taux.rename(columns=rename_taux, inplace=True)

    rename_odf = {
        "CODE COMMUNE": "CODE_COMMUNE",
        "MILIEU": "MILIEU",
        "ODF_2023": "ODF_2023",
        "ODF_2024": "ODF_2024",
    }
    df_odf.rename(columns=rename_odf, inplace=True)

    for df in [df_taux, df_odf]:
        df["REGION"] = clean_text(df["REGION"])
        df["DISTRICT"] = clean_text(df["DISTRICT"])
        df["COMMUNE"] = clean_text(df["COMMUNE"])

    if "MILIEU" in df_odf.columns:
        df_odf["MILIEU"] = clean_text(df_odf["MILIEU"])

    if "TAUX_ASSAINISSEMENT" in df_taux.columns:
        df_taux["TAUX_ASSAINISSEMENT"] = df_taux["TAUX_ASSAINISSEMENT"].apply(
            clean_percentage
        )

    print("⏳ [3/4] Fusion (Merge) des datasets...")
    cols_odf_select = [
        c
        for c in [
            "CODE_COMMUNE",
            "REGION",
            "DISTRICT",
            "COMMUNE",
            "MILIEU",
            "ODF_2023",
            "ODF_2024",
        ]
        if c in df_odf.columns
    ]

    df_merged = pd.merge(
        df_taux,
        df_odf[cols_odf_select],
        on=["REGION", "DISTRICT", "COMMUNE"],
        how="left",
    )

    print(f"✅ Fusion réussie ! Lignes fusionnées : {len(df_merged)}")

    print("⏳ [4/4] Insertion dans PostgreSQL (staging_wash_data)...")
    df_merged.to_sql(
        name="staging_wash_data",
        con=db_engine,
        if_exists="replace",
        index=False,
    )

    # Dispatching dans les tables relationnelles
    dispatch_to_relational_tables(db_engine)

    print("🚀 Pipeline ETL exécuté avec succès !")
    return df_merged


# ==========================================
# EXÉCUTION DU PIPELINE
# ==========================================
if __name__ == "__main__":
    PATH_TAUX = "./data/SEAM_Taux_2024.xlsx"
    PATH_ODF = "./data/SEAM_ODF_2023_2024.xlsx"

    engine = get_db_engine()
    df_resultat = run_etl_pipeline(PATH_TAUX, PATH_ODF, db_engine=engine)

    print("\n--- APERÇU DES DONNÉES FUSIONNÉES ET NETTOYÉES ---")
    print(
        df_resultat[
            [
                "REGION",
                "COMMUNE",
                "POPULATION",
                "BENEFICIAIRES_TOTAL",
                "TAUX_ASSAINISSEMENT",
                "ODF_2024",
            ]
        ].head()
    )