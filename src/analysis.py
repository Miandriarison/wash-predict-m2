import pandas as pd
from database import get_db_engine


def run_descriptive_analysis():
    engine = get_db_engine()

    print("📥 Extraction des données nettoyées depuis 'wash_cleaned_data'...")
    with engine.connect() as conn:
        df = pd.read_sql("SELECT * FROM wash_cleaned_data;", conn)

    print(f"✅ {len(df)} communes chargées avec succès.\n")

    # 1. Statistiques descriptives globales
    print("==================================================")
    print("📊 STATISTIQUES DESCRIPTIVES GLOBALES (.describe())")
    print("==================================================")
    numeric_cols = [
        "population",
        "beneficiaires_latrine_basique_total",
        "menages_latrine_amelioree_partagee",
        "menages_latrine_amelioree_non_partagee",
        "taux_assainissement",
    ]

    # Formatage propre des nombres décimaux
    pd.set_option("display.float_format", lambda x: "%.2f" % x)
    stats_globales = df[numeric_cols].describe()
    print(stats_globales)

    # 2. Statistiques par Région (Focus sur le Taux d'assainissement)
    print("\n==================================================")
    print("🗺️ TAUX D'ASSAINISSEMENT MOYEN ET MÉDIAN PAR RÉGION")
    print("==================================================")
    stats_region = (
        df.groupby("nom_region")["taux_assainissement"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    stats_region.columns = [
        "REGION",
        "NB_COMMUNES",
        "MOYENNE (%)",
        "MÉDIANE (%)",
        "ÉCART-TYPE",
        "MIN (%)",
        "MAX (%)",
    ]
    print(stats_region.sort_values(by="MOYENNE (%)", ascending=False).to_string(index=False))

    # 3. Répartition des communes ODF en 2024
    print("\n==================================================")
    print("🚩 RÉPARTITION DES COMMUNES CERTIFIÉES ODF (2024)")
    print("==================================================")
    print(df["odf_2024"].value_counts(dropna=False))

    return stats_globales, stats_region


if __name__ == "__main__":
    run_descriptive_analysis()
    