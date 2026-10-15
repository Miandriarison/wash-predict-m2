"""Analyse exploratoire du panel nettoyé (statistiques, évolution, régions)."""

import pandas as pd

from database import obtenir_donnees_wash

B = "beneficiaires_latrine_basique_total"


def analyser(df: pd.DataFrame):
    pd.set_option("display.float_format", lambda x: "%.2f" % x)

    print("==================================================")
    print("📊 STATISTIQUES DESCRIPTIVES GLOBALES")
    print("==================================================")
    stats_globales = df[["population", B, "taux_assainissement"]].describe()
    print(stats_globales)

    print("\n==================================================")
    print("📈 ÉVOLUTION NATIONALE PAR ANNÉE")
    print("==================================================")
    par_annee = df.groupby("annee").apply(
        lambda g: pd.Series(
            {
                "nb_communes": len(g),
                "taux_moyen": g["taux_assainissement"].mean(),
                "taux_median": g["taux_assainissement"].median(),
                "taux_national_pondere": g[B].sum() / g["population"].sum() * 100,
                "population_totale": g["population"].sum(),
            }
        ),
        include_groups=False,
    )
    par_annee["variation_pts"] = par_annee["taux_national_pondere"].diff()
    print(par_annee.to_string())

    derniere = df["annee"].max()
    print(f"\n==================================================")
    print(f"🗺️ TAUX PAR RÉGION ({derniere})")
    print("==================================================")
    stats_region = (
        df[df["annee"] == derniere]
        .groupby("nom_region")["taux_assainissement"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    stats_region.columns = ["REGION", "NB_COMMUNES", "MOYENNE", "MEDIANE", "ECART_TYPE", "MIN", "MAX"]
    print(stats_region.to_string(index=False))

    print("\n==================================================")
    print("🏘️ PANEL : communes présentes sur N années")
    print("==================================================")
    print(df.groupby("commune_id")["annee"].nunique().value_counts().sort_index().to_string())

    return stats_globales, par_annee, stats_region


def run_descriptive_analysis():
    df = obtenir_donnees_wash()
    print(f"✅ {len(df)} lignes ({df['commune_id'].nunique()} communes) chargées.\n")
    return analyser(df)


if __name__ == "__main__":
    run_descriptive_analysis()