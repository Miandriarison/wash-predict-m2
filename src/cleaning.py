"""Nettoyage et imputation du panel commune x année -> table 'wash_cleaned_data'."""

import numpy as np
import pandas as pd

from database import get_db_engine

B = "beneficiaires_latrine_basique_total"


def load_raw_wash_data(engine) -> pd.DataFrame:
    """Extrait les données (une ligne par commune et par année) depuis PostgreSQL."""
    query = """
        SELECT
            c.commune_id,
            r.nom_region,
            r.nom_district,
            c.nom_commune,
            c.milieu,
            iw.annee,
            iw.population,
            iw.beneficiaires_latrine_basique_total,
            iw.taux_assainissement
        FROM indicateurs_wash iw
        JOIN communes c ON c.commune_id = iw.commune_id
        JOIN regions  r ON r.region_id  = c.region_id
        ORDER BY c.commune_id, iw.annee;
    """
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


def clean_and_impute_wash_data(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage, imputation (dans le temps puis par zone) et recalcul du taux."""
    print("⏳ [1/3] Incohérences et valeurs aberrantes...")
    d = df.sort_values(["commune_id", "annee"]).reset_index(drop=True).copy()
    n_initial = len(d)

    for c in ["population", B, "taux_assainissement"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d.loc[d["population"] <= 0, "population"] = np.nan
    d.loc[(d["taux_assainissement"] < 0) | (d["taux_assainissement"] > 100), "taux_assainissement"] = np.nan

    print("⏳ [2/3] Imputation (interpolation par commune, puis médiane district/région de l'année)...")
    # Population : interpolation linéaire dans le temps pour chaque commune
    d["population"] = d.groupby("commune_id")["population"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    d["population"] = d["population"].fillna(d.groupby(["nom_district", "annee"])["population"].transform("median"))
    d["population"] = d["population"].fillna(d.groupby(["nom_region", "annee"])["population"].transform("median"))

    # Bénéficiaires manquants : déduits du taux publié
    d[B] = d[B].fillna(d["taux_assainissement"] * d["population"] / 100)
    d[B] = np.minimum(d[B], d["population"])  # ne peut pas dépasser la population

    # Taux manquant : recalculé depuis les composantes
    calc_taux = d[B] / d["population"] * 100
    ecarts = ((d["taux_assainissement"] - calc_taux).abs() > 1).sum()
    print(f"   ℹ️ {ecarts} lignes où le taux publié s'écarte de bénéficiaires/population de plus de 1 point.")
    d["taux_assainissement"] = d["taux_assainissement"].fillna(calc_taux)

    # Cible impossible à établir : suppression contrôlée (on n'invente pas la cible)
    avant = len(d)
    d = d.dropna(subset=["taux_assainissement", "population", B]).copy()
    if len(d) < avant:
        print(f"   ⚠️ {avant - len(d)} lignes supprimées (taux, population ou bénéficiaires impossibles à établir).")

    d[["population", B]] = d[["population", B]].round().astype(int)
    d["taux_assainissement"] = d["taux_assainissement"].round(2)

    manquants = d["milieu"].isna().sum()
    if manquants:
        print(f"   ⚠️ {manquants} lignes sans milieu.")

    print(f"✅ Nettoyage terminé : {len(d)}/{n_initial} lignes conservées.")
    return d.reset_index(drop=True)


def save_cleaned_data(df: pd.DataFrame, engine):
    print("⏳ [3/3] Sauvegarde dans 'wash_cleaned_data'...")
    df.to_sql(name="wash_cleaned_data", con=engine, if_exists="replace", index=False)
    print("🚀 Données nettoyées enregistrées en base !")


def verifier_donnees_nettoyees(engine):
    with engine.connect() as conn:
        resume = pd.read_sql(
            """
            SELECT annee,
                   COUNT(*) AS nb_communes,
                   ROUND(AVG(taux_assainissement)::numeric, 2) AS taux_moyen,
                   ROUND((100.0 * SUM(beneficiaires_latrine_basique_total) / SUM(population))::numeric, 2)
                       AS taux_national_pondere
            FROM wash_cleaned_data
            GROUP BY annee ORDER BY annee;
            """,
            conn,
        )
    print("📊 Contrôle par année :")
    print(resume.to_string(index=False))


def run_cleaning_pipeline():
    engine = get_db_engine()
    print("📥 Extraction depuis PostgreSQL...")
    df_raw = load_raw_wash_data(engine)
    print(f"📊 {len(df_raw)} lignes chargées.")
    df_clean = clean_and_impute_wash_data(df_raw)
    save_cleaned_data(df_clean, engine)
    verifier_donnees_nettoyees(engine)


if __name__ == "__main__":
    run_cleaning_pipeline()