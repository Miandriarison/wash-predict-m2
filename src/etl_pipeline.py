"""ETL : un fichier Excel par année (2020-2024) -> staging_wash_data -> tables relationnelles.

Colonnes attendues dans chaque fichier :
REGION | DISTRICT | COMMUNE | MILIEU | ANNEE | POPULATION | NOMBRE DE BENEFICIAIRE LATRINE FAMILIALE BASIQUE | TAUX
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from database import get_db_engine

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CLES = ["REGION", "DISTRICT", "COMMUNE"]
COLONNES_FINALES = CLES + [
    "MILIEU", "ANNEE", "POPULATION", "BENEFICIAIRES_TOTAL", "TAUX_ASSAINISSEMENT",
]


# ==========================================
# 1. FONCTIONS DE NETTOYAGE
# ==========================================
def _normaliser(v):
    if pd.isna(v):
        return None
    v = unicodedata.normalize("NFKD", str(v)).encode("ascii", errors="ignore").decode("utf-8")
    return re.sub(r"\s+", " ", v).strip().upper()


def clean_text(series: pd.Series) -> pd.Series:
    """MAJUSCULES, sans accents, espaces normalisés. Les vides restent None (pas 'NAN')."""
    return series.map(_normaliser)


def standardiser_milieu(series: pd.Series) -> pd.Series:
    def _m(v):
        v = _normaliser(v)
        if v is None:
            return None
        if v.startswith("URB"):
            return "URBAIN"
        if v.startswith("RUR"):
            return "RURAL"
        return None

    return series.map(_m)


def parse_taux(series: pd.Series) -> pd.Series:
    """Convertit la colonne Taux en pourcentage (0-100).

    '45,56%' -> 45.56 ; nombre Excel au format % (0.4556) -> 45.56 (détecté pour la colonne).
    """

    def _parse(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        s = str(v).replace("%", "").replace("\xa0", "").replace(",", ".").strip()
        try:
            return float(s)
        except ValueError:
            return np.nan

    out = series.map(_parse).astype(float)
    est_nombre = series.map(lambda v: isinstance(v, (int, float, np.number)) and not pd.isna(v))
    if est_nombre.any() and out[est_nombre].max() <= 1.0:
        print("   ℹ️ Taux lus comme fractions Excel (max <= 1) : conversion x100.")
        out[est_nombre] = out[est_nombre] * 100
    return out.round(2)


def _renommer_colonnes(colonnes) -> dict:
    """Associe les en-têtes du fichier aux noms standard (insensible aux accents/espaces)."""
    exact = {"REGION", "DISTRICT", "COMMUNE", "MILIEU", "ANNEE", "POPULATION"}
    mapping = {}
    for c in colonnes:
        n = _normaliser(c) or ""
        if n.startswith("NOMBRE DE BENEFICIAIRE"):
            mapping[c] = "BENEFICIAIRES_TOTAL"
        elif n.startswith("TAUX"):
            mapping[c] = "TAUX_ASSAINISSEMENT"
        elif n in exact:
            mapping[c] = n
    return mapping


# ==========================================
# 2. LECTURE DES FICHIERS
# ==========================================
def lire_fichier(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path)
    df = df.rename(columns=_renommer_colonnes(df.columns))

    manque = set(COLONNES_FINALES) - set(df.columns)
    if manque:
        raise ValueError(f"{path.name} : colonnes introuvables {sorted(manque)}. Présentes : {list(df.columns)}")
    if df.columns.duplicated().any():
        raise ValueError(f"{path.name} : plusieurs colonnes correspondent au même champ : {list(df.columns)}")

    df = df[COLONNES_FINALES].dropna(subset=["REGION", "COMMUNE"]).copy()
    for c in CLES:
        df[c] = clean_text(df[c])
    df["MILIEU"] = standardiser_milieu(df["MILIEU"])

    df["ANNEE"] = pd.to_numeric(df["ANNEE"], errors="coerce").round().astype("Int64")
    for c in ("POPULATION", "BENEFICIAIRES_TOTAL"):
        df[c] = pd.to_numeric(df[c], errors="coerce").round().astype("Int64")

    df["TAUX_ASSAINISSEMENT"] = parse_taux(df["TAUX_ASSAINISSEMENT"])
    hors = ((df["TAUX_ASSAINISSEMENT"] < 0) | (df["TAUX_ASSAINISSEMENT"] > 100)).sum()
    if hors:
        print(f"   ⚠️ {hors} taux hors [0, 100] mis à NULL (recalculés au nettoyage).")
        df.loc[(df["TAUX_ASSAINISSEMENT"] < 0) | (df["TAUX_ASSAINISSEMENT"] > 100), "TAUX_ASSAINISSEMENT"] = np.nan

    annees = df["ANNEE"].dropna().unique().tolist()
    print(f"   📄 {path.name} : {len(df)} lignes, année(s) {annees}, taux moyen {df['TAUX_ASSAINISSEMENT'].mean():.1f} %")
    return df


def charger_tous_les_fichiers(dossier=DATA_DIR) -> pd.DataFrame:
    fichiers = sorted(p for p in Path(dossier).glob("*.xlsx") if not p.name.startswith("~$"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier .xlsx dans {dossier}")

    print(f"⏳ [1/3] Lecture de {len(fichiers)} fichiers Excel...")
    df = pd.concat([lire_fichier(p) for p in fichiers], ignore_index=True)

    if df["ANNEE"].isna().any():
        raise ValueError(f"{df['ANNEE'].isna().sum()} lignes sans ANNEE valide.")

    # Doublons (même commune, même année)
    cle_ligne = CLES + ["ANNEE"]
    n_dup = df.duplicated(subset=cle_ligne).sum()
    if n_dup:
        print(f"   ⚠️ {n_dup} doublons (région/district/commune/année) supprimés (première ligne gardée).")
        df = df.drop_duplicates(subset=cle_ligne, keep="first")

    # Milieu unique par commune (valeur la plus fréquente sur les années)
    def _mode(s):
        m = s.dropna().mode()
        return m.iloc[0] if len(m) else None

    df["MILIEU"] = df.groupby(CLES)["MILIEU"].transform(_mode)
    return df.sort_values(CLES + ["ANNEE"]).reset_index(drop=True)


def diagnostic_panel(df: pd.DataFrame):
    """Vérifie que les mêmes communes se retrouvent d'une année à l'autre (noms stables)."""
    annees = sorted(df["ANNEE"].unique())
    par_commune = df.groupby(CLES)["ANNEE"].nunique()
    print(f"\n📊 Années : {annees} | communes distinctes : {len(par_commune)}")
    print("   Communes présentes sur N années :")
    print(par_commune.value_counts().sort_index().to_string())
    rares = par_commune[par_commune < len(annees)]
    if len(rares):
        print(f"   ⚠️ {len(rares)} communes absentes d'au moins une année (orthographe différente ?). Exemples :")
        print(rares.head(10).to_string())


# ==========================================
# 3. DISPATCH VERS LES TABLES RELATIONNELLES
# ==========================================
def dispatch_to_relational_tables(engine: Engine):
    print("⏳ [3b/3] Dispatching staging -> tables finales...")

    query_dispatch = text("""
        -- A. Régions & Districts
        INSERT INTO regions (nom_region, nom_district)
        SELECT DISTINCT "REGION", "DISTRICT"
        FROM staging_wash_data
        WHERE "REGION" IS NOT NULL AND "DISTRICT" IS NOT NULL
        ON CONFLICT (nom_region, nom_district) DO NOTHING;

        -- B. Communes
        INSERT INTO communes (region_id, nom_commune, milieu)
        SELECT r.region_id, s."COMMUNE", MAX(s."MILIEU")
        FROM staging_wash_data s
        JOIN regions r ON r.nom_region = s."REGION" AND r.nom_district = s."DISTRICT"
        WHERE s."COMMUNE" IS NOT NULL
        GROUP BY r.region_id, s."COMMUNE"
        ON CONFLICT (region_id, nom_commune) DO UPDATE
            SET milieu = COALESCE(EXCLUDED.milieu, communes.milieu);

        -- C. Indicateurs WASH (une ligne par commune et par année)
        INSERT INTO indicateurs_wash (
            commune_id, annee, population,
            beneficiaires_latrine_basique_total, taux_assainissement
        )
        SELECT DISTINCT ON (c.commune_id, s."ANNEE")
            c.commune_id,
            s."ANNEE"::INT,
            s."POPULATION"::INT,
            s."BENEFICIAIRES_TOTAL"::INT,
            s."TAUX_ASSAINISSEMENT"::NUMERIC(5,2)
        FROM staging_wash_data s
        JOIN regions  r ON r.nom_region = s."REGION" AND r.nom_district = s."DISTRICT"
        JOIN communes c ON c.region_id = r.region_id AND c.nom_commune = s."COMMUNE"
        ORDER BY c.commune_id, s."ANNEE"
        ON CONFLICT (commune_id, annee) DO UPDATE SET
            population                          = EXCLUDED.population,
            beneficiaires_latrine_basique_total = EXCLUDED.beneficiaires_latrine_basique_total,
            taux_assainissement                 = EXCLUDED.taux_assainissement;
    """)

    with engine.begin() as conn:
        conn.execute(query_dispatch)
    print("✨ Tables finales remplies.")


# ==========================================
# 4. PIPELINE PRINCIPAL
# ==========================================
def run_etl_pipeline(dossier=DATA_DIR, db_engine: Engine = None) -> pd.DataFrame:
    if db_engine is None:
        db_engine = get_db_engine()

    df = charger_tous_les_fichiers(dossier)
    diagnostic_panel(df)

    print("\n⏳ [2/3] Insertion dans PostgreSQL (staging_wash_data)...")
    df.to_sql(name="staging_wash_data", con=db_engine, if_exists="replace", index=False)
    dispatch_to_relational_tables(db_engine)

    print("🚀 Pipeline ETL exécuté avec succès !")
    return df


if __name__ == "__main__":
    run_etl_pipeline()