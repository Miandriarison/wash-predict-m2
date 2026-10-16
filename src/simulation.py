"""Logique métier de l'application web (indépendante de FastAPI et de Streamlit).

Ce module assemble les briques de forecast.py pour répondre à UNE requête de simulation :
territoire choisi (Madagascar, une région ou une commune) + paramètres (années, objectif, scénario,
population, latrines) -> séries de courbes, indicateurs, besoin sur le terrain, tableau et avertissements.

Tout est renvoyé sous forme de dictionnaires JSON-compatibles (pas de NaN, pas de types numpy), prêts à être
envoyés par l'API puis dessinés par le front (Chart.js).
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from forecast import (
    EFFETS_MAX,
    LIBELLES_SCENARIO,
    PERSONNES_PAR_LATRINE,
    besoin_terrain,
    charger_modele,
    effet_scenario,
    preparer_depart,
    prevision_lineaire_nationale,
    prevoir_communes,
    retro_prediction,
    taux_avec_latrines,
    taux_pondere,
    trajectoire_objectif,
)
from prepa_data import ROOT_DIR

NATIONAL, REGION, COMMUNE = "national", "region", "commune"
NOMS_CAS = ["Maintenir le taux actuel", "Tendance (MLP)", "Avec scénario", "Atteindre l'objectif"]
IMAGES_PERFORMANCE = {
    "predit_vs_reel": "predit_vs_reel.png",
    "courbes_perte": "courbes_perte.png",
    "comparaison_modeles": "comparaison_modeles.png",
}
LIMITES = [
    "Les années 2015-2023 et 2025 sont simulées à partir des données réelles de 2024 "
    "(population ≈ +2,5 %/an, progression propre à chaque commune avec une part aléatoire) ; "
    "seules les données 2024 sont réelles.",
    "2025 est une rupture (+19 points) : elle est exclue de l'entraînement et sert de point de départ des prévisions.",
    "Sur ces données, le MLP fait à peu près aussi bien que les méthodes simples (persistance, tendance nationale) : "
    "la progression annuelle y est en grande partie aléatoire, aucune méthode ne la prédit mieux qu'une moyenne.",
    "L'effet de la sensibilisation, de l'éducation et des campagnes de santé est une hypothèse paramétrable : "
    "ces variables n'existent pas dans la base.",
    "Population : +2,5 % par an par défaut, modifiable.",
]


# ---------------------------------------------------------------------------
# Contexte (données + modèle chargés une seule fois)
# ---------------------------------------------------------------------------
@dataclass
class Contexte:
    panel: pd.DataFrame
    model: object
    scaler: object
    dep: pd.DataFrame
    annee_depart: int
    _cache: dict = field(default_factory=dict)

    @classmethod
    def depuis_panel(cls, panel: pd.DataFrame, model, scaler) -> "Contexte":
        dep, annee_depart = preparer_depart(panel)
        return cls(panel=panel, model=model, scaler=scaler, dep=dep, annee_depart=annee_depart)

    @classmethod
    def depuis_base(cls) -> "Contexte":
        """Charge le panel depuis PostgreSQL (database.obtenir_donnees_wash) et le modèle depuis le disque."""
        from database import obtenir_donnees_wash

        model, scaler = charger_modele()
        return cls.depuis_panel(obtenir_donnees_wash(), model, scaler)

    def previsions(self, nb_annees: int, effet: float, croissance: float) -> pd.DataFrame:
        """Prévisions de TOUTES les communes (mises en cache : les curseurs reviennent souvent aux mêmes valeurs)."""
        cle = (int(nb_annees), round(float(effet), 6), round(float(croissance), 6))
        if cle not in self._cache:
            if len(self._cache) >= 64:
                self._cache.clear()
            self._cache[cle] = prevoir_communes(
                self.dep, self.annee_depart, self.model, self.scaler, nb_annees, effet, croissance
            )
        return self._cache[cle]


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def _f(v, nd=2):
    """Nombre JSON propre : None pour NaN/None, float arrondi sinon."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(v), nd)


def _entier(v):
    return int(round(float(v)))


def meta(ctx: Contexte) -> dict:
    """Informations pour construire le formulaire (régions, scénarios, valeurs par défaut)."""
    return {
        "annee_depart": ctx.annee_depart,
        "regions": sorted(ctx.panel["nom_region"].dropna().unique().tolist()),
        "scenarios": [
            {"cle": k, "libelle": LIBELLES_SCENARIO[k], "effet_max": EFFETS_MAX[k]} for k in EFFETS_MAX
        ],
        "defauts": {
            "nb_annees": 4,
            "objectif_final": 90.0,
            "croissance_pct": 2.5,
            "personnes_par_latrine": PERSONNES_PAR_LATRINE,
        },
    }


def communes(ctx: Contexte, region: str) -> list:
    sub = ctx.panel[ctx.panel["nom_region"] == region].drop_duplicates("commune_id")
    if sub.empty:
        raise KeyError(f"Région inconnue : {region}")
    sub = sub.assign(label=sub["nom_commune"] + " — " + sub["nom_district"]).sort_values("label")
    return [{"id": int(i), "label": l} for i, l in zip(sub["commune_id"], sub["label"])]


def _selection(ctx: Contexte, type_territoire: str, valeur):
    """-> (ids des communes, libellé du territoire)."""
    p = ctx.panel
    if type_territoire == NATIONAL:
        return p["commune_id"].unique(), "Madagascar (toutes les communes)"
    if type_territoire == REGION:
        ids = p.loc[p["nom_region"] == valeur, "commune_id"].unique()
        if len(ids) == 0:
            raise KeyError(f"Région inconnue : {valeur}")
        return ids, str(valeur)
    if type_territoire == COMMUNE:
        try:
            cid = int(valeur)
        except (TypeError, ValueError):
            raise ValueError("Identifiant de commune invalide")
        ligne = p[p["commune_id"] == cid].drop_duplicates("commune_id")
        if ligne.empty:
            raise KeyError(f"Commune inconnue : {cid}")
        r = ligne.iloc[0]
        return np.array([cid]), f"{r['nom_commune']} — {r['nom_district']}"
    raise ValueError(f"Type de territoire inconnu : {type_territoire}")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simuler(ctx: Contexte, type_territoire: str = NATIONAL, valeur=None, nb_annees: int = 4,
            objectif_final: float = 90.0, sensibilisation: float = 0.0, education: float = 0.0,
            campagnes_sante: float = 0.0, croissance_pct: float = 2.5,
            personnes_par_latrine: float = PERSONNES_PAR_LATRINE, latrines_par_an: float = 0.0) -> dict:
    # --- validation ---
    if not 1 <= int(nb_annees) <= 10:
        raise ValueError("nb_annees doit être entre 1 et 10")
    if not 0 <= objectif_final <= 100:
        raise ValueError("objectif_final doit être entre 0 et 100")
    for nom, v in (("sensibilisation", sensibilisation), ("education", education), ("campagnes_sante", campagnes_sante)):
        if not 0 <= v <= 1:
            raise ValueError(f"{nom} doit être entre 0 et 1")
    if not 0 <= croissance_pct <= 10:
        raise ValueError("croissance_pct doit être entre 0 et 10")
    if personnes_par_latrine < 1:
        raise ValueError("personnes_par_latrine doit être >= 1")
    if latrines_par_an < 0:
        raise ValueError("latrines_par_an doit être >= 0")

    nb_annees = int(nb_annees)
    a0 = ctx.annee_depart
    annee_cible = a0 + nb_annees
    croissance = croissance_pct / 100
    effet = effet_scenario(
        {"sensibilisation": sensibilisation, "education": education, "campagnes_sante": campagnes_sante}
    )

    ids, libelle = _selection(ctx, type_territoire, valeur)
    base = ctx.previsions(nb_annees, 0.0, croissance)
    scen = ctx.previsions(nb_annees, effet, croissance)
    base_s, scen_s = base[base["commune_id"].isin(ids)], scen[scen["commune_id"].isin(ids)]
    dep_s = ctx.dep[ctx.dep["commune_id"].isin(ids)]
    h = ctx.panel[ctx.panel["commune_id"].isin(ids)]

    # --- séries ---
    if type_territoire == COMMUNE:
        hist = h.sort_values("annee").set_index("annee")["taux_assainissement"]
        pb = base_s.set_index("annee")["taux"]
        ps = scen_s.set_index("annee")["taux"]
        lin = None
        retro = retro_prediction(ctx.model, ctx.scaler, h.sort_values("annee"))
        retro = retro.set_index("annee")["taux_predit"] if len(retro) else None
    else:
        hist = taux_pondere(h.assign(taux=lambda d: d["taux_assainissement"]), "taux")
        pb, ps = taux_pondere(base_s, "taux"), taux_pondere(scen_s, "taux")
        lin = prevision_lineaire_nationale(hist, a0, nb_annees)
        retro = None
    taux0 = float(hist.loc[a0])
    obj = trajectoire_objectif(taux0, objectif_final, a0, nb_annees)

    lat = None
    if latrines_par_an > 0:
        lat = taux_avec_latrines(dep_s, base_s, latrines_par_an, a0, personnes_par_latrine)

    axe = sorted(set(hist.index) | set(range(a0 + 1, annee_cible + 1)))

    def aligner(serie, prolonger=False):
        if serie is None:
            return None
        s = serie
        if prolonger:  # la prévision démarre sur la dernière valeur réelle (courbe continue)
            s = pd.concat([pd.Series({a0: taux0}), serie])
        return [_f(s.get(a)) if a in s.index else None for a in axe]

    series = {
        "realisation": aligner(hist),
        "prevision": aligner(pb, True),
        "scenario": aligner(ps, True) if effet > 0 else None,
        "lineaire": aligner(lin, True),
        "objectif": aligner(obj),
        "retro": aligner(retro),
        "latrines": aligner(lat),
    }

    # --- tableau (vue région / nationale) : l'objectif démarre en a0, il n'est pas une cible pour cette année ---
    tableau = None
    if type_territoire != COMMUNE:
        tableau = []
        for i, a in enumerate(axe):
            tableau.append({
                "annee": int(a),
                "realisation": series["realisation"][i],
                "prevision": None if a <= a0 else series["prevision"][i],
                "scenario": None if (a <= a0 or series["scenario"] is None) else series["scenario"][i],
                "lineaire": None if a <= a0 else series["lineaire"][i],
                "objectif": None if a <= a0 else series["objectif"][i],
            })

    # --- besoin sur le terrain ---
    t = besoin_terrain(dep_s, base_s, scen_s, obj, a0, personnes_par_latrine)
    f = t.iloc[-1]
    besoin = {
        "annee_cible": annee_cible,
        "population_cible": _entier(f["Population"]),
        "cas": [
            {"nom": nom, "personnes": _entier(f[f"{nom} — personnes"]), "latrines": _entier(f[f"{nom} — latrines"])}
            for nom in NOMS_CAS
        ],
        "detail": [
            {"annee": int(a), "population": _entier(r["Population"]),
             **{f"{nom}|personnes": _entier(r[f"{nom} — personnes"]) for nom in NOMS_CAS},
             **{f"{nom}|latrines": _entier(r[f"{nom} — latrines"]) for nom in NOMS_CAS}}
            for a, r in t.iterrows()
        ],
        "simulation_latrines": None if lat is None else {
            "latrines_par_an": float(latrines_par_an),
            "taux_cible": _f(lat.iloc[-1], 1),
            "variation_pts": _f(lat.iloc[-1] - float(lat.iloc[0]), 1),
        },
    }

    # --- avertissements ---
    avertissements = []
    if objectif_final < taux0:
        avertissements.append(
            f"L'objectif ({objectif_final:g} %) est inférieur au taux actuel ({taux0:.1f} %) : ".replace(".", ",") +
            "aucune latrine supplémentaire n'est nécessaire pour l'atteindre."
        )
    nb_plafonnees = int((dep_s["taux_depart"] >= 99.99).sum())
    if nb_plafonnees and type_territoire != COMMUNE:
        avertissements.append(
            f"{nb_plafonnees} commune(s) déjà à 100 % en {a0} : elles ne progressent plus, "
            "ce qui ralentit la hausse du taux."
        )

    return {
        "territoire": {"type": type_territoire, "libelle": libelle, "nb_communes": int(len(ids))},
        "annee_depart": a0,
        "annee_cible": annee_cible,
        "effet": _f(effet, 4),
        "annees": [int(a) for a in axe],
        "series": series,
        "indicateurs": {
            "realisation_depart": _f(taux0, 1),
            "prevision_cible": _f(pb.iloc[-1], 1),
            "scenario_cible": _f(ps.iloc[-1], 1),
            "gain_scenario_pts": _f(ps.iloc[-1] - pb.iloc[-1], 1),
            "ecart_objectif_pts": _f(ps.iloc[-1] - objectif_final, 1),
        },
        "besoin": besoin,
        "tableau": tableau,
        "avertissements": avertissements,
    }


# ---------------------------------------------------------------------------
# Performance du modèle
# ---------------------------------------------------------------------------
def performance(racine=ROOT_DIR) -> Optional[dict]:
    """Contenu de metrics.json (None si le modèle n'a pas encore été entraîné) + limites à citer."""
    chemin = racine / "metrics.json"
    if not chemin.exists():
        return None
    data = json.loads(chemin.read_text(encoding="utf-8"))
    return {
        "infos": data.get("infos", {}),
        "modeles": data.get("modeles", {}),
        "images": {cle: (racine / nom).exists() for cle, nom in IMAGES_PERFORMANCE.items()},
        "limites": LIMITES,
    }
