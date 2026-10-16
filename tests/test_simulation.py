"""Tests de la logique de simulation (aucune base de données ni serveur web nécessaires).

Lancement :  python -m unittest discover -s tests -v      (ou : pytest tests)
"""

import json
import math
import unittest

import outils  # noqa: F401  (ajoute src/ au chemin d'import)
import simulation as S


def valeurs(serie):
    return [v for v in serie if v is not None]


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = outils.contexte_test()

    # ----- invariants des courbes
    def test_taux_toujours_entre_0_et_100(self):
        r = S.simuler(self.ctx, sensibilisation=1, education=1, campagnes_sante=1, nb_annees=5)
        for nom, serie in r["series"].items():
            if serie:
                for v in valeurs(serie):
                    self.assertTrue(0 <= v <= 100, f"{nom} hors de [0, 100] : {v}")

    def test_axe_des_annees(self):
        r = S.simuler(self.ctx, nb_annees=3)
        self.assertEqual(r["annees"][0], 2015)
        self.assertEqual(r["annees"][-1], 2028)
        self.assertEqual(r["annee_depart"], 2025)
        for serie in r["series"].values():
            if serie:
                self.assertEqual(len(serie), len(r["annees"]))

    def test_les_previsions_demarrent_sur_la_derniere_valeur_reelle(self):
        r = S.simuler(self.ctx)
        i = r["annees"].index(2025)
        self.assertEqual(r["series"]["prevision"][i], r["series"]["realisation"][i])
        self.assertEqual(r["series"]["objectif"][i], r["series"]["realisation"][i])

    def test_objectif_atteint_a_l_annee_finale(self):
        r = S.simuler(self.ctx, objectif_final=80, nb_annees=4)
        self.assertAlmostEqual(r["series"]["objectif"][-1], 80.0, places=2)

    # ----- scénarios
    def test_sans_scenario_pas_de_courbe_scenario(self):
        r = S.simuler(self.ctx)
        self.assertIsNone(r["series"]["scenario"])
        self.assertEqual(r["effet"], 0)
        self.assertEqual(r["indicateurs"]["gain_scenario_pts"], 0)

    def test_effet_du_scenario(self):
        r = S.simuler(self.ctx, sensibilisation=1, education=0.5, campagnes_sante=0.5)
        self.assertAlmostEqual(r["effet"], 0.30 + 0.075 + 0.075, places=4)  # 0,30 x 1 + 0,15 x 0,5 + 0,15 x 0,5
        self.assertGreaterEqual(r["indicateurs"]["scenario_cible"], r["indicateurs"]["prevision_cible"])
        self.assertGreater(r["indicateurs"]["gain_scenario_pts"], 0)

    def test_plus_d_effet_ne_fait_jamais_baisser_le_taux(self):
        faible = S.simuler(self.ctx, sensibilisation=0.2)["indicateurs"]["scenario_cible"]
        fort = S.simuler(self.ctx, sensibilisation=1.0)["indicateurs"]["scenario_cible"]
        self.assertGreaterEqual(fort, faible)

    # ----- territoires
    def test_national_region_commune(self):
        nat = S.simuler(self.ctx)
        reg = S.simuler(self.ctx, S.REGION, "REGION 1")
        com = S.simuler(self.ctx, S.COMMUNE, 1)
        self.assertEqual(nat["territoire"]["nb_communes"], 60)
        self.assertEqual(reg["territoire"]["nb_communes"], 20)
        self.assertEqual(com["territoire"]["nb_communes"], 1)
        self.assertGreater(nat["besoin"]["population_cible"], reg["besoin"]["population_cible"])
        self.assertIsNotNone(nat["tableau"])
        self.assertIsNone(com["tableau"])                 # pas de tableau pour une commune
        self.assertIsNotNone(com["series"]["retro"])      # prédictions sur le passé seulement pour une commune
        self.assertIsNone(nat["series"]["retro"])

    def test_tableau_sans_objectif_ni_prevision_en_annee_de_depart(self):
        ligne = next(l for l in S.simuler(self.ctx, sensibilisation=0.5)["tableau"] if l["annee"] == 2025)
        self.assertIsNotNone(ligne["realisation"])
        for cle in ("prevision", "scenario", "lineaire", "objectif"):
            self.assertIsNone(ligne[cle], cle)

    # ----- besoin sur le terrain
    def test_besoin_latrines_coherent_avec_personnes(self):
        r = S.simuler(self.ctx, personnes_par_latrine=5)
        for cas in r["besoin"]["cas"]:
            self.assertGreaterEqual(cas["personnes"], 0)
            self.assertLessEqual(abs(cas["latrines"] - cas["personnes"] / 5), 1)  # arrondi au-dessus

    def test_maintenir_le_taux_demande_moins_que_l_objectif_eleve(self):
        cas = {c["nom"]: c for c in S.simuler(self.ctx, objectif_final=95)["besoin"]["cas"]}
        self.assertLessEqual(cas["Maintenir le taux actuel"]["personnes"], cas["Atteindre l'objectif"]["personnes"])

    def test_sans_croissance_de_population(self):
        r = S.simuler(self.ctx, croissance_pct=0)
        pop0 = int(self.ctx.dep["pop_depart"].sum())
        self.assertAlmostEqual(r["besoin"]["population_cible"], pop0, delta=1)
        maintenir = r["besoin"]["cas"][0]
        self.assertEqual(maintenir["personnes"], 0)   # taux maintenu + population stable = rien à construire

    def test_simulation_latrines(self):
        sans = S.simuler(self.ctx)
        avec = S.simuler(self.ctx, latrines_par_an=2000)
        self.assertIsNone(sans["besoin"]["simulation_latrines"])
        self.assertIsNone(sans["series"]["latrines"])
        self.assertEqual(avec["series"]["latrines"][avec["annees"].index(2025)], avec["series"]["realisation"][avec["annees"].index(2025)])
        self.assertIsNotNone(avec["besoin"]["simulation_latrines"])
        plus = S.simuler(self.ctx, latrines_par_an=20000)["besoin"]["simulation_latrines"]["taux_cible"]
        self.assertGreater(plus, avec["besoin"]["simulation_latrines"]["taux_cible"])

    # ----- avertissements
    def test_avertissement_objectif_inferieur_au_taux_actuel(self):
        r = S.simuler(self.ctx, objectif_final=1)
        self.assertTrue(any("inférieur au taux actuel" in a for a in r["avertissements"]))
        self.assertFalse(any("inférieur au taux actuel" in a for a in S.simuler(self.ctx, objectif_final=100)["avertissements"]))

    # ----- sortie JSON
    def test_sortie_json_sans_nan(self):
        for args in [(S.NATIONAL, None), (S.REGION, "REGION 2"), (S.COMMUNE, 7)]:
            r = S.simuler(self.ctx, *args, sensibilisation=0.3, latrines_par_an=500)
            texte = json.dumps(r, allow_nan=False)  # lève une erreur s'il reste un NaN
            self.assertIn("series", texte)

    # ----- entrées invalides
    def test_parametres_invalides(self):
        for kwargs in [dict(nb_annees=0), dict(nb_annees=11), dict(objectif_final=101), dict(objectif_final=-1),
                       dict(sensibilisation=1.5), dict(croissance_pct=-1), dict(personnes_par_latrine=0),
                       dict(latrines_par_an=-5)]:
            with self.assertRaises(ValueError, msg=str(kwargs)):
                S.simuler(self.ctx, **kwargs)

    def test_territoires_inconnus(self):
        with self.assertRaises(KeyError):
            S.simuler(self.ctx, S.REGION, "N'EXISTE PAS")
        with self.assertRaises(KeyError):
            S.simuler(self.ctx, S.COMMUNE, 99999)
        with self.assertRaises(ValueError):
            S.simuler(self.ctx, S.COMMUNE, "abc")
        with self.assertRaises(ValueError):
            S.simuler(self.ctx, "planete")

    # ----- méta et communes
    def test_meta_et_communes(self):
        m = S.meta(self.ctx)
        self.assertEqual(m["annee_depart"], 2025)
        self.assertEqual(m["regions"], ["REGION 1", "REGION 2", "REGION 3"])
        self.assertEqual([s["cle"] for s in m["scenarios"]], ["sensibilisation", "education", "campagnes_sante"])
        self.assertEqual(len(S.communes(self.ctx, "REGION 1")), 20)
        with self.assertRaises(KeyError):
            S.communes(self.ctx, "XXX")

    def test_performance_absente(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(S.performance(Path(d)))


if __name__ == "__main__":
    unittest.main()
