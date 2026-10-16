"""Tests de l'API FastAPI (nécessitent fastapi et httpx : pip install -r requirements.txt).

Le contexte (données + modèle) est remplacé par un jeu synthétique : aucune base PostgreSQL n'est utilisée.
"""

import unittest
from unittest.mock import patch

import outils

try:
    from fastapi.testclient import TestClient
    DISPONIBLE = True
except ImportError:  # fastapi / httpx absents
    DISPONIBLE = False

if DISPONIBLE:
    import api  # hors du try : une vraie erreur dans api.py doit faire échouer les tests, pas les « ignorer »


@unittest.skipUnless(DISPONIBLE, "fastapi/httpx non installés")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api.set_contexte(outils.contexte_test())
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        api.set_contexte(None)

    def test_page_d_accueil(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("WASH Predict", r.text)

    def test_fichiers_statiques(self):
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/style.css").status_code, 200)

    def test_sante_et_meta(self):
        self.assertEqual(self.client.get("/sante").json()["statut"], "ok")
        meta = self.client.get("/api/meta").json()
        self.assertEqual(meta["annee_depart"], 2025)
        self.assertEqual(len(meta["regions"]), 3)

    def test_communes(self):
        r = self.client.get("/api/communes", params={"region": "REGION 1"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 20)
        self.assertEqual(self.client.get("/api/communes", params={"region": "XXX"}).status_code, 404)
        self.assertEqual(self.client.get("/api/communes").status_code, 422)  # paramètre manquant

    def test_simulation_nationale(self):
        r = self.client.post("/api/simulation", json={})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["annee_cible"], 2029)
        self.assertEqual(len(d["series"]["realisation"]), len(d["annees"]))

    def test_simulation_commune_avec_scenario(self):
        r = self.client.post("/api/simulation", json={
            "type_territoire": "commune", "valeur": 3, "sensibilisation": 1, "latrines_par_an": 100})
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIsNotNone(d["series"]["scenario"])
        self.assertIsNotNone(d["series"]["latrines"])

    def test_simulation_entrees_invalides(self):
        for corps in [{"nb_annees": 0}, {"objectif_final": 150}, {"sensibilisation": -0.1},
                      {"type_territoire": "planete"}, {"croissance_pct": 50}, {"nb_annees": "beaucoup"}]:
            self.assertEqual(self.client.post("/api/simulation", json=corps).status_code, 422, corps)

    def test_simulation_territoire_inconnu(self):
        r = self.client.post("/api/simulation", json={"type_territoire": "commune", "valeur": 123456})
        self.assertEqual(r.status_code, 404)
        r = self.client.post("/api/simulation", json={"type_territoire": "region", "valeur": "XXX"})
        self.assertEqual(r.status_code, 404)

    def test_images_liste_blanche(self):
        self.assertEqual(self.client.get("/api/performance/image/inconnue").status_code, 404)
        self.assertEqual(self.client.get("/api/performance/image/..%2F..%2Fetc%2Fpasswd").status_code, 404)

    def test_contexte_indisponible_renvoie_503(self):
        # base arrêtée ou modèle absent : la route doit répondre proprement (503) et non planter
        api.set_contexte(None)
        try:
            with patch.object(api.sim.Contexte, "depuis_base", side_effect=RuntimeError("base arrêtée")):
                r = self.client.get("/api/meta")
            self.assertEqual(r.status_code, 503)
            self.assertIn("indisponibles", r.json()["detail"])
        finally:
            api.set_contexte(outils.contexte_test())


if __name__ == "__main__":
    unittest.main()
