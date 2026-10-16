-- =====================================================================
-- schema.sql (PostgreSQL) : base WASH, données 2020-2024
-- Usage :
--   psql -U postgres -c "DROP DATABASE IF EXISTS wash_seam_db;"
--   psql -U postgres -c "CREATE DATABASE wash_seam_db;"
--   psql -U postgres -d wash_seam_db -f schema.sql
-- =====================================================================

-- 1. Régions & Districts (une ligne = un couple région/district)
CREATE TABLE IF NOT EXISTS regions (
    region_id    SERIAL PRIMARY KEY,
    nom_region   VARCHAR(100) NOT NULL,
    nom_district VARCHAR(100) NOT NULL,
    CONSTRAINT unique_region_district UNIQUE (nom_region, nom_district)
);

-- 2. Communes
CREATE TABLE IF NOT EXISTS communes (
    commune_id  SERIAL PRIMARY KEY,
    region_id   INT NOT NULL REFERENCES regions(region_id) ON DELETE CASCADE,
    nom_commune VARCHAR(100) NOT NULL,
    milieu      VARCHAR(10) CHECK (milieu IN ('URBAIN', 'RURAL')),
    CONSTRAINT unique_commune_district UNIQUE (region_id, nom_commune)
);
CREATE INDEX IF NOT EXISTS idx_communes_region ON communes (region_id);

-- 3. Indicateurs WASH : une ligne par commune et par année
-- NULL = donnée inconnue (aucun DEFAULT 0)
CREATE TABLE IF NOT EXISTS indicateurs_wash (
    indicateur_id SERIAL PRIMARY KEY,
    commune_id    INT NOT NULL REFERENCES communes(commune_id) ON DELETE CASCADE,
    annee         INT NOT NULL CHECK (annee BETWEEN 2000 AND 2100),
    population    INT,
    beneficiaires_latrine_basique_total INT,
    taux_assainissement NUMERIC(5, 2),   -- en %, ex : 45.56
    CONSTRAINT unique_commune_annee UNIQUE (commune_id, annee)
);
CREATE INDEX IF NOT EXISTS idx_indicateurs_commune ON indicateurs_wash (commune_id);
CREATE INDEX IF NOT EXISTS idx_indicateurs_annee   ON indicateurs_wash (annee);
