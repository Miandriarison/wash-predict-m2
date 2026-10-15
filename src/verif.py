import pandas as pd
from database import obtenir_donnees_wash

df = obtenir_donnees_wash().pivot(index="commune_id", columns="annee", values="taux_assainissement")
print(df.diff(axis=1).iloc[:, 1:].describe().round(2))