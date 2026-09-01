import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()


def get_db_engine(custom_url: str = None) -> Engine:
    """Crée et retourne un moteur SQLAlchemy réutilisable dans tout le projet.

    Si aucune URL n'est fournie, elle est construite à partir des variables d'environnement.
    """
    if custom_url:
        return create_engine(custom_url)

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "wash_seam_db")
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASS", "")

    database_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
    return create_engine(database_url)