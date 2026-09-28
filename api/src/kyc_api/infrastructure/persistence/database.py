from __future__ import annotations

from importlib.resources import files

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError


def create_database_engine(database_url: str) -> Engine:
    try:
        return create_engine(
            database_url,
            pool_pre_ping=True,
            echo=False,
            hide_parameters=True,
            future=True,
        )
    except SQLAlchemyError:
        raise RuntimeError("PostgreSQL database configuration is invalid") from None


def migration_config(database_url: str) -> Config:
    root = files("kyc_api.infrastructure.persistence")
    config = Config(str(root.joinpath("alembic.ini")))
    config.set_main_option("script_location", str(root.joinpath("migrations")))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def expected_revision(database_url: str) -> str:
    return ScriptDirectory.from_config(migration_config(database_url)).get_current_head()


def verify_database(engine: Engine, database_url: str) -> None:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        if revision != expected_revision(database_url):
            raise RuntimeError("PostgreSQL schema revision is incompatible")
    except RuntimeError:
        raise
    except (SQLAlchemyError, LookupError):
        raise RuntimeError("PostgreSQL persistence is unavailable or not migrated") from None
