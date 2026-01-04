"""
Database connection management for ollama-llm-proxy.

Handles SQLite (default) and PostgreSQL connections.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _ensure_data_dir() -> Path:
    """Ensure the data directory exists and return its path."""
    # Use /data in Docker (separate volume), otherwise relative to this file
    if Path("/app").exists():
        # Running in Docker container - use /data which is a mounted volume
        data_dir = Path("/data")
    else:
        # Running locally
        data_dir = Path(__file__).parent / "data"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        # Test that we can write to it
        test_file = data_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        logger.info(f"Data directory ready: {data_dir}")
    except PermissionError as e:
        logger.error(f"Cannot write to data directory {data_dir}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error setting up data directory {data_dir}: {e}")
        raise

    return data_dir


def get_database_url() -> str:
    """
    Get database URL from environment or use default SQLite.

    Supports:
    - DATABASE_URL env var (for PostgreSQL or custom SQLite path)
    - Default: sqlite:////data/proxy.db (Docker) or local db/data/proxy.db
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        # Handle Heroku-style postgres:// URLs
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    data_dir = _ensure_data_dir()
    return f"sqlite:///{data_dir}/proxy.db"


def get_engine() -> Engine:
    """Get or create the database engine."""
    global _engine

    if _engine is None:
        url = get_database_url()
        logger.info(
            f"Connecting to database: {url.split('@')[-1] if '@' in url else url}"
        )

        # SQLite-specific settings
        if url.startswith("sqlite"):
            _engine = create_engine(
                url,
                connect_args={
                    "check_same_thread": False
                },  # Allow multi-threaded access
                echo=os.environ.get("SQL_DEBUG", "").lower() == "true",
            )

            # Enable foreign keys for SQLite
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        else:
            # PostgreSQL or other databases
            _engine = create_engine(
                url,
                echo=os.environ.get("SQL_DEBUG", "").lower() == "true",
                pool_pre_ping=True,  # Verify connections before use
            )

    return _engine


def get_session_factory() -> sessionmaker:
    """Get or create the session factory."""
    global _SessionLocal

    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )

    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.

    Usage:
        with get_db() as db:
            db.query(Provider).all()

    Or as a FastAPI dependency:
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for database sessions."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


_migrations_run = False


def run_migrations() -> None:
    """
    Run database migrations if not already run in this process.

    Safe to call multiple times - will only run once per process.
    """
    global _migrations_run
    if _migrations_run:
        return

    engine = get_engine()
    _run_migrations(engine)
    _migrations_run = True


def _run_migrations(engine) -> None:
    """
    Run database migrations to add new columns to existing tables.

    This handles schema evolution without requiring users to delete their database.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)

    # Migration: Add override fields to model_overrides table
    if "model_overrides" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("model_overrides")
        }

        migrations = []

        # New columns added for model override support
        if "input_cost" not in existing_columns:
            migrations.append("ALTER TABLE model_overrides ADD COLUMN input_cost REAL")
        if "output_cost" not in existing_columns:
            migrations.append("ALTER TABLE model_overrides ADD COLUMN output_cost REAL")
        if "capabilities_json" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN capabilities_json TEXT"
            )
        if "context_length" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN context_length INTEGER"
            )
        if "description" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN description VARCHAR(500)"
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for model_overrides table"
            )
            with engine.connect() as conn:
                for migration in migrations:
                    logger.debug(f"Running migration: {migration}")
                    conn.execute(text(migration))
                conn.commit()

    # Migration: Add cost column to request_logs table
    if "request_logs" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("request_logs")
        }

        if "cost" not in existing_columns:
            logger.info("Running migration: Adding cost column to request_logs table")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE request_logs ADD COLUMN cost REAL"))
                conn.commit()


def init_db(drop_all: bool = False) -> None:
    """
    Initialize the database schema.

    Args:
        drop_all: If True, drop all tables before creating (for testing)
    """
    engine = get_engine()

    if drop_all:
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=engine)

    logger.info("Creating database tables")
    Base.metadata.create_all(bind=engine)

    # Run migrations for existing databases
    _run_migrations(engine)


def check_db_initialized() -> bool:
    """Check if the database has been initialized with tables."""
    engine = get_engine()
    from sqlalchemy import inspect

    inspector = inspect(engine)
    return "providers" in inspector.get_table_names()


def reset_connection() -> None:
    """Reset the database connection (useful for testing)."""
    global _engine, _SessionLocal

    if _engine:
        _engine.dispose()

    _engine = None
    _SessionLocal = None
