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


def get_database_url() -> str:
    """
    Get database URL from environment or use default SQLite.

    Supports:
    - DATABASE_URL env var (for PostgreSQL or custom SQLite path)
    - Default: sqlite:///data/proxy.db
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        # Handle Heroku-style postgres:// URLs
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url

    # Default to SQLite in data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
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
