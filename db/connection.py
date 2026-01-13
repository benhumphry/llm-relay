"""
Database connection management for LLM Relay.

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
        # v2.2.3: Extended cost parameters
        if "cache_read_multiplier" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN cache_read_multiplier REAL"
            )
        if "cache_write_multiplier" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN cache_write_multiplier REAL"
            )
        # v3.0.1: Provider quirks (API-specific behavior)
        if "use_max_completion_tokens" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN use_max_completion_tokens BOOLEAN"
            )
        if "supports_system_prompt" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN supports_system_prompt BOOLEAN"
            )
        if "unsupported_params_json" not in existing_columns:
            migrations.append(
                "ALTER TABLE model_overrides ADD COLUMN unsupported_params_json TEXT"
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

    # Migration: Add columns to request_logs table
    if "request_logs" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("request_logs")
        }

        migrations = []

        # v2.2.2: Add cost column
        if "cost" not in existing_columns:
            migrations.append(("cost", "ALTER TABLE request_logs ADD COLUMN cost REAL"))

        # v2.2.3: Add extended token tracking columns
        if "reasoning_tokens" not in existing_columns:
            migrations.append(
                (
                    "reasoning_tokens",
                    "ALTER TABLE request_logs ADD COLUMN reasoning_tokens INTEGER",
                )
            )
        if "cached_input_tokens" not in existing_columns:
            migrations.append(
                (
                    "cached_input_tokens",
                    "ALTER TABLE request_logs ADD COLUMN cached_input_tokens INTEGER",
                )
            )
        if "cache_creation_tokens" not in existing_columns:
            migrations.append(
                (
                    "cache_creation_tokens",
                    "ALTER TABLE request_logs ADD COLUMN cache_creation_tokens INTEGER",
                )
            )
        if "cache_read_tokens" not in existing_columns:
            migrations.append(
                (
                    "cache_read_tokens",
                    "ALTER TABLE request_logs ADD COLUMN cache_read_tokens INTEGER",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for request_logs table"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add extended cost columns to custom_models table
    if "custom_models" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("custom_models")
        }

        migrations = []

        # v2.2.3: Add cost columns (were missing entirely)
        if "input_cost" not in existing_columns:
            migrations.append(
                ("input_cost", "ALTER TABLE custom_models ADD COLUMN input_cost REAL")
            )
        if "output_cost" not in existing_columns:
            migrations.append(
                ("output_cost", "ALTER TABLE custom_models ADD COLUMN output_cost REAL")
            )
        if "cache_read_multiplier" not in existing_columns:
            migrations.append(
                (
                    "cache_read_multiplier",
                    "ALTER TABLE custom_models ADD COLUMN cache_read_multiplier REAL",
                )
            )
        if "cache_write_multiplier" not in existing_columns:
            migrations.append(
                (
                    "cache_write_multiplier",
                    "ALTER TABLE custom_models ADD COLUMN cache_write_multiplier REAL",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for custom_models table"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add extended cost columns to models table
    if "models" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("models")}

        migrations = []

        # v2.2.3: Base cost parameters
        if "input_cost" not in existing_columns:
            migrations.append(
                ("input_cost", "ALTER TABLE models ADD COLUMN input_cost REAL")
            )
        if "output_cost" not in existing_columns:
            migrations.append(
                ("output_cost", "ALTER TABLE models ADD COLUMN output_cost REAL")
            )

        # v2.2.3: Extended cost parameters
        if "cache_read_multiplier" not in existing_columns:
            migrations.append(
                (
                    "cache_read_multiplier",
                    "ALTER TABLE models ADD COLUMN cache_read_multiplier REAL",
                )
            )
        if "cache_write_multiplier" not in existing_columns:
            migrations.append(
                (
                    "cache_write_multiplier",
                    "ALTER TABLE models ADD COLUMN cache_write_multiplier REAL",
                )
            )

        # v3.0: Source tracking for DB-driven config
        if "source" not in existing_columns:
            migrations.append(
                (
                    "source",
                    "ALTER TABLE models ADD COLUMN source VARCHAR(20) DEFAULT 'litellm'",
                )
            )
        if "last_synced" not in existing_columns:
            migrations.append(
                (
                    "last_synced",
                    "ALTER TABLE models ADD COLUMN last_synced DATETIME",
                )
            )

        if migrations:
            logger.info(f"Running {len(migrations)} migration(s) for models table")
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add columns to providers table
    if "providers" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("providers")}

        migrations = []

        # v3.0: Source tracking and display name
        if "source" not in existing_columns:
            migrations.append(
                (
                    "source",
                    "ALTER TABLE providers ADD COLUMN source VARCHAR(20) DEFAULT 'system'",
                )
            )
        if "display_name" not in existing_columns:
            migrations.append(
                (
                    "display_name",
                    "ALTER TABLE providers ADD COLUMN display_name VARCHAR(100)",
                )
            )

        if migrations:
            logger.info(f"Running {len(migrations)} migration(s) for providers table")
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add alias column to request_logs table (v3.1)
    if "request_logs" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("request_logs")
        }

        if "alias" not in existing_columns:
            logger.info("Adding alias column to request_logs table")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE request_logs ADD COLUMN alias VARCHAR(100)")
                )
                conn.commit()

    # Migration: Add alias column to daily_stats table (v3.1)
    if "daily_stats" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("daily_stats")}

        if "alias" not in existing_columns:
            logger.info("Adding alias column to daily_stats table")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE daily_stats ADD COLUMN alias VARCHAR(100)")
                )
                conn.commit()

    # Migration: Add smart router tracking columns to request_logs (v3.2)
    if "request_logs" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("request_logs")
        }

        migrations = []

        if "is_designator" not in existing_columns:
            migrations.append(
                (
                    "is_designator",
                    "ALTER TABLE request_logs ADD COLUMN is_designator BOOLEAN DEFAULT 0",
                )
            )
        if "router_name" not in existing_columns:
            migrations.append(
                (
                    "router_name",
                    "ALTER TABLE request_logs ADD COLUMN router_name VARCHAR(100)",
                )
            )

        # v3.5: Add augmentor_name column
        if "augmentor_name" not in existing_columns:
            migrations.append(
                (
                    "augmentor_name",
                    "ALTER TABLE request_logs ADD COLUMN augmentor_name VARCHAR(100)",
                )
            )

        # v3.5.1: Add augmentation detail columns
        if "augmentation_type" not in existing_columns:
            migrations.append(
                (
                    "augmentation_type",
                    "ALTER TABLE request_logs ADD COLUMN augmentation_type VARCHAR(20)",
                )
            )
        if "augmentation_query" not in existing_columns:
            migrations.append(
                (
                    "augmentation_query",
                    "ALTER TABLE request_logs ADD COLUMN augmentation_query VARCHAR(500)",
                )
            )
        if "augmentation_urls" not in existing_columns:
            migrations.append(
                (
                    "augmentation_urls",
                    "ALTER TABLE request_logs ADD COLUMN augmentation_urls TEXT",
                )
            )
        # v3.8: Smart RAG tracking
        if "rag_name" not in existing_columns:
            migrations.append(
                (
                    "rag_name",
                    "ALTER TABLE request_logs ADD COLUMN rag_name VARCHAR(100)",
                )
            )

        # v1.6: Cache tracking
        if "is_cache_hit" not in existing_columns:
            migrations.append(
                (
                    "is_cache_hit",
                    "ALTER TABLE request_logs ADD COLUMN is_cache_hit BOOLEAN DEFAULT false",
                )
            )
        if "cache_name" not in existing_columns:
            migrations.append(
                (
                    "cache_name",
                    "ALTER TABLE request_logs ADD COLUMN cache_name VARCHAR(100)",
                )
            )
        if "cache_tokens_saved" not in existing_columns:
            migrations.append(
                (
                    "cache_tokens_saved",
                    "ALTER TABLE request_logs ADD COLUMN cache_tokens_saved INTEGER DEFAULT 0",
                )
            )
        if "cache_cost_saved" not in existing_columns:
            migrations.append(
                (
                    "cache_cost_saved",
                    "ALTER TABLE request_logs ADD COLUMN cache_cost_saved REAL DEFAULT 0.0",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for request_logs table (v3.2)"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Alter augmentation_query column to TEXT (v3.10)
    if "request_logs" in inspector.get_table_names():
        # Check if column type needs to be changed (VARCHAR -> TEXT)
        # This is safe to run multiple times - TEXT can hold any VARCHAR data
        try:
            with engine.connect() as conn:
                # PostgreSQL syntax
                conn.execute(
                    text(
                        "ALTER TABLE request_logs ALTER COLUMN augmentation_query TYPE TEXT"
                    )
                )
                conn.commit()
                logger.info("Migrated augmentation_query column to TEXT (v3.10)")
        except Exception:
            # SQLite doesn't support ALTER COLUMN TYPE, but SQLite TEXT is already unlimited
            pass

    # Migration: Add router_name column to daily_stats table (v3.2)
    if "daily_stats" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("daily_stats")}

        if "router_name" not in existing_columns:
            logger.info("Adding router_name column to daily_stats table")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE daily_stats ADD COLUMN router_name VARCHAR(100)")
                )
                conn.commit()

    # Migration: Add match_last_message_only column to smart_caches table (v3.3)
    if "smart_caches" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("smart_caches")
        }

        if "match_last_message_only" not in existing_columns:
            logger.info("Adding match_last_message_only column to smart_caches table")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_caches ADD COLUMN match_last_message_only BOOLEAN DEFAULT false"
                    )
                )
                conn.commit()

        if "min_cached_tokens" not in existing_columns:
            logger.info("Adding min_cached_tokens column to smart_caches table")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_caches ADD COLUMN min_cached_tokens INTEGER DEFAULT 50"
                    )
                )
                conn.commit()

    # Migration: Add use_model_intelligence column to smart_routers table (v3.6)
    if "smart_routers" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("smart_routers")
        }

        if "use_model_intelligence" not in existing_columns:
            logger.info("Adding use_model_intelligence column to smart_routers table")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_routers ADD COLUMN use_model_intelligence BOOLEAN DEFAULT false"
                    )
                )
                conn.commit()

        if "search_provider" not in existing_columns:
            logger.info("Adding search_provider column to smart_routers table")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_routers ADD COLUMN search_provider VARCHAR(50)"
                    )
                )
                conn.commit()

        if "intelligence_model" not in existing_columns:
            logger.info("Adding intelligence_model column to smart_routers table")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_routers ADD COLUMN intelligence_model VARCHAR(150)"
                    )
                )
                conn.commit()

    # Migration: Add tags_json column to redirects table (v3.7)
    if "redirects" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("redirects")}

        if "tags_json" not in existing_columns:
            logger.info("Adding tags_json column to redirects table")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE redirects ADD COLUMN tags_json TEXT"))
                conn.commit()

    # Migration: Add vision model columns to smart_rags table (v3.9)
    if "smart_rags" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("smart_rags")}

        migrations = []

        if "vision_provider" not in existing_columns:
            migrations.append(
                (
                    "vision_provider",
                    "ALTER TABLE smart_rags ADD COLUMN vision_provider VARCHAR(100) DEFAULT 'local'",
                )
            )
        if "vision_model" not in existing_columns:
            migrations.append(
                (
                    "vision_model",
                    "ALTER TABLE smart_rags ADD COLUMN vision_model VARCHAR(150)",
                )
            )
        if "vision_ollama_url" not in existing_columns:
            migrations.append(
                (
                    "vision_ollama_url",
                    "ALTER TABLE smart_rags ADD COLUMN vision_ollama_url VARCHAR(500)",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for smart_rags table (v3.9)"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add reranking columns to smart_rags table (v1.5)
    if "smart_rags" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("smart_rags")}

        migrations = []

        if "rerank_provider" not in existing_columns:
            migrations.append(
                (
                    "rerank_provider",
                    "ALTER TABLE smart_rags ADD COLUMN rerank_provider VARCHAR(50) DEFAULT 'local'",
                )
            )
        if "rerank_model" not in existing_columns:
            migrations.append(
                (
                    "rerank_model",
                    "ALTER TABLE smart_rags ADD COLUMN rerank_model VARCHAR(150) DEFAULT 'cross-encoder/ms-marco-MiniLM-L-6-v2'",
                )
            )
        if "rerank_top_n" not in existing_columns:
            migrations.append(
                (
                    "rerank_top_n",
                    "ALTER TABLE smart_rags ADD COLUMN rerank_top_n INTEGER DEFAULT 20",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for smart_rags table (v1.5 reranking)"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add MCP source columns to smart_rags table (v1.6)
    if "smart_rags" in inspector.get_table_names():
        existing_columns = {col["name"] for col in inspector.get_columns("smart_rags")}

        migrations = []

        if "source_type" not in existing_columns:
            migrations.append(
                (
                    "source_type",
                    "ALTER TABLE smart_rags ADD COLUMN source_type VARCHAR(20) DEFAULT 'local'",
                )
            )
        if "mcp_server_config_json" not in existing_columns:
            migrations.append(
                (
                    "mcp_server_config_json",
                    "ALTER TABLE smart_rags ADD COLUMN mcp_server_config_json TEXT",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for smart_rags table (v1.6 MCP)"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Make source_path nullable for MCP sources (v1.6.1)
    if "smart_rags" in inspector.get_table_names():
        # Check if source_path is currently NOT NULL (PostgreSQL only - SQLite doesn't enforce this well)
        db_url = get_database_url()
        if "postgresql" in db_url:
            with engine.connect() as conn:
                # Check if column is nullable
                result = conn.execute(
                    text("""
                        SELECT is_nullable
                        FROM information_schema.columns
                        WHERE table_name = 'smart_rags' AND column_name = 'source_path'
                    """)
                )
                row = result.fetchone()
                if row and row[0] == "NO":
                    logger.info("Making source_path nullable for MCP sources (v1.6.1)")
                    conn.execute(
                        text(
                            "ALTER TABLE smart_rags ALTER COLUMN source_path DROP NOT NULL"
                        )
                    )
                    conn.commit()

    # Migration: Add scraper and reranking columns to smart_augmentors table (v1.5)
    if "smart_augmentors" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("smart_augmentors")
        }

        migrations = []

        if "scraper_provider" not in existing_columns:
            migrations.append(
                (
                    "scraper_provider",
                    "ALTER TABLE smart_augmentors ADD COLUMN scraper_provider VARCHAR(50) DEFAULT 'builtin'",
                )
            )
        if "rerank_provider" not in existing_columns:
            migrations.append(
                (
                    "rerank_provider",
                    "ALTER TABLE smart_augmentors ADD COLUMN rerank_provider VARCHAR(50) DEFAULT 'local'",
                )
            )
        if "rerank_model" not in existing_columns:
            migrations.append(
                (
                    "rerank_model",
                    "ALTER TABLE smart_augmentors ADD COLUMN rerank_model VARCHAR(150) DEFAULT 'cross-encoder/ms-marco-MiniLM-L-6-v2'",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for smart_augmentors table (v1.5)"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Create document_stores table and migrate existing RAG sources (v3.9)
    if "document_stores" not in inspector.get_table_names():
        logger.info("Creating document_stores and smart_rag_stores tables (v3.9)")

        # Create the document_stores table
        with engine.connect() as conn:
            conn.execute(
                text("""
                    CREATE TABLE document_stores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name VARCHAR(100) NOT NULL UNIQUE,
                        source_type VARCHAR(20) DEFAULT 'local',
                        source_path VARCHAR(500),
                        mcp_server_config_json TEXT,
                        embedding_provider VARCHAR(100) DEFAULT 'local',
                        embedding_model VARCHAR(150),
                        ollama_url VARCHAR(500),
                        vision_provider VARCHAR(100) DEFAULT 'local',
                        vision_model VARCHAR(150),
                        vision_ollama_url VARCHAR(500),
                        chunk_size INTEGER DEFAULT 512,
                        chunk_overlap INTEGER DEFAULT 50,
                        index_schedule VARCHAR(100),
                        last_indexed DATETIME,
                        index_status VARCHAR(20) DEFAULT 'pending',
                        index_error TEXT,
                        document_count INTEGER DEFAULT 0,
                        chunk_count INTEGER DEFAULT 0,
                        collection_name VARCHAR(100),
                        description TEXT,
                        enabled BOOLEAN DEFAULT 1,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
            )

            # Create the junction table
            conn.execute(
                text("""
                    CREATE TABLE smart_rag_stores (
                        smart_rag_id INTEGER NOT NULL,
                        document_store_id INTEGER NOT NULL,
                        PRIMARY KEY (smart_rag_id, document_store_id),
                        FOREIGN KEY (smart_rag_id) REFERENCES smart_rags(id) ON DELETE CASCADE,
                        FOREIGN KEY (document_store_id) REFERENCES document_stores(id) ON DELETE CASCADE
                    )
                """)
            )

            # Create index on document_stores.name
            conn.execute(
                text("CREATE INDEX ix_document_stores_name ON document_stores(name)")
            )

            conn.commit()
            logger.info("Created document_stores and smart_rag_stores tables")

        # Migrate existing SmartRAG source configurations to DocumentStore
        if "smart_rags" in inspector.get_table_names():
            logger.info("Migrating existing SmartRAG sources to DocumentStores")
            with engine.connect() as conn:
                # Get all existing SmartRAGs that have source configuration
                result = conn.execute(
                    text("""
                        SELECT id, name, source_type, source_path, mcp_server_config_json,
                               embedding_provider, embedding_model, ollama_url,
                               vision_provider, vision_model, vision_ollama_url,
                               chunk_size, chunk_overlap, index_schedule,
                               last_indexed, index_status, index_error,
                               document_count, chunk_count, collection_name
                        FROM smart_rags
                        WHERE source_path IS NOT NULL OR mcp_server_config_json IS NOT NULL
                    """)
                )

                migrated_count = 0
                for row in result:
                    rag_id = row[0]
                    rag_name = row[1]

                    # Create a DocumentStore with the same config
                    store_name = f"{rag_name}-store"

                    # Insert the new document store
                    conn.execute(
                        text("""
                            INSERT INTO document_stores (
                                name, source_type, source_path, mcp_server_config_json,
                                embedding_provider, embedding_model, ollama_url,
                                vision_provider, vision_model, vision_ollama_url,
                                chunk_size, chunk_overlap, index_schedule,
                                last_indexed, index_status, index_error,
                                document_count, chunk_count, collection_name,
                                enabled, created_at, updated_at
                            ) VALUES (
                                :name, :source_type, :source_path, :mcp_config,
                                :embed_provider, :embed_model, :ollama_url,
                                :vision_provider, :vision_model, :vision_ollama_url,
                                :chunk_size, :chunk_overlap, :index_schedule,
                                :last_indexed, :index_status, :index_error,
                                :doc_count, :chunk_count, :collection_name,
                                1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                            )
                        """),
                        {
                            "name": store_name,
                            "source_type": row[2] or "local",
                            "source_path": row[3],
                            "mcp_config": row[4],
                            "embed_provider": row[5] or "local",
                            "embed_model": row[6],
                            "ollama_url": row[7],
                            "vision_provider": row[8] or "local",
                            "vision_model": row[9],
                            "vision_ollama_url": row[10],
                            "chunk_size": row[11] or 512,
                            "chunk_overlap": row[12] or 50,
                            "index_schedule": row[13],
                            "last_indexed": row[14],
                            "index_status": row[15] or "pending",
                            "index_error": row[16],
                            "doc_count": row[17] or 0,
                            "chunk_count": row[18] or 0,
                            "collection_name": row[19],
                        },
                    )

                    # Get the new store ID
                    store_id_result = conn.execute(
                        text("SELECT id FROM document_stores WHERE name = :name"),
                        {"name": store_name},
                    )
                    store_id = store_id_result.scalar()

                    # Link the RAG to the store
                    conn.execute(
                        text("""
                            INSERT INTO smart_rag_stores (smart_rag_id, document_store_id)
                            VALUES (:rag_id, :store_id)
                        """),
                        {"rag_id": rag_id, "store_id": store_id},
                    )

                    migrated_count += 1

                conn.commit()
                logger.info(
                    f"Migrated {migrated_count} SmartRAG source(s) to DocumentStores"
                )

    # Migration: Add google_account_id column to document_stores table (v1.7)
    if "document_stores" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("document_stores")
        }

        if "google_account_id" not in existing_columns:
            logger.info("Adding google_account_id column to document_stores table")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE document_stores ADD COLUMN google_account_id INTEGER REFERENCES oauth_tokens(id) ON DELETE SET NULL"
                    )
                )
                conn.commit()

    # Migration: Add Google Drive folder columns to document_stores table (v1.7.1)
    if "document_stores" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("document_stores")
        }

        migrations = []

        if "gdrive_folder_id" not in existing_columns:
            migrations.append(
                (
                    "gdrive_folder_id",
                    "ALTER TABLE document_stores ADD COLUMN gdrive_folder_id VARCHAR(100)",
                )
            )
        if "gdrive_folder_name" not in existing_columns:
            migrations.append(
                (
                    "gdrive_folder_name",
                    "ALTER TABLE document_stores ADD COLUMN gdrive_folder_name VARCHAR(255)",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for document_stores table (v1.7.1 Drive folders)"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add Gmail label columns to document_stores table (v1.7.2)
    if "document_stores" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("document_stores")
        }

        migrations = []

        if "gmail_label_id" not in existing_columns:
            migrations.append(
                (
                    "gmail_label_id",
                    "ALTER TABLE document_stores ADD COLUMN gmail_label_id VARCHAR(100)",
                )
            )
        if "gmail_label_name" not in existing_columns:
            migrations.append(
                (
                    "gmail_label_name",
                    "ALTER TABLE document_stores ADD COLUMN gmail_label_name VARCHAR(255)",
                )
            )
        if "gcalendar_calendar_id" not in existing_columns:
            migrations.append(
                (
                    "gcalendar_calendar_id",
                    "ALTER TABLE document_stores ADD COLUMN gcalendar_calendar_id VARCHAR(255)",
                )
            )
        if "gcalendar_calendar_name" not in existing_columns:
            migrations.append(
                (
                    "gcalendar_calendar_name",
                    "ALTER TABLE document_stores ADD COLUMN gcalendar_calendar_name VARCHAR(255)",
                )
            )
        if "paperless_url" not in existing_columns:
            migrations.append(
                (
                    "paperless_url",
                    "ALTER TABLE document_stores ADD COLUMN paperless_url VARCHAR(500)",
                )
            )
        if "paperless_token" not in existing_columns:
            migrations.append(
                (
                    "paperless_token",
                    "ALTER TABLE document_stores ADD COLUMN paperless_token VARCHAR(255)",
                )
            )
        if "github_repo" not in existing_columns:
            migrations.append(
                (
                    "github_repo",
                    "ALTER TABLE document_stores ADD COLUMN github_repo VARCHAR(255)",
                )
            )
        if "github_branch" not in existing_columns:
            migrations.append(
                (
                    "github_branch",
                    "ALTER TABLE document_stores ADD COLUMN github_branch VARCHAR(100)",
                )
            )
        if "github_path" not in existing_columns:
            migrations.append(
                (
                    "github_path",
                    "ALTER TABLE document_stores ADD COLUMN github_path VARCHAR(500)",
                )
            )

        if "notion_database_id" not in existing_columns:
            migrations.append(
                (
                    "notion_database_id",
                    "ALTER TABLE document_stores ADD COLUMN notion_database_id VARCHAR(100)",
                )
            )

        if "notion_page_id" not in existing_columns:
            migrations.append(
                (
                    "notion_page_id",
                    "ALTER TABLE document_stores ADD COLUMN notion_page_id VARCHAR(100)",
                )
            )

        if "nextcloud_folder" not in existing_columns:
            migrations.append(
                (
                    "nextcloud_folder",
                    "ALTER TABLE document_stores ADD COLUMN nextcloud_folder VARCHAR(500)",
                )
            )

        if "paperless_tag_id" not in existing_columns:
            migrations.append(
                (
                    "paperless_tag_id",
                    "ALTER TABLE document_stores ADD COLUMN paperless_tag_id INTEGER",
                )
            )

        if "paperless_tag_name" not in existing_columns:
            migrations.append(
                (
                    "paperless_tag_name",
                    "ALTER TABLE document_stores ADD COLUMN paperless_tag_name VARCHAR(255)",
                )
            )

        # v1.9: Temporal filtering for document stores
        if "use_temporal_filtering" not in existing_columns:
            migrations.append(
                (
                    "use_temporal_filtering",
                    "ALTER TABLE document_stores ADD COLUMN use_temporal_filtering BOOLEAN DEFAULT FALSE",
                )
            )

        # v1.9: max_documents limit for document stores
        if "max_documents" not in existing_columns:
            migrations.append(
                (
                    "max_documents",
                    "ALTER TABLE document_stores ADD COLUMN max_documents INTEGER",
                )
            )

        # v1.10: Website crawling fields for document stores
        if "website_url" not in existing_columns:
            migrations.append(
                (
                    "website_url",
                    "ALTER TABLE document_stores ADD COLUMN website_url VARCHAR(1000)",
                )
            )
        if "website_crawl_depth" not in existing_columns:
            migrations.append(
                (
                    "website_crawl_depth",
                    "ALTER TABLE document_stores ADD COLUMN website_crawl_depth INTEGER DEFAULT 1",
                )
            )
        if "website_max_pages" not in existing_columns:
            migrations.append(
                (
                    "website_max_pages",
                    "ALTER TABLE document_stores ADD COLUMN website_max_pages INTEGER DEFAULT 50",
                )
            )
        if "website_include_pattern" not in existing_columns:
            migrations.append(
                (
                    "website_include_pattern",
                    "ALTER TABLE document_stores ADD COLUMN website_include_pattern VARCHAR(500)",
                )
            )
        if "website_exclude_pattern" not in existing_columns:
            migrations.append(
                (
                    "website_exclude_pattern",
                    "ALTER TABLE document_stores ADD COLUMN website_exclude_pattern VARCHAR(500)",
                )
            )

        if migrations:
            logger.info(
                f"Running {len(migrations)} migration(s) for document_stores table"
            )
            with engine.connect() as conn:
                for col_name, sql in migrations:
                    logger.debug(f"Adding column: {col_name}")
                    conn.execute(text(sql))
                conn.commit()

    # Migration: Add system_prompt column to smart_aliases table (v1.7.4)
    if "smart_aliases" in inspector.get_table_names():
        existing_columns = {
            col["name"] for col in inspector.get_columns("smart_aliases")
        }

        if "system_prompt" not in existing_columns:
            logger.info("Adding system_prompt column to smart_aliases table (v1.7.4)")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE smart_aliases ADD COLUMN system_prompt TEXT")
                )
                conn.commit()

        # v1.7 Smart Tags: Add is_smart_tag and passthrough_model columns
        if "is_smart_tag" not in existing_columns:
            logger.info("Adding is_smart_tag column to smart_aliases table (v1.7)")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN is_smart_tag BOOLEAN DEFAULT FALSE"
                    )
                )
                conn.commit()

        if "passthrough_model" not in existing_columns:
            logger.info("Adding passthrough_model column to smart_aliases table (v1.7)")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN passthrough_model BOOLEAN DEFAULT FALSE"
                    )
                )
                conn.commit()

        # v1.8 Context Priority: Add context_priority column for hybrid RAG+Web
        if "context_priority" not in existing_columns:
            logger.info("Adding context_priority column to smart_aliases table (v1.8)")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN context_priority VARCHAR(20) DEFAULT 'balanced'"
                    )
                )
                conn.commit()

        # v1.9 Smart Source Selection: Designator decides which stores/web to use
        if "use_smart_source_selection" not in existing_columns:
            logger.info(
                "Adding use_smart_source_selection column to smart_aliases table (v1.9)"
            )
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN use_smart_source_selection BOOLEAN DEFAULT FALSE"
                    )
                )
                conn.commit()

        # v1.9: Memory fields for smart aliases
        if "use_memory" not in existing_columns:
            logger.info("Adding memory columns to smart_aliases table (v1.9)")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN use_memory BOOLEAN DEFAULT FALSE"
                    )
                )
                conn.execute(text("ALTER TABLE smart_aliases ADD COLUMN memory TEXT"))
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN memory_updated_at TIMESTAMP"
                    )
                )
                conn.commit()

        # v1.9.1: Memory max tokens field
        if "memory_max_tokens" not in existing_columns:
            logger.info("Adding memory_max_tokens to smart_aliases table (v1.9.1)")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN memory_max_tokens INTEGER DEFAULT 500"
                    )
                )
                conn.commit()

        # v1.9.2: Document store intelligence fields
        ds_columns = {col["name"] for col in inspector.get_columns("document_stores")}
        if "themes_json" not in ds_columns:
            logger.info("Adding intelligence columns to document_stores table (v1.9.2)")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE document_stores ADD COLUMN themes_json TEXT")
                )
                conn.execute(
                    text("ALTER TABLE document_stores ADD COLUMN best_for TEXT")
                )
                conn.execute(
                    text("ALTER TABLE document_stores ADD COLUMN content_summary TEXT")
                )
                conn.execute(
                    text(
                        "ALTER TABLE document_stores ADD COLUMN intelligence_updated_at TIMESTAMP"
                    )
                )
                conn.commit()

        # v1.9.3: Show sources attribution setting
        if "show_sources" not in existing_columns:
            logger.info("Adding show_sources to smart_aliases table (v1.9.3)")
            with engine.connect() as conn:
                conn.execute(
                    text(
                        "ALTER TABLE smart_aliases ADD COLUMN show_sources BOOLEAN DEFAULT FALSE"
                    )
                )
                conn.commit()

    # Migration: Drop legacy tables (v1.8 - Smart Aliases unification)
    # These tables have been replaced by the unified smart_aliases table
    legacy_tables = [
        "smart_enricher_stores",  # Junction table first (has FK constraints)
        "smart_enrichers",
        "smart_routers",
        "aliases",
    ]

    tables_to_drop = [t for t in legacy_tables if t in inspector.get_table_names()]
    if tables_to_drop:
        logger.info(f"Dropping legacy tables (v1.8): {', '.join(tables_to_drop)}")
        with engine.connect() as conn:
            for table_name in tables_to_drop:
                logger.debug(f"Dropping table: {table_name}")
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
            conn.commit()
        logger.info("Legacy tables dropped successfully")


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
