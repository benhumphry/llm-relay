"""
CRUD operations for plugin configurations.

This module provides functions for managing plugin configuration records
in the database.
"""

import json
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db.connection import get_db_context
from db.models import PluginConfig

logger = logging.getLogger(__name__)


def get_plugin_config(config_id: int) -> Optional[PluginConfig]:
    """
    Get a plugin configuration by ID.

    Args:
        config_id: The configuration ID

    Returns:
        PluginConfig or None if not found
    """
    with get_db_context() as session:
        config = session.get(PluginConfig, config_id)
        if config:
            # Detach from session before returning
            session.expunge(config)
        return config


def get_plugin_config_by_name(name: str) -> Optional[PluginConfig]:
    """
    Get a plugin configuration by name.

    Args:
        name: The configuration name (unique)

    Returns:
        PluginConfig or None if not found
    """
    with get_db_context() as session:
        stmt = select(PluginConfig).where(PluginConfig.name == name)
        config = session.execute(stmt).scalar_one_or_none()
        if config:
            session.expunge(config)
        return config


def get_plugin_configs_by_type(
    plugin_type: str, source_type: Optional[str] = None, enabled_only: bool = False
) -> list[PluginConfig]:
    """
    Get plugin configurations by type.

    Args:
        plugin_type: Plugin type ('document_source', 'live_source', 'action')
        source_type: Optional source type filter
        enabled_only: If True, only return enabled configurations

    Returns:
        List of PluginConfig records
    """
    with get_db_context() as session:
        stmt = select(PluginConfig).where(PluginConfig.plugin_type == plugin_type)

        if source_type:
            stmt = stmt.where(PluginConfig.source_type == source_type)

        if enabled_only:
            stmt = stmt.where(PluginConfig.enabled == True)

        stmt = stmt.order_by(PluginConfig.name)

        configs = session.execute(stmt).scalars().all()

        # Detach from session
        for config in configs:
            session.expunge(config)

        return list(configs)


def get_all_plugin_configs(enabled_only: bool = False) -> list[PluginConfig]:
    """
    Get all plugin configurations.

    Args:
        enabled_only: If True, only return enabled configurations

    Returns:
        List of all PluginConfig records
    """
    with get_db_context() as session:
        stmt = select(PluginConfig)

        if enabled_only:
            stmt = stmt.where(PluginConfig.enabled == True)

        stmt = stmt.order_by(PluginConfig.plugin_type, PluginConfig.name)

        configs = session.execute(stmt).scalars().all()

        for config in configs:
            session.expunge(config)

        return list(configs)


def create_plugin_config(
    plugin_type: str,
    source_type: str,
    name: str,
    config: Optional[dict] = None,
    enabled: bool = True,
) -> Optional[PluginConfig]:
    """
    Create a new plugin configuration.

    Args:
        plugin_type: Plugin type ('document_source', 'live_source', 'action')
        source_type: Plugin's source_type/action_type identifier
        name: User-friendly name (must be unique)
        config: Plugin-specific configuration dict
        enabled: Whether the configuration is enabled

    Returns:
        Created PluginConfig or None if name already exists
    """
    with get_db_context() as session:
        try:
            plugin_config = PluginConfig(
                plugin_type=plugin_type,
                source_type=source_type,
                name=name,
                config_json=json.dumps(config) if config else None,
                enabled=enabled,
            )
            session.add(plugin_config)
            session.commit()
            session.refresh(plugin_config)

            logger.info(f"Created plugin config: {name} ({plugin_type}/{source_type})")

            session.expunge(plugin_config)
            return plugin_config

        except IntegrityError:
            session.rollback()
            logger.warning(f"Plugin config with name '{name}' already exists")
            return None


def update_plugin_config(
    config_id: int,
    name: Optional[str] = None,
    config: Optional[dict] = None,
    enabled: Optional[bool] = None,
) -> Optional[PluginConfig]:
    """
    Update a plugin configuration.

    Args:
        config_id: The configuration ID to update
        name: New name (optional)
        config: New configuration dict (optional)
        enabled: New enabled state (optional)

    Returns:
        Updated PluginConfig or None if not found or name conflict
    """
    with get_db_context() as session:
        try:
            plugin_config = session.get(PluginConfig, config_id)
            if not plugin_config:
                return None

            if name is not None:
                plugin_config.name = name

            if config is not None:
                plugin_config.config_json = json.dumps(config)

            if enabled is not None:
                plugin_config.enabled = enabled

            session.commit()
            session.refresh(plugin_config)

            logger.info(
                f"Updated plugin config: {plugin_config.name} (ID: {config_id})"
            )

            session.expunge(plugin_config)
            return plugin_config

        except IntegrityError:
            session.rollback()
            logger.warning(f"Plugin config name conflict on update: {name}")
            return None


def delete_plugin_config(config_id: int) -> bool:
    """
    Delete a plugin configuration.

    Args:
        config_id: The configuration ID to delete

    Returns:
        True if deleted, False if not found
    """
    with get_db_context() as session:
        plugin_config = session.get(PluginConfig, config_id)
        if not plugin_config:
            return False

        name = plugin_config.name
        session.delete(plugin_config)
        session.commit()

        logger.info(f"Deleted plugin config: {name} (ID: {config_id})")
        return True


def get_action_configs(enabled_only: bool = True) -> list[PluginConfig]:
    """
    Get all action plugin configurations.

    Convenience function for getting action handler configs.

    Args:
        enabled_only: If True, only return enabled configurations

    Returns:
        List of action PluginConfig records
    """
    return get_plugin_configs_by_type("action", enabled_only=enabled_only)


def get_document_source_configs(enabled_only: bool = True) -> list[PluginConfig]:
    """
    Get all document source plugin configurations.

    Args:
        enabled_only: If True, only return enabled configurations

    Returns:
        List of document source PluginConfig records
    """
    return get_plugin_configs_by_type("document_source", enabled_only=enabled_only)


def get_live_source_configs(enabled_only: bool = True) -> list[PluginConfig]:
    """
    Get all live source plugin configurations.

    Args:
        enabled_only: If True, only return enabled configurations

    Returns:
        List of live source PluginConfig records
    """
    return get_plugin_configs_by_type("live_source", enabled_only=enabled_only)
