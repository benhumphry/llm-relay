"""
Admin Flask blueprint for LLM Relay.

Provides:
- Web UI for managing providers, models, and settings
- REST API for CRUD operations
- Authentication via session cookies
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from db import (
    CustomModel,
    DailyStats,
    Model,
    ModelOverride,
    Provider,
    RequestLog,
    Setting,
    get_db_context,
)

# db.importer is deprecated - seeding now happens via db.seed
from providers import registry
from providers.hybrid_loader import get_all_models_with_metadata
from providers.loader import get_all_provider_names
from version import VERSION

from .auth import (
    authenticate,
    is_auth_enabled,
    is_authenticated,
    login_user,
    logout_user,
    require_auth,
    require_auth_api,
    set_admin_password,
)

logger = logging.getLogger(__name__)

# Directory paths
ADMIN_DIR = Path(__file__).parent
STATIC_DIR = ADMIN_DIR / "static"
TEMPLATE_DIR = ADMIN_DIR / "templates"


def create_admin_blueprint(url_prefix: str = "/admin") -> Blueprint:
    """
    Create and configure the admin blueprint.

    Args:
        url_prefix: URL prefix for admin routes. Use "" for root when running
                   as a dedicated admin server on a separate port.
    """
    admin = Blueprint(
        "admin",
        __name__,
        url_prefix=url_prefix,
        static_folder=str(STATIC_DIR),
        static_url_path="/static",
        template_folder=str(TEMPLATE_DIR),
    )

    @admin.context_processor
    def inject_globals():
        """Make global variables available in all templates."""
        return {"auth_enabled": is_auth_enabled(), "version": VERSION}

    # -------------------------------------------------------------------------
    # Authentication Routes
    # -------------------------------------------------------------------------

    @admin.route("/login", methods=["GET", "POST"])
    def login():
        """Login page and handler."""
        # If auth is disabled, redirect to dashboard
        if not is_auth_enabled():
            return redirect(url_for("admin.dashboard"))

        error = None

        if request.method == "POST":
            password = request.form.get("password", "")
            if authenticate(password):
                login_user()
                next_url = request.args.get("next", url_for("admin.dashboard"))
                return redirect(next_url)
            error = "Invalid password"

        # If already authenticated, redirect to dashboard
        if is_authenticated():
            return redirect(url_for("admin.dashboard"))

        return render_template("login.html", error=error)

    @admin.route("/logout")
    def logout():
        """Logout and redirect to login page."""
        logout_user()
        return redirect(url_for("admin.login"))

    @admin.route("/api/login", methods=["POST"])
    def api_login():
        """API login endpoint."""
        data = request.get_json() or {}
        password = data.get("password", "")

        if authenticate(password):
            login_user()
            return jsonify({"success": True})

        return jsonify({"error": "Invalid password"}), 401

    @admin.route("/api/logout", methods=["POST"])
    def api_logout():
        """API logout endpoint."""
        logout_user()
        return jsonify({"success": True})

    # -------------------------------------------------------------------------
    # Page Routes
    # -------------------------------------------------------------------------

    @admin.route("/")
    @require_auth
    def dashboard():
        """Main dashboard page."""
        return render_template("dashboard.html")

    @admin.route("/providers")
    @require_auth
    def providers_page():
        """Redirect to consolidated Providers & Models page."""
        return redirect(url_for("admin.models_page"))

    @admin.route("/models")
    @require_auth
    def models_page():
        """Models management page."""
        return render_template("models.html")

    @admin.route("/settings")
    @require_auth
    def settings_page():
        """Settings page."""
        return render_template("settings.html")

    @admin.route("/ollama")
    @require_auth
    def ollama_page():
        """Ollama management page."""
        return render_template("ollama.html")

    @admin.route("/aliases")
    @require_auth
    def aliases_page():
        """Aliases management page."""
        return render_template("aliases.html")

    # -------------------------------------------------------------------------
    # Provider API
    # -------------------------------------------------------------------------

    @admin.route("/api/providers", methods=["GET"])
    @require_auth_api
    def list_providers():
        """List all providers from the registry and DB."""
        from db import OllamaInstance, Provider
        from providers import registry
        from providers.loader import get_provider_config
        from providers.ollama_provider import OllamaProvider

        providers_list = []
        seen_ids = set()

        # Get source and enabled info from database
        provider_db_info = {}
        with get_db_context() as db:
            for p in db.query(Provider).all():
                provider_db_info[p.id] = {"source": p.source, "enabled": p.enabled}

        for provider in registry.get_all_providers():
            config = get_provider_config(provider.name)
            api_key_env = config.get("api_key_env")
            has_api_key = bool(api_key_env and os.environ.get(api_key_env))

            # Determine provider type
            if isinstance(provider, OllamaProvider):
                provider_type = "ollama"
                has_api_key = True  # Ollama doesn't need API key
                base_url = provider.base_url
            else:
                provider_type = config.get("type", provider.name)
                base_url = config.get("base_url")

            # Get DB info for this provider
            db_info = provider_db_info.get(provider.name, {})
            is_system = db_info.get("source") == "system"
            # Use runtime enabled status from provider instance
            is_enabled = getattr(provider, "enabled", True)

            providers_list.append(
                {
                    "id": provider.name,
                    "type": provider_type,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "enabled": is_enabled,
                    "has_api_key": has_api_key,
                    "model_count": len(provider.get_models()),
                    "is_system": is_system,
                    "has_custom_cost": getattr(
                        provider, "has_custom_cost_calculation", False
                    ),
                }
            )
            seen_ids.add(provider.name)

        # Also include disabled providers from DB that aren't in registry
        with get_db_context() as db:
            # DB providers (system providers that might be disabled)
            for p in db.query(Provider).all():
                if p.id not in seen_ids:
                    has_api_key = bool(p.api_key_env and os.environ.get(p.api_key_env))
                    providers_list.append(
                        {
                            "id": p.id,
                            "type": p.type,
                            "base_url": p.base_url,
                            "api_key_env": p.api_key_env,
                            "enabled": p.enabled,
                            "has_api_key": has_api_key,
                            "model_count": 0,
                            "is_system": p.source == "system",
                            "has_custom_cost": p.type
                            in ("openrouter", "perplexity", "gemini"),
                        }
                    )
                    seen_ids.add(p.id)

            # DB Ollama instances not yet in registry (e.g., pending restart)
            db_instances = db.query(OllamaInstance).all()
            for inst in db_instances:
                if inst.name not in seen_ids:
                    providers_list.append(
                        {
                            "id": inst.name,
                            "type": "ollama",
                            "base_url": inst.base_url,
                            "api_key_env": None,
                            "enabled": inst.enabled,
                            "has_api_key": True,
                            "model_count": 0,
                            "is_system": False,
                            "pending_restart": True,
                        }
                    )

        return jsonify(providers_list)

    @admin.route("/api/providers/<provider_id>", methods=["GET"])
    @require_auth_api
    def get_provider(provider_id: str):
        """Get a single provider."""
        with get_db_context() as db:
            provider = db.query(Provider).filter(Provider.id == provider_id).first()
            if not provider:
                return jsonify({"error": "Provider not found"}), 404
            return jsonify(provider.to_dict())

    @admin.route("/api/providers", methods=["POST"])
    @require_auth_api
    def create_provider():
        """Create a new provider (currently only Ollama type supported)."""
        import re

        from db import OllamaInstance
        from providers import registry
        from providers.ollama_provider import OllamaProvider

        data = request.get_json() or {}

        required = ["id", "type"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        provider_id = data["id"].strip()
        provider_type = data["type"]

        # Validate provider ID format
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", provider_id):
            return jsonify(
                {
                    "error": "Provider ID must start with a letter and contain only letters, numbers, hyphens, and underscores"
                }
            ), 400

        # Check if provider already exists in registry
        if registry.get_provider(provider_id):
            return jsonify({"error": f"Provider '{provider_id}' already exists"}), 409

        # Currently only Ollama providers can be created via UI
        if provider_type == "ollama":
            base_url = data.get("base_url", "").strip()
            if not base_url:
                return jsonify(
                    {"error": "Base URL is required for Ollama providers"}
                ), 400

            with get_db_context() as db:
                # Check DB for existing instance
                existing = (
                    db.query(OllamaInstance)
                    .filter(OllamaInstance.name == provider_id)
                    .first()
                )
                if existing:
                    return jsonify(
                        {"error": f"Provider '{provider_id}' already exists"}
                    ), 409

                # Create new instance in DB
                instance = OllamaInstance(
                    name=provider_id,
                    base_url=base_url,
                    enabled=data.get("enabled", True),
                )
                db.add(instance)
                db.commit()

                # Dynamically register the provider
                provider = OllamaProvider(name=provider_id, base_url=base_url)
                registry.register(provider)

                return jsonify(
                    {
                        "id": provider_id,
                        "type": "ollama",
                        "base_url": base_url,
                        "enabled": True,
                        "has_api_key": True,  # Ollama doesn't need API key
                        "model_count": len(provider.get_models()),
                    }
                ), 201
        else:
            # Other provider types (openai-compatible, anthropic, etc.) are system providers
            # They are seeded from defaults and cannot be added via UI
            return jsonify(
                {
                    "error": f"Provider type '{provider_type}' is a system provider. Only Ollama providers can be added via UI."
                }
            ), 400

    @admin.route("/api/providers/<provider_id>", methods=["PUT"])
    @require_auth_api
    def update_provider(provider_id: str):
        """Update a provider."""
        from db import OllamaInstance, Provider
        from providers import registry
        from providers.loader import clear_config_cache

        data = request.get_json() or {}

        # Check if it's an Ollama instance in DB (custom Ollama providers)
        with get_db_context() as db:
            instance = (
                db.query(OllamaInstance)
                .filter(OllamaInstance.name == provider_id)
                .first()
            )
            if instance:
                # Update Ollama instance fields
                if "base_url" in data:
                    instance.base_url = data["base_url"].strip()
                if "enabled" in data:
                    instance.enabled = data["enabled"]

                db.commit()

                # Update the provider in registry if it exists
                provider = registry.get_provider(provider_id)
                if provider:
                    provider.base_url = instance.base_url

                return jsonify(
                    {
                        "id": provider_id,
                        "type": "ollama",
                        "base_url": instance.base_url,
                        "enabled": instance.enabled,
                        "has_api_key": True,
                        "model_count": len(provider.get_models()) if provider else 0,
                    }
                )

        # Check if it's a system/DB provider
        with get_db_context() as db:
            provider_record = (
                db.query(Provider).filter(Provider.id == provider_id).first()
            )
            if not provider_record:
                return jsonify({"error": f"Provider '{provider_id}' not found"}), 404

            # Only allow updating enabled status for system providers
            if "enabled" in data:
                provider_record.enabled = data["enabled"]
                db.commit()

                # Clear config cache so changes take effect
                clear_config_cache()

                # Also update the provider instance in the registry immediately
                provider_instance = registry.get_provider(provider_id)
                if provider_instance:
                    provider_instance.enabled = data["enabled"]

                return jsonify(
                    {
                        "id": provider_id,
                        "type": provider_record.type,
                        "enabled": provider_record.enabled,
                    }
                )

            return jsonify(
                {"error": "Only 'enabled' can be updated for system providers"}
            ), 400

    @admin.route("/api/providers/<provider_id>", methods=["DELETE"])
    @require_auth_api
    def delete_provider(provider_id: str):
        """Delete a provider (only Ollama providers can be deleted via UI)."""
        from db import OllamaInstance, Provider
        from providers import registry

        # Check if this is a system provider (seeded from defaults)
        with get_db_context() as db:
            provider_record = (
                db.query(Provider).filter(Provider.id == provider_id).first()
            )
            if provider_record and provider_record.source == "system":
                return jsonify(
                    {"error": "System providers cannot be deleted via UI."}
                ), 400

        with get_db_context() as db:
            instance = (
                db.query(OllamaInstance)
                .filter(OllamaInstance.name == provider_id)
                .first()
            )
            if not instance:
                return jsonify({"error": f"Provider '{provider_id}' not found"}), 404

            db.delete(instance)
            db.commit()

            # Remove from registry
            if provider_id in registry._providers:
                del registry._providers[provider_id]

            return jsonify({"success": True})

    @admin.route("/api/providers/<provider_id>/test", methods=["POST"])
    @require_auth_api
    def test_provider(provider_id: str):
        """Test a provider's connection."""
        try:
            # Get provider from registry
            provider = registry.get_provider(provider_id)
            if not provider:
                return jsonify({"success": False, "error": "Provider not found"}), 404

            # Check provider type to determine test method
            provider_type = (
                getattr(provider, "type", None) or provider.__class__.__name__
            )

            # For Ollama providers, test the actual connection
            if "ollama" in provider_type.lower() or "ollama" in provider.name.lower():
                try:
                    import urllib.error
                    import urllib.request

                    base_url = getattr(provider, "base_url", "http://localhost:11434")
                    req = urllib.request.Request(
                        f"{base_url}/api/tags",
                        method="GET",
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        if resp.status == 200:
                            return jsonify(
                                {
                                    "success": True,
                                    "message": f"Connected to Ollama at {base_url}",
                                }
                            )
                        else:
                            return jsonify(
                                {
                                    "success": False,
                                    "error": f"Ollama returned status {resp.status}",
                                }
                            )
                except urllib.error.URLError as e:
                    return jsonify(
                        {
                            "success": False,
                            "error": f"Cannot connect to Ollama: {e.reason}",
                        }
                    )
                except Exception as e:
                    return jsonify(
                        {
                            "success": False,
                            "error": f"Ollama connection failed: {str(e)}",
                        }
                    )

            # For API-key based providers, check if configured
            try:
                is_configured = provider.is_configured()
            except Exception as e:
                return jsonify(
                    {
                        "success": False,
                        "error": f"Configuration check failed: {str(e)}",
                    }
                )

            if not is_configured:
                api_key_env = getattr(provider, "api_key_env", None)
                if api_key_env:
                    return jsonify(
                        {
                            "success": False,
                            "error": f"API key not found. Set {api_key_env} environment variable.",
                        }
                    )
                else:
                    return jsonify(
                        {
                            "success": False,
                            "error": "Provider is not configured.",
                        }
                    )

            # Provider is configured
            return jsonify(
                {
                    "success": True,
                    "message": "Provider is configured and ready",
                }
            )

        except Exception as e:
            return jsonify(
                {
                    "success": False,
                    "error": f"Test failed: {str(e)}",
                }
            )

    # -------------------------------------------------------------------------
    # Model API (Hybrid: System + Custom)
    # -------------------------------------------------------------------------

    @admin.route("/api/models", methods=["GET"])
    @require_auth_api
    def list_models():
        """List all models (system + custom) with metadata."""
        provider_id = request.args.get("provider")

        if provider_id:
            # Get models for specific provider
            try:
                models = get_all_models_with_metadata(provider_id)
                return jsonify(models)
            except Exception as e:
                logger.error(f"Failed to get models for {provider_id}: {e}")
                return jsonify({"error": f"Failed to load models: {str(e)}"}), 500

        # Get models for all providers (database + registry)
        all_models = []
        seen_providers = set()

        # First, get database providers
        for prov_name in get_all_provider_names():
            try:
                all_models.extend(get_all_models_with_metadata(prov_name))
            except Exception as e:
                logger.warning(f"Failed to get models for {prov_name}: {e}")
            seen_providers.add(prov_name)

        # Then, get any additional providers from registry (custom DB providers)
        for provider in registry.get_all_providers():
            if provider.name not in seen_providers:
                try:
                    all_models.extend(get_all_models_with_metadata(provider.name))
                except Exception as e:
                    logger.warning(f"Failed to get models for {provider.name}: {e}")

        return jsonify(all_models)

    @admin.route("/api/models/override", methods=["POST"])
    @require_auth_api
    def set_model_override():
        """Create or update an override for a system model."""
        data = request.get_json() or {}

        required = ["provider_id", "model_id"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        provider_id = data["provider_id"]
        model_id = data["model_id"]
        disabled = data.get("disabled", False)

        with get_db_context() as db:
            # Find existing override or create new
            override = (
                db.query(ModelOverride)
                .filter(
                    ModelOverride.provider_id == provider_id,
                    ModelOverride.model_id == model_id,
                )
                .first()
            )

            if override:
                override.disabled = disabled
                # Update optional override fields
                if "input_cost" in data:
                    override.input_cost = (
                        data["input_cost"] if data["input_cost"] else None
                    )
                if "output_cost" in data:
                    override.output_cost = (
                        data["output_cost"] if data["output_cost"] else None
                    )
                if "cache_read_multiplier" in data:
                    override.cache_read_multiplier = (
                        data["cache_read_multiplier"]
                        if data["cache_read_multiplier"]
                        else None
                    )
                if "cache_write_multiplier" in data:
                    override.cache_write_multiplier = (
                        data["cache_write_multiplier"]
                        if data["cache_write_multiplier"]
                        else None
                    )
                if "capabilities" in data:
                    override.capabilities = (
                        data["capabilities"] if data["capabilities"] else None
                    )
                if "context_length" in data:
                    override.context_length = (
                        data["context_length"] if data["context_length"] else None
                    )
                if "description" in data:
                    override.description = (
                        data["description"] if data["description"] else None
                    )
                # Provider quirks
                if "use_max_completion_tokens" in data:
                    override.use_max_completion_tokens = data[
                        "use_max_completion_tokens"
                    ]
                if "supports_system_prompt" in data:
                    override.supports_system_prompt = data["supports_system_prompt"]
                if "unsupported_params" in data:
                    override.unsupported_params = (
                        data["unsupported_params"]
                        if data["unsupported_params"]
                        else None
                    )
            else:
                override = ModelOverride(
                    provider_id=provider_id,
                    model_id=model_id,
                    disabled=disabled,
                    input_cost=data.get("input_cost"),
                    output_cost=data.get("output_cost"),
                    cache_read_multiplier=data.get("cache_read_multiplier"),
                    cache_write_multiplier=data.get("cache_write_multiplier"),
                    capabilities=data.get("capabilities"),
                    context_length=data.get("context_length"),
                    description=data.get("description"),
                    use_max_completion_tokens=data.get("use_max_completion_tokens"),
                    supports_system_prompt=data.get("supports_system_prompt"),
                )
                # Handle unsupported_params via setter
                if data.get("unsupported_params"):
                    override.unsupported_params = data["unsupported_params"]
                db.add(override)

            db.commit()
            return jsonify(override.to_dict())

    @admin.route("/api/models/override/<provider_id>/<model_id>", methods=["DELETE"])
    @require_auth_api
    def delete_model_override(provider_id: str, model_id: str):
        """Remove an override for a system model (re-enables it)."""
        with get_db_context() as db:
            override = (
                db.query(ModelOverride)
                .filter(
                    ModelOverride.provider_id == provider_id,
                    ModelOverride.model_id == model_id,
                )
                .first()
            )

            if override:
                db.delete(override)
                db.commit()

            return jsonify({"success": True})

    @admin.route("/api/models/custom", methods=["POST"])
    @require_auth_api
    def create_custom_model():
        """Create a new custom model."""
        data = request.get_json() or {}

        required = ["provider_id", "model_id", "family"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        with get_db_context() as db:
            # Check if custom model already exists
            existing = (
                db.query(CustomModel)
                .filter(
                    CustomModel.provider_id == data["provider_id"],
                    CustomModel.model_id == data["model_id"],
                )
                .first()
            )
            if existing:
                return jsonify({"error": "Custom model already exists"}), 409

            model = CustomModel(
                provider_id=data["provider_id"],
                model_id=data["model_id"],
                family=data["family"],
                description=data.get("description"),
                context_length=data.get("context_length", 128000),
                supports_system_prompt=data.get("supports_system_prompt", True),
                use_max_completion_tokens=data.get("use_max_completion_tokens", False),
                enabled=data.get("enabled", True),
                input_cost=data.get("input_cost"),
                output_cost=data.get("output_cost"),
                cache_read_multiplier=data.get("cache_read_multiplier"),
                cache_write_multiplier=data.get("cache_write_multiplier"),
            )
            model.capabilities = data.get("capabilities", [])
            model.unsupported_params = data.get("unsupported_params", [])

            db.add(model)
            db.commit()

            return jsonify(model.to_dict()), 201

    @admin.route("/api/models/custom/<int:db_id>", methods=["PUT"])
    @require_auth_api
    def update_custom_model(db_id: int):
        """Update a custom model."""
        data = request.get_json() or {}

        with get_db_context() as db:
            model = db.query(CustomModel).filter(CustomModel.id == db_id).first()
            if not model:
                return jsonify({"error": "Custom model not found"}), 404

            # Update allowed fields
            if "model_id" in data:
                model.model_id = data["model_id"]
            if "family" in data:
                model.family = data["family"]
            if "description" in data:
                model.description = data["description"]
            if "context_length" in data:
                model.context_length = data["context_length"]
            if "capabilities" in data:
                model.capabilities = data["capabilities"]
            if "unsupported_params" in data:
                model.unsupported_params = data["unsupported_params"]
            if "supports_system_prompt" in data:
                model.supports_system_prompt = data["supports_system_prompt"]
            if "use_max_completion_tokens" in data:
                model.use_max_completion_tokens = data["use_max_completion_tokens"]
            if "enabled" in data:
                model.enabled = data["enabled"]
            if "input_cost" in data:
                model.input_cost = data["input_cost"] if data["input_cost"] else None
            if "output_cost" in data:
                model.output_cost = data["output_cost"] if data["output_cost"] else None
            if "cache_read_multiplier" in data:
                model.cache_read_multiplier = (
                    data["cache_read_multiplier"]
                    if data["cache_read_multiplier"]
                    else None
                )
            if "cache_write_multiplier" in data:
                model.cache_write_multiplier = (
                    data["cache_write_multiplier"]
                    if data["cache_write_multiplier"]
                    else None
                )

            db.commit()
            return jsonify(model.to_dict())

    @admin.route("/api/models/custom/<int:db_id>", methods=["DELETE"])
    @require_auth_api
    def delete_custom_model(db_id: int):
        """Delete a custom model."""
        with get_db_context() as db:
            model = db.query(CustomModel).filter(CustomModel.id == db_id).first()
            if not model:
                return jsonify({"error": "Custom model not found"}), 404

            db.delete(model)
            db.commit()
            return jsonify({"success": True})

    # -------------------------------------------------------------------------
    # Settings API
    # -------------------------------------------------------------------------

    @admin.route("/api/settings", methods=["GET"])
    @require_auth_api
    def get_settings():
        """Get all settings (excluding sensitive ones)."""
        sensitive_keys = {Setting.KEY_ADMIN_PASSWORD_HASH, Setting.KEY_SESSION_SECRET}

        with get_db_context() as db:
            settings = db.query(Setting).filter(~Setting.key.in_(sensitive_keys)).all()
            return jsonify({s.key: s.value for s in settings})

    @admin.route("/api/settings", methods=["PUT"])
    @require_auth_api
    def update_settings():
        """Update settings."""
        data = request.get_json() or {}

        # Don't allow updating sensitive settings via this endpoint
        sensitive_keys = {Setting.KEY_ADMIN_PASSWORD_HASH, Setting.KEY_SESSION_SECRET}

        with get_db_context() as db:
            for key, value in data.items():
                if key in sensitive_keys:
                    continue

                setting = db.query(Setting).filter(Setting.key == key).first()
                if setting:
                    setting.value = value
                else:
                    db.add(Setting(key=key, value=value))

            db.commit()
            return jsonify({"success": True})

    @admin.route("/api/settings/password", methods=["PUT"])
    @require_auth_api
    def change_password():
        """Change the admin password."""
        data = request.get_json() or {}

        new_password = data.get("new_password")
        if not new_password or len(new_password) < 8:
            return jsonify({"error": "Password must be at least 8 characters"}), 400

        set_admin_password(new_password)
        return jsonify({"success": True})

    # -------------------------------------------------------------------------
    # Config Import/Export
    # -------------------------------------------------------------------------

    @admin.route("/api/config/export", methods=["GET"])
    @require_auth_api
    def export_config():
        """Export database configuration as JSON for backup/migration."""
        from db.models import (
            Alias,
            CustomModel,
            CustomProvider,
            DailyStats,
            ModelOverride,
            OllamaInstance,
            Provider,
            RequestLog,
            Setting,
            SmartRouter,
        )

        # Check if request logs should be included
        include_logs = request.args.get("include_logs", "false").lower() == "true"

        with get_db_context() as db:
            export_data = {
                "version": "1.0",
                "exported_at": datetime.utcnow().isoformat(),
                "settings": [
                    {"key": s.key, "value": s.value}
                    for s in db.query(Setting).all()
                    # Exclude password hash from export
                    if s.key != Setting.KEY_ADMIN_PASSWORD_HASH
                ],
                "model_overrides": [
                    {
                        "provider_id": o.provider_id,
                        "model_id": o.model_id,
                        "disabled": o.disabled,
                        "input_cost": o.input_cost,
                        "output_cost": o.output_cost,
                        "cache_read_multiplier": o.cache_read_multiplier,
                        "cache_write_multiplier": o.cache_write_multiplier,
                        "capabilities": o.capabilities,
                        "context_length": o.context_length,
                        "description": o.description,
                    }
                    for o in db.query(ModelOverride).all()
                ],
                "custom_models": [
                    {
                        "provider_id": m.provider_id,
                        "model_id": m.model_id,
                        "family": m.family,
                        "description": m.description,
                        "context_length": m.context_length,
                        "capabilities": m.capabilities,
                        "unsupported_params": m.unsupported_params,
                        "supports_system_prompt": m.supports_system_prompt,
                        "use_max_completion_tokens": m.use_max_completion_tokens,
                        "enabled": m.enabled,
                        "input_cost": m.input_cost,
                        "output_cost": m.output_cost,
                        "cache_read_multiplier": m.cache_read_multiplier,
                        "cache_write_multiplier": m.cache_write_multiplier,
                    }
                    for m in db.query(CustomModel).all()
                ],
                "ollama_instances": [
                    {
                        "name": o.name,
                        "base_url": o.base_url,
                        "enabled": o.enabled,
                    }
                    for o in db.query(OllamaInstance).all()
                ],
                "custom_providers": [
                    {
                        "name": p.name,
                        "type": p.type,
                        "base_url": p.base_url,
                        "api_key_env": p.api_key_env,
                        "enabled": p.enabled,
                    }
                    for p in db.query(CustomProvider).all()
                ],
                "aliases": [
                    {
                        "name": a.name,
                        "target_model": a.target_model,
                        "tags": a.tags,
                        "description": a.description,
                        "enabled": a.enabled,
                    }
                    for a in db.query(Alias).all()
                ],
                "smart_routers": [
                    {
                        "name": r.name,
                        "designator_model": r.designator_model,
                        "purpose": r.purpose,
                        "candidates": r.candidates,
                        "strategy": r.strategy,
                        "fallback_model": r.fallback_model,
                        "session_ttl": r.session_ttl,
                        "tags": r.tags,
                        "description": r.description,
                        "enabled": r.enabled,
                    }
                    for r in db.query(SmartRouter).all()
                ],
                # Provider enabled/disabled state (only system providers)
                "providers": [
                    {
                        "id": p.id,
                        "enabled": p.enabled,
                    }
                    for p in db.query(Provider)
                    .filter(Provider.source == "system")
                    .all()
                ],
            }

            # Optionally include request logs
            if include_logs:
                export_data["request_logs"] = [
                    {
                        "timestamp": log.timestamp.isoformat()
                        if log.timestamp
                        else None,
                        "client_ip": log.client_ip,
                        "hostname": log.hostname,
                        "tag": log.tag,
                        "alias": log.alias,
                        "is_designator": log.is_designator,
                        "router_name": log.router_name,
                        "provider_id": log.provider_id,
                        "model_id": log.model_id,
                        "endpoint": log.endpoint,
                        "input_tokens": log.input_tokens,
                        "output_tokens": log.output_tokens,
                        "reasoning_tokens": log.reasoning_tokens,
                        "cached_input_tokens": log.cached_input_tokens,
                        "cache_creation_tokens": log.cache_creation_tokens,
                        "cache_read_tokens": log.cache_read_tokens,
                        "cost": log.cost,
                        "response_time_ms": log.response_time_ms,
                        "status_code": log.status_code,
                        "error_message": log.error_message,
                        "is_streaming": log.is_streaming,
                    }
                    for log in db.query(RequestLog).order_by(RequestLog.timestamp).all()
                ]
                export_data["daily_stats"] = [
                    {
                        "date": stat.date.strftime("%Y-%m-%d") if stat.date else None,
                        "tag": stat.tag,
                        "provider_id": stat.provider_id,
                        "model_id": stat.model_id,
                        "alias": stat.alias,
                        "router_name": stat.router_name,
                        "request_count": stat.request_count,
                        "success_count": stat.success_count,
                        "error_count": stat.error_count,
                        "input_tokens": stat.input_tokens,
                        "output_tokens": stat.output_tokens,
                        "total_response_time_ms": stat.total_response_time_ms,
                        "estimated_cost": stat.estimated_cost,
                    }
                    for stat in db.query(DailyStats).order_by(DailyStats.date).all()
                ]

        response = jsonify(export_data)
        response.headers["Content-Disposition"] = (
            f"attachment; filename=llm-relay-config-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        )
        return response

    @admin.route("/api/config/import", methods=["POST"])
    @require_auth_api
    def import_config():
        """Import database configuration from JSON backup."""
        from db.models import (
            Alias,
            CustomModel,
            CustomProvider,
            DailyStats,
            ModelOverride,
            OllamaInstance,
            Provider,
            RequestLog,
            Setting,
            SmartRouter,
        )

        try:
            data = request.get_json()
        except Exception as e:
            return jsonify({"success": False, "error": f"Invalid JSON: {str(e)}"}), 400

        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # No version validation - import any recognized fields, ignore unknown ones

        stats = {
            "settings": 0,
            "model_overrides": 0,
            "custom_models": 0,
            "ollama_instances": 0,
            "custom_providers": 0,
            "aliases": 0,
            "smart_routers": 0,
            "providers": 0,
            "request_logs": 0,
            "daily_stats": 0,
        }

        try:
            with get_db_context() as db:
                # Import settings (skip password)
                for s in data.get("settings", []):
                    if s["key"] == Setting.KEY_ADMIN_PASSWORD_HASH:
                        continue
                    existing = db.query(Setting).filter(Setting.key == s["key"]).first()
                    if existing:
                        existing.value = s["value"]
                    else:
                        db.add(Setting(key=s["key"], value=s["value"]))
                    stats["settings"] += 1

                # Import model overrides
                for o in data.get("model_overrides", []):
                    existing = (
                        db.query(ModelOverride)
                        .filter(
                            ModelOverride.provider_id == o["provider_id"],
                            ModelOverride.model_id == o["model_id"],
                        )
                        .first()
                    )
                    if existing:
                        existing.disabled = o.get("disabled", False)
                        existing.input_cost = o.get("input_cost")
                        existing.output_cost = o.get("output_cost")
                        existing.cache_read_multiplier = o.get("cache_read_multiplier")
                        existing.cache_write_multiplier = o.get(
                            "cache_write_multiplier"
                        )
                        existing.capabilities = o.get("capabilities")
                        existing.context_length = o.get("context_length")
                        existing.description = o.get("description")
                    else:
                        override = ModelOverride(
                            provider_id=o["provider_id"],
                            model_id=o["model_id"],
                            disabled=o.get("disabled", False),
                            input_cost=o.get("input_cost"),
                            output_cost=o.get("output_cost"),
                            cache_read_multiplier=o.get("cache_read_multiplier"),
                            cache_write_multiplier=o.get("cache_write_multiplier"),
                            context_length=o.get("context_length"),
                            description=o.get("description"),
                        )
                        override.capabilities = o.get("capabilities")
                        db.add(override)
                    stats["model_overrides"] += 1

                # Import custom models
                for m in data.get("custom_models", []):
                    existing = (
                        db.query(CustomModel)
                        .filter(
                            CustomModel.provider_id == m["provider_id"],
                            CustomModel.model_id == m["model_id"],
                        )
                        .first()
                    )
                    if existing:
                        existing.family = m.get("family")
                        existing.description = m.get("description")
                        existing.context_length = m.get("context_length", 128000)
                        existing.capabilities = m.get("capabilities", [])
                        existing.unsupported_params = m.get("unsupported_params", [])
                        existing.supports_system_prompt = m.get(
                            "supports_system_prompt", True
                        )
                        existing.use_max_completion_tokens = m.get(
                            "use_max_completion_tokens", False
                        )
                        existing.enabled = m.get("enabled", True)
                        existing.input_cost = m.get("input_cost")
                        existing.output_cost = m.get("output_cost")
                        existing.cache_read_multiplier = m.get("cache_read_multiplier")
                        existing.cache_write_multiplier = m.get(
                            "cache_write_multiplier"
                        )
                    else:
                        model = CustomModel(
                            provider_id=m["provider_id"],
                            model_id=m["model_id"],
                            family=m.get("family"),
                            description=m.get("description"),
                            context_length=m.get("context_length", 128000),
                            supports_system_prompt=m.get(
                                "supports_system_prompt", True
                            ),
                            use_max_completion_tokens=m.get(
                                "use_max_completion_tokens", False
                            ),
                            enabled=m.get("enabled", True),
                            input_cost=m.get("input_cost"),
                            output_cost=m.get("output_cost"),
                            cache_read_multiplier=m.get("cache_read_multiplier"),
                            cache_write_multiplier=m.get("cache_write_multiplier"),
                        )
                        model.capabilities = m.get("capabilities", [])
                        model.unsupported_params = m.get("unsupported_params", [])
                        db.add(model)
                    stats["custom_models"] += 1

                # Import Ollama instances
                for o in data.get("ollama_instances", []):
                    existing = (
                        db.query(OllamaInstance)
                        .filter(OllamaInstance.name == o["name"])
                        .first()
                    )
                    if existing:
                        existing.base_url = o["base_url"]
                        existing.enabled = o.get("enabled", True)
                    else:
                        db.add(
                            OllamaInstance(
                                name=o["name"],
                                base_url=o["base_url"],
                                enabled=o.get("enabled", True),
                            )
                        )
                    stats["ollama_instances"] += 1

                # Import custom providers
                for p in data.get("custom_providers", []):
                    existing = (
                        db.query(CustomProvider)
                        .filter(CustomProvider.name == p["name"])
                        .first()
                    )
                    if existing:
                        existing.type = p["type"]
                        existing.base_url = p["base_url"]
                        existing.api_key_env = p.get("api_key_env")
                        existing.enabled = p.get("enabled", True)
                    else:
                        db.add(
                            CustomProvider(
                                name=p["name"],
                                type=p["type"],
                                base_url=p["base_url"],
                                api_key_env=p.get("api_key_env"),
                                enabled=p.get("enabled", True),
                            )
                        )
                    stats["custom_providers"] += 1

                # Import aliases
                for a in data.get("aliases", []):
                    existing = db.query(Alias).filter(Alias.name == a["name"]).first()
                    if existing:
                        existing.target_model = a["target_model"]
                        existing.tags = a.get("tags", [])
                        existing.description = a.get("description")
                        existing.enabled = a.get("enabled", True)
                    else:
                        alias = Alias(
                            name=a["name"],
                            target_model=a["target_model"],
                            description=a.get("description"),
                            enabled=a.get("enabled", True),
                        )
                        alias.tags = a.get("tags", [])
                        db.add(alias)
                    stats["aliases"] += 1

                # Import smart routers
                for r in data.get("smart_routers", []):
                    existing = (
                        db.query(SmartRouter)
                        .filter(SmartRouter.name == r["name"])
                        .first()
                    )
                    if existing:
                        existing.designator_model = r["designator_model"]
                        existing.purpose = r["purpose"]
                        existing.candidates_json = json.dumps(r.get("candidates", []))
                        existing.strategy = r.get("strategy", "per_request")
                        existing.fallback_model = r["fallback_model"]
                        existing.session_ttl = r.get("session_ttl", 3600)
                        existing.tags = r.get("tags", [])
                        existing.description = r.get("description")
                        existing.enabled = r.get("enabled", True)
                    else:
                        router = SmartRouter(
                            name=r["name"],
                            designator_model=r["designator_model"],
                            purpose=r["purpose"],
                            candidates_json=json.dumps(r.get("candidates", [])),
                            strategy=r.get("strategy", "per_request"),
                            fallback_model=r["fallback_model"],
                            session_ttl=r.get("session_ttl", 3600),
                            description=r.get("description"),
                            enabled=r.get("enabled", True),
                        )
                        router.tags = r.get("tags", [])
                        db.add(router)
                    stats["smart_routers"] += 1

                # Import provider enabled/disabled state
                for p in data.get("providers", []):
                    existing = db.query(Provider).filter(Provider.id == p["id"]).first()
                    if existing:
                        existing.enabled = p.get("enabled", True)
                        stats["providers"] += 1
                db.flush()  # Ensure provider changes are written

                # Import request logs (if present)
                for log in data.get("request_logs", []):
                    # Parse timestamp
                    timestamp = None
                    if log.get("timestamp"):
                        try:
                            timestamp = datetime.fromisoformat(log["timestamp"])
                        except (ValueError, TypeError):
                            timestamp = datetime.utcnow()

                    db.add(
                        RequestLog(
                            timestamp=timestamp or datetime.utcnow(),
                            client_ip=log.get("client_ip", "unknown"),
                            hostname=log.get("hostname"),
                            tag=log.get("tag", "unknown"),
                            alias=log.get("alias"),
                            is_designator=log.get("is_designator", False),
                            router_name=log.get("router_name"),
                            provider_id=log.get("provider_id", "unknown"),
                            model_id=log.get("model_id", "unknown"),
                            endpoint=log.get("endpoint", "unknown"),
                            input_tokens=log.get("input_tokens", 0),
                            output_tokens=log.get("output_tokens", 0),
                            reasoning_tokens=log.get("reasoning_tokens"),
                            cached_input_tokens=log.get("cached_input_tokens"),
                            cache_creation_tokens=log.get("cache_creation_tokens"),
                            cache_read_tokens=log.get("cache_read_tokens"),
                            cost=log.get("cost"),
                            response_time_ms=log.get("response_time_ms", 0),
                            status_code=log.get("status_code", 200),
                            error_message=log.get("error_message"),
                            is_streaming=log.get("is_streaming", False),
                        )
                    )
                    stats["request_logs"] += 1

                # Import daily stats (if present)
                for stat in data.get("daily_stats", []):
                    # Parse date
                    date = None
                    if stat.get("date"):
                        try:
                            date = datetime.strptime(stat["date"], "%Y-%m-%d")
                        except (ValueError, TypeError):
                            continue  # Skip invalid dates

                    if date:
                        db.add(
                            DailyStats(
                                date=date,
                                tag=stat.get("tag"),
                                provider_id=stat.get("provider_id"),
                                model_id=stat.get("model_id"),
                                alias=stat.get("alias"),
                                router_name=stat.get("router_name"),
                                request_count=stat.get("request_count", 0),
                                success_count=stat.get("success_count", 0),
                                error_count=stat.get("error_count", 0),
                                input_tokens=stat.get("input_tokens", 0),
                                output_tokens=stat.get("output_tokens", 0),
                                total_response_time_ms=stat.get(
                                    "total_response_time_ms", 0
                                ),
                                estimated_cost=stat.get("estimated_cost", 0.0),
                            )
                        )
                        stats["daily_stats"] += 1

        except Exception as e:
            return jsonify({"success": False, "error": f"Import failed: {str(e)}"}), 500

        return jsonify({"success": True, "stats": stats})

    # -------------------------------------------------------------------------
    # Dashboard Stats API
    # -------------------------------------------------------------------------

    @admin.route("/api/stats", methods=["GET"])
    @require_auth_api
    def get_stats():
        """Get dashboard statistics."""
        from providers import registry
        from providers.loader import get_provider_config
        from providers.ollama_provider import OllamaProvider

        # Get provider enabled status from database
        provider_enabled = {}
        with get_db_context() as db:
            for p in db.query(Provider).all():
                provider_enabled[p.id] = p.enabled

        # Count providers from registry
        all_providers = registry.get_all_providers()
        provider_count = len(all_providers)

        # Build sets of configured and active providers
        configured_providers = set()  # Has API key or is Ollama
        active_providers = set()  # Configured AND enabled
        configured_count = 0

        for provider in all_providers:
            is_enabled = provider_enabled.get(provider.name, True)
            is_configured = False

            # Check if this is an Ollama-type provider (doesn't need API key)
            if isinstance(provider, OllamaProvider):
                if provider.is_configured():
                    is_configured = True
            else:
                # Other providers need an API key
                config = get_provider_config(provider.name)
                api_key_env = config.get("api_key_env")
                if api_key_env and os.environ.get(api_key_env):
                    is_configured = True

            if is_configured:
                configured_providers.add(provider.name)
                configured_count += 1
                if is_enabled:
                    active_providers.add(provider.name)

        # Count available models (all models from configured providers)
        available_models = 0
        for prov_name in get_all_provider_names():
            if prov_name not in configured_providers:
                continue
            models = get_all_models_with_metadata(prov_name)
            available_models += len(models)

        # Count active models (from active providers - enabled AND configured)
        active_models = 0
        for prov_name in get_all_provider_names():
            if prov_name not in active_providers:
                continue
            models = get_all_models_with_metadata(prov_name)
            active_models += sum(1 for m in models if m.get("enabled", True))

        return jsonify(
            {
                "providers": {
                    "total": provider_count,
                    "configured": configured_count,
                },
                "models": {
                    "active": active_models,
                    "available": available_models,
                },
            }
        )

    # -------------------------------------------------------------------------
    # Ollama API (supports multiple Ollama instances)
    # -------------------------------------------------------------------------

    def _get_ollama_providers():
        """Get all Ollama-type providers from registry."""
        from providers import registry
        from providers.ollama_provider import OllamaProvider

        return [
            p for p in registry.get_all_providers() if isinstance(p, OllamaProvider)
        ]

    def _get_ollama_provider(provider_id: str):
        """Get a specific Ollama provider by name."""
        from providers import registry
        from providers.ollama_provider import OllamaProvider

        provider = registry.get_provider(provider_id)
        if provider and isinstance(provider, OllamaProvider):
            return provider
        return None

    @admin.route("/api/ollama/instances", methods=["GET"])
    @require_auth_api
    def list_ollama_instances():
        """List all configured Ollama instances (system + user-created)."""
        from db import OllamaInstance, Provider

        instances = []

        # Get source info from database
        provider_sources = {}
        with get_db_context() as db:
            for p in db.query(Provider).all():
                provider_sources[p.id] = p.source

        # Get instances from registry (includes both system and dynamically added)
        for provider in _get_ollama_providers():
            running = provider.is_configured()
            # Check if this is a system provider (seeded from defaults)
            is_system = provider_sources.get(provider.name) == "system"

            instances.append(
                {
                    "id": provider.name,
                    "base_url": provider.base_url,
                    "running": running,
                    "model_count": len(provider.get_models()) if running else 0,
                    "is_system": is_system,
                    "enabled": True,
                }
            )

        # Also include DB instances that aren't yet in the registry
        # (e.g., newly added ones that require restart)
        with get_db_context() as db:
            db_instances = db.query(OllamaInstance).all()
            registered_names = {p.name for p in _get_ollama_providers()}

            for inst in db_instances:
                if inst.name not in registered_names:
                    instances.append(
                        {
                            "id": inst.name,
                            "db_id": inst.id,
                            "base_url": inst.base_url,
                            "running": False,
                            "model_count": 0,
                            "is_system": False,
                            "enabled": inst.enabled,
                            "pending_restart": True,  # Needs restart to activate
                        }
                    )

        return jsonify(instances)

    @admin.route("/api/ollama/instances", methods=["POST"])
    @require_auth_api
    def create_ollama_instance():
        """Create a new Ollama instance."""
        from db import OllamaInstance
        from providers import registry
        from providers.ollama_provider import OllamaProvider

        data = request.get_json() or {}

        name = data.get("name", "").strip()
        base_url = data.get("base_url", "").strip()

        if not name:
            return jsonify({"error": "Name is required"}), 400
        if not base_url:
            return jsonify({"error": "Base URL is required"}), 400

        # Validate name format (alphanumeric, hyphens, underscores)
        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
            return jsonify(
                {
                    "error": "Name must start with a letter and contain only letters, numbers, hyphens, and underscores"
                }
            ), 400

        # Check for conflicts with existing providers
        if registry.get_provider(name):
            return jsonify({"error": f"Provider '{name}' already exists"}), 409

        with get_db_context() as db:
            # Check DB for existing instance
            existing = (
                db.query(OllamaInstance).filter(OllamaInstance.name == name).first()
            )
            if existing:
                return jsonify(
                    {"error": f"Ollama instance '{name}' already exists"}
                ), 409

            # Create new instance in DB
            instance = OllamaInstance(
                name=name,
                base_url=base_url,
                enabled=data.get("enabled", True),
            )
            db.add(instance)
            db.commit()

            # Dynamically register the provider
            provider = OllamaProvider(name=name, base_url=base_url)
            registry.register(provider)

            return jsonify(
                {
                    **instance.to_dict(),
                    "running": provider.is_configured(),
                }
            ), 201

    @admin.route("/api/ollama/instances/<instance_name>", methods=["PUT"])
    @require_auth_api
    def update_ollama_instance(instance_name: str):
        """Update an Ollama instance."""
        from db import OllamaInstance, Provider
        from providers import registry

        # Check if this is a system provider (seeded from defaults)
        with get_db_context() as db:
            provider_record = (
                db.query(Provider).filter(Provider.id == instance_name).first()
            )
            if provider_record and provider_record.source == "system":
                return jsonify(
                    {"error": "Cannot modify system-configured instances."}
                ), 400

        data = request.get_json() or {}

        with get_db_context() as db:
            instance = (
                db.query(OllamaInstance)
                .filter(OllamaInstance.name == instance_name)
                .first()
            )
            if not instance:
                return jsonify(
                    {"error": f"Ollama instance '{instance_name}' not found"}
                ), 404

            # Update fields
            if "base_url" in data:
                instance.base_url = data["base_url"].strip()
            if "enabled" in data:
                instance.enabled = data["enabled"]

            db.commit()

            # Update the provider in registry if it exists
            provider = registry.get_provider(instance_name)
            if provider:
                provider.base_url = instance.base_url

            return jsonify(instance.to_dict())

    @admin.route("/api/ollama/instances/<instance_name>", methods=["DELETE"])
    @require_auth_api
    def delete_ollama_instance(instance_name: str):
        """Delete an Ollama instance."""
        from db import OllamaInstance, Provider
        from providers import registry

        # Check if this is a system provider (seeded from defaults)
        with get_db_context() as db:
            provider_record = (
                db.query(Provider).filter(Provider.id == instance_name).first()
            )
            if provider_record and provider_record.source == "system":
                return jsonify(
                    {"error": "Cannot delete system-configured instances."}
                ), 400

        with get_db_context() as db:
            instance = (
                db.query(OllamaInstance)
                .filter(OllamaInstance.name == instance_name)
                .first()
            )
            if not instance:
                return jsonify(
                    {"error": f"Ollama instance '{instance_name}' not found"}
                ), 404

            db.delete(instance)
            db.commit()

            # Remove from registry
            if instance_name in registry._providers:
                del registry._providers[instance_name]

            return jsonify({"success": True})

    @admin.route("/api/ollama/<provider_id>/status", methods=["GET"])
    @require_auth_api
    def ollama_status(provider_id: str):
        """Check if a specific Ollama instance is running."""
        provider = _get_ollama_provider(provider_id)
        if not provider:
            return jsonify(
                {
                    "configured": False,
                    "running": False,
                    "error": f"Ollama provider '{provider_id}' not configured",
                }
            )

        running = provider.is_configured()
        return jsonify(
            {
                "configured": True,
                "running": running,
                "base_url": provider.base_url,
                "provider_id": provider_id,
            }
        )

    @admin.route("/api/ollama/<provider_id>/models", methods=["GET"])
    @require_auth_api
    def list_ollama_models(provider_id: str):
        """List models from a specific Ollama instance."""
        provider = _get_ollama_provider(provider_id)
        if not provider:
            return jsonify(
                {"error": f"Ollama provider '{provider_id}' not configured"}
            ), 404

        if not provider.is_configured():
            return jsonify({"error": "Ollama is not running"}), 503

        models = provider.get_raw_models()
        return jsonify({"models": models})

    @admin.route("/api/ollama/<provider_id>/models/<path:model_name>", methods=["GET"])
    @require_auth_api
    def get_ollama_model(provider_id: str, model_name: str):
        """Get detailed info about a specific model."""
        provider = _get_ollama_provider(provider_id)
        if not provider:
            return jsonify(
                {"error": f"Ollama provider '{provider_id}' not configured"}
            ), 404

        info = provider.get_model_info(model_name)
        if not info:
            return jsonify({"error": "Model not found"}), 404

        return jsonify(info)

    @admin.route("/api/ollama/<provider_id>/pull", methods=["POST"])
    @require_auth_api
    def pull_ollama_model(provider_id: str):
        """Pull a model to a specific Ollama instance."""
        import json

        from flask import Response, stream_with_context

        data = request.get_json() or {}
        model_name = data.get("name")

        if not model_name:
            return jsonify({"error": "Model name required"}), 400

        provider = _get_ollama_provider(provider_id)
        if not provider:
            return jsonify(
                {"error": f"Ollama provider '{provider_id}' not configured"}
            ), 404

        if not provider.is_configured():
            return jsonify({"error": "Ollama is not running"}), 503

        def generate():
            for progress in provider.pull_model(model_name):
                yield json.dumps(progress) + "\n"

        return Response(
            stream_with_context(generate()),
            mimetype="application/x-ndjson",
        )

    @admin.route(
        "/api/ollama/<provider_id>/models/<path:model_name>", methods=["DELETE"]
    )
    @require_auth_api
    def delete_ollama_model(provider_id: str, model_name: str):
        """Delete a model from a specific Ollama instance."""
        provider = _get_ollama_provider(provider_id)
        if not provider:
            return jsonify(
                {"error": f"Ollama provider '{provider_id}' not configured"}
            ), 404

        if not provider.is_configured():
            return jsonify({"error": "Ollama is not running"}), 503

        success = provider.delete_model(model_name)
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to delete model"}), 500

    @admin.route("/api/ollama/<provider_id>/refresh", methods=["POST"])
    @require_auth_api
    def refresh_ollama_models(provider_id: str):
        """Force refresh of model list for a specific Ollama instance."""
        provider = _get_ollama_provider(provider_id)
        if not provider:
            return jsonify(
                {"error": f"Ollama provider '{provider_id}' not configured"}
            ), 404

        if not provider.is_configured():
            return jsonify({"error": "Ollama is not running"}), 503

        provider._refresh_models()
        return jsonify(
            {
                "success": True,
                "model_count": len(provider.get_models()),
            }
        )

    # -------------------------------------------------------------------------
    # Usage Tracking Routes (v2.1)
    # -------------------------------------------------------------------------

    @admin.route("/usage")
    @require_auth
    def usage():
        """Usage statistics page."""
        return render_template("usage.html")

    @admin.route("/requests")
    @require_auth
    def requests_page():
        """Request log page."""
        return render_template("requests.html")

    def parse_usage_filters():
        """Parse common filter parameters from request args.

        Returns dict with:
        - start_date: datetime
        - end_date: datetime (or None for "now")
        - tags: list of tag strings (or None for all)
        - providers: list of provider strings (or None for all)
        - models: list of model strings (or None for all)
        """
        from datetime import datetime, timedelta

        # Date range - support both preset days and custom range
        # Accept both "start"/"end" and "start_date"/"end_date" for compatibility
        days = request.args.get("days")
        start = request.args.get("start") or request.args.get("start_date")
        end = request.args.get("end") or request.args.get("end_date")

        if start:
            # Custom date range
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d") if end else datetime.utcnow()
            # Include full end day
            end_date = end_date.replace(hour=23, minute=59, second=59)
        else:
            # Preset days
            days = int(days) if days else 30
            start_date = datetime.utcnow() - timedelta(days=days)
            end_date = None  # No upper bound

        # Tag filter
        tags_param = request.args.get("tags")
        tags = (
            [t.strip() for t in tags_param.split(",") if t.strip()]
            if tags_param
            else None
        )

        # Provider filter
        providers_param = request.args.get("providers")
        providers = (
            [p.strip() for p in providers_param.split(",") if p.strip()]
            if providers_param
            else None
        )

        # Model filter
        models_param = request.args.get("models")
        models = (
            [m.strip() for m in models_param.split(",") if m.strip()]
            if models_param
            else None
        )

        # Client filter
        clients_param = request.args.get("clients")
        clients = (
            [c.strip() for c in clients_param.split(",") if c.strip()]
            if clients_param
            else None
        )

        # Alias filter
        aliases_param = request.args.get("aliases")
        aliases = (
            [a.strip() for a in aliases_param.split(",") if a.strip()]
            if aliases_param
            else None
        )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "tags": tags,
            "providers": providers,
            "models": models,
            "clients": clients,
            "aliases": aliases,
        }

    @admin.route("/api/usage/filters", methods=["GET"])
    @require_auth_api
    def get_usage_filters():
        """Get available filter options (tags, providers, models, clients with data)."""
        from sqlalchemy import distinct, func

        with get_db_context() as db:
            # Get distinct tags that have usage data
            tags = [
                r[0]
                for r in db.query(distinct(DailyStats.tag))
                .filter(DailyStats.tag.isnot(None))
                .order_by(DailyStats.tag)
                .all()
            ]

            # Get distinct providers
            providers = [
                r[0]
                for r in db.query(distinct(DailyStats.provider_id))
                .filter(DailyStats.provider_id.isnot(None))
                .order_by(DailyStats.provider_id)
                .all()
            ]

            # Get distinct models (with their providers)
            model_results = (
                db.query(distinct(DailyStats.provider_id), DailyStats.model_id)
                .filter(
                    DailyStats.provider_id.isnot(None),
                    DailyStats.model_id.isnot(None),
                )
                .order_by(DailyStats.provider_id, DailyStats.model_id)
                .all()
            )
            models = [{"provider": r[0], "model": r[1]} for r in model_results]

            # Get distinct clients (hostname or IP) from RequestLog
            client_results = (
                db.query(
                    func.coalesce(RequestLog.hostname, RequestLog.client_ip).label(
                        "client"
                    )
                )
                .distinct()
                .order_by(func.coalesce(RequestLog.hostname, RequestLog.client_ip))
                .limit(100)  # Limit to prevent huge lists
                .all()
            )
            clients = [r[0] for r in client_results if r[0]]

            # Get distinct aliases (v3.1)
            aliases = [
                r[0]
                for r in db.query(distinct(DailyStats.alias))
                .filter(DailyStats.alias.isnot(None))
                .order_by(DailyStats.alias)
                .all()
            ]

            return jsonify(
                {
                    "tags": tags,
                    "providers": providers,
                    "models": models,
                    "clients": clients,
                    "aliases": aliases,
                }
            )

    @admin.route("/api/usage/summary", methods=["GET"])
    @require_auth_api
    def get_usage_summary():
        """Get usage summary for dashboard cards with optional filters."""
        from sqlalchemy import func

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Build base query - use RequestLog for filtered queries
            # since DailyStats pre-aggregation doesn't support arbitrary filters
            has_filters = (
                filters["tags"]
                or filters["providers"]
                or filters["models"]
                or filters["clients"]
                or filters["aliases"]
            )
            if has_filters:
                # Query from RequestLog for filtered results
                from sqlalchemy import or_

                query = db.query(
                    func.count(RequestLog.id),
                    func.sum(RequestLog.input_tokens),
                    func.sum(RequestLog.output_tokens),
                    func.count(RequestLog.id).filter(RequestLog.status_code < 400),
                    func.count(RequestLog.id).filter(RequestLog.status_code >= 400),
                    func.sum(RequestLog.cost),
                ).filter(RequestLog.timestamp >= filters["start_date"])

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["tags"]:
                    # Support multi-tag filtering (tag column may contain comma-separated tags)
                    tag_conditions = []
                    for tag in filters["tags"]:
                        tag_conditions.append(RequestLog.tag == tag)
                        tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                    query = query.filter(or_(*tag_conditions))

                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

                if filters["clients"]:
                    # Filter by client (hostname or IP)
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                stats = query.first()

                return jsonify(
                    {
                        "period_days": request.args.get("days", 30),
                        "total_requests": stats[0] or 0,
                        "total_input_tokens": stats[1] or 0,
                        "total_output_tokens": stats[2] or 0,
                        "total_tokens": (stats[1] or 0) + (stats[2] or 0),
                        "estimated_cost": round(stats[5] or 0, 4),
                        "success_count": stats[3] or 0,
                        "error_count": stats[4] or 0,
                    }
                )
            else:
                # No filters - use pre-aggregated DailyStats for performance
                query = db.query(
                    func.sum(DailyStats.request_count),
                    func.sum(DailyStats.input_tokens),
                    func.sum(DailyStats.output_tokens),
                    func.sum(DailyStats.estimated_cost),
                    func.sum(DailyStats.success_count),
                    func.sum(DailyStats.error_count),
                ).filter(
                    DailyStats.date >= filters["start_date"],
                    DailyStats.tag.is_(None),
                    DailyStats.provider_id.is_(None),
                    DailyStats.model_id.is_(None),
                )

                if filters["end_date"]:
                    query = query.filter(DailyStats.date <= filters["end_date"])

                stats = query.first()

                return jsonify(
                    {
                        "period_days": request.args.get("days", 30),
                        "total_requests": stats[0] or 0,
                        "total_input_tokens": stats[1] or 0,
                        "total_output_tokens": stats[2] or 0,
                        "total_tokens": (stats[1] or 0) + (stats[2] or 0),
                        "estimated_cost": round(stats[3] or 0, 4),
                        "success_count": stats[4] or 0,
                        "error_count": stats[5] or 0,
                    }
                )

    @admin.route("/api/usage/by-tag", methods=["GET"])
    @require_auth_api
    def get_usage_by_tag():
        """Get usage breakdown by tag with optional filters."""
        from sqlalchemy import func, or_

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Use RequestLog for filtered queries, DailyStats for unfiltered
            has_filters = (
                filters["providers"]
                or filters["models"]
                or filters["clients"]
                or filters["aliases"]
            )
            if has_filters:
                # Query from RequestLog when filtering
                query = db.query(
                    RequestLog.tag,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                    func.sum(RequestLog.cost).label("cost"),
                ).filter(
                    RequestLog.timestamp >= filters["start_date"],
                    RequestLog.tag.isnot(None),
                    RequestLog.tag != "",
                )

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                results = (
                    query.group_by(RequestLog.tag)
                    .order_by(func.count(RequestLog.id).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "tag": r.tag,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )
            else:
                # Use pre-aggregated DailyStats for performance
                query = db.query(
                    DailyStats.tag,
                    func.sum(DailyStats.request_count).label("requests"),
                    func.sum(DailyStats.input_tokens).label("input_tokens"),
                    func.sum(DailyStats.output_tokens).label("output_tokens"),
                    func.sum(DailyStats.estimated_cost).label("cost"),
                ).filter(
                    DailyStats.date >= filters["start_date"],
                    DailyStats.tag.isnot(None),
                    DailyStats.tag != "",
                    DailyStats.provider_id.is_(None),
                    DailyStats.model_id.is_(None),
                )

                if filters["end_date"]:
                    query = query.filter(DailyStats.date <= filters["end_date"])

                # If tag filter is provided, filter to those tags
                if filters["tags"]:
                    query = query.filter(DailyStats.tag.in_(filters["tags"]))

                results = (
                    query.group_by(DailyStats.tag)
                    .order_by(func.sum(DailyStats.request_count).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "tag": r.tag,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )

    @admin.route("/api/usage/by-provider", methods=["GET"])
    @require_auth_api
    def get_usage_by_provider():
        """Get usage breakdown by provider with optional filters."""
        from sqlalchemy import func, or_

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Use RequestLog for filtered queries, DailyStats for unfiltered
            has_filters = (
                filters["tags"]
                or filters["models"]
                or filters["clients"]
                or filters["aliases"]
            )
            if has_filters:
                # Query from RequestLog when filtering
                query = db.query(
                    RequestLog.provider_id,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                    func.sum(RequestLog.cost).label("cost"),
                ).filter(
                    RequestLog.timestamp >= filters["start_date"],
                    RequestLog.provider_id.isnot(None),
                )

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["tags"]:
                    # Support multi-tag filtering
                    tag_conditions = []
                    for tag in filters["tags"]:
                        tag_conditions.append(RequestLog.tag == tag)
                        tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                    query = query.filter(or_(*tag_conditions))

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                # If provider filter is provided, filter to those providers
                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                results = (
                    query.group_by(RequestLog.provider_id)
                    .order_by(func.count(RequestLog.id).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "provider_id": r.provider_id,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )
            else:
                # Use pre-aggregated DailyStats for performance
                query = db.query(
                    DailyStats.provider_id,
                    func.sum(DailyStats.request_count).label("requests"),
                    func.sum(DailyStats.input_tokens).label("input_tokens"),
                    func.sum(DailyStats.output_tokens).label("output_tokens"),
                    func.sum(DailyStats.estimated_cost).label("cost"),
                ).filter(
                    DailyStats.date >= filters["start_date"],
                    DailyStats.tag.is_(None),
                    DailyStats.provider_id.isnot(None),
                    DailyStats.model_id.is_(None),
                )

                if filters["end_date"]:
                    query = query.filter(DailyStats.date <= filters["end_date"])

                # If provider filter is provided, filter to those providers
                if filters["providers"]:
                    query = query.filter(
                        DailyStats.provider_id.in_(filters["providers"])
                    )

                results = (
                    query.group_by(DailyStats.provider_id)
                    .order_by(func.sum(DailyStats.request_count).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "provider_id": r.provider_id,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )

    @admin.route("/api/usage/by-model", methods=["GET"])
    @require_auth_api
    def get_usage_by_model():
        """Get usage breakdown by model with optional filters."""
        from sqlalchemy import func, or_

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Use RequestLog for filtered queries, DailyStats for unfiltered
            has_filters = filters["tags"] or filters["clients"] or filters["aliases"]
            if has_filters:
                # Query from RequestLog when filtering
                query = db.query(
                    RequestLog.provider_id,
                    RequestLog.model_id,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                    func.sum(RequestLog.cost).label("cost"),
                ).filter(
                    RequestLog.timestamp >= filters["start_date"],
                    RequestLog.provider_id.isnot(None),
                    RequestLog.model_id.isnot(None),
                )

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["tags"]:
                    # Support multi-tag filtering
                    tag_conditions = []
                    for tag in filters["tags"]:
                        tag_conditions.append(RequestLog.tag == tag)
                        tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                    query = query.filter(or_(*tag_conditions))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

                results = (
                    query.group_by(RequestLog.provider_id, RequestLog.model_id)
                    .order_by(func.count(RequestLog.id).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "provider_id": r.provider_id,
                            "model_id": r.model_id,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )
            else:
                # Use pre-aggregated DailyStats for performance
                query = db.query(
                    DailyStats.provider_id,
                    DailyStats.model_id,
                    func.sum(DailyStats.request_count).label("requests"),
                    func.sum(DailyStats.input_tokens).label("input_tokens"),
                    func.sum(DailyStats.output_tokens).label("output_tokens"),
                    func.sum(DailyStats.estimated_cost).label("cost"),
                ).filter(
                    DailyStats.date >= filters["start_date"],
                    DailyStats.tag.is_(None),
                    DailyStats.provider_id.isnot(None),
                    DailyStats.model_id.isnot(None),
                )

                if filters["end_date"]:
                    query = query.filter(DailyStats.date <= filters["end_date"])

                if filters["providers"]:
                    query = query.filter(
                        DailyStats.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(DailyStats.model_id.in_(filters["models"]))

                results = (
                    query.group_by(DailyStats.provider_id, DailyStats.model_id)
                    .order_by(func.sum(DailyStats.request_count).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "provider_id": r.provider_id,
                            "model_id": r.model_id,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )

    @admin.route("/api/usage/by-client", methods=["GET"])
    @require_auth_api
    def get_usage_by_client():
        """Get usage breakdown by client (hostname or IP) with optional filters."""
        from sqlalchemy import func, or_

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Always query from RequestLog since DailyStats doesn't have client dimension
            client_col = func.coalesce(RequestLog.hostname, RequestLog.client_ip)

            query = db.query(
                client_col.label("client"),
                func.count(RequestLog.id).label("requests"),
                func.sum(RequestLog.input_tokens).label("input_tokens"),
                func.sum(RequestLog.output_tokens).label("output_tokens"),
                func.sum(RequestLog.cost).label("cost"),
            ).filter(RequestLog.timestamp >= filters["start_date"])

            if filters["end_date"]:
                query = query.filter(RequestLog.timestamp <= filters["end_date"])

            if filters["tags"]:
                # Support multi-tag filtering
                tag_conditions = []
                for tag in filters["tags"]:
                    tag_conditions.append(RequestLog.tag == tag)
                    tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                    tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                    tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                query = query.filter(or_(*tag_conditions))

            if filters["providers"]:
                query = query.filter(RequestLog.provider_id.in_(filters["providers"]))

            if filters["models"]:
                query = query.filter(RequestLog.model_id.in_(filters["models"]))

            if filters["clients"]:
                client_conditions = []
                for client in filters["clients"]:
                    client_conditions.append(RequestLog.hostname == client)
                    client_conditions.append(RequestLog.client_ip == client)
                query = query.filter(or_(*client_conditions))

            if filters["aliases"]:
                query = query.filter(RequestLog.alias.in_(filters["aliases"]))

            results = (
                query.group_by(client_col)
                .order_by(func.count(RequestLog.id).desc())
                .all()
            )

            return jsonify(
                [
                    {
                        "client": r.client,
                        "requests": r.requests or 0,
                        "input_tokens": r.input_tokens or 0,
                        "output_tokens": r.output_tokens or 0,
                        "total_tokens": (r.input_tokens or 0) + (r.output_tokens or 0),
                        "cost": round(r.cost or 0, 4),
                    }
                    for r in results
                ]
            )

    @admin.route("/api/usage/by-alias", methods=["GET"])
    @require_auth_api
    def get_usage_by_alias():
        """Get usage breakdown by alias (v3.1)."""
        from sqlalchemy import func, or_

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Use RequestLog for filtered queries, DailyStats for unfiltered
            has_filters = (
                filters["tags"]
                or filters["providers"]
                or filters["models"]
                or filters["clients"]
            )
            if has_filters:
                # Query from RequestLog when filtering
                query = db.query(
                    RequestLog.alias,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                    func.sum(RequestLog.cost).label("cost"),
                ).filter(
                    RequestLog.timestamp >= filters["start_date"],
                    RequestLog.alias.isnot(None),
                )

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["tags"]:
                    tag_conditions = []
                    for tag in filters["tags"]:
                        tag_conditions.append(RequestLog.tag == tag)
                        tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                    query = query.filter(or_(*tag_conditions))

                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                results = (
                    query.group_by(RequestLog.alias)
                    .order_by(func.count(RequestLog.id).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "alias": r.alias,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )
            else:
                # Use pre-aggregated DailyStats for performance
                query = db.query(
                    DailyStats.alias,
                    func.sum(DailyStats.request_count).label("requests"),
                    func.sum(DailyStats.input_tokens).label("input_tokens"),
                    func.sum(DailyStats.output_tokens).label("output_tokens"),
                    func.sum(DailyStats.estimated_cost).label("cost"),
                ).filter(
                    DailyStats.date >= filters["start_date"],
                    DailyStats.alias.isnot(None),
                    DailyStats.tag.is_(None),  # Get alias-level aggregation only
                    DailyStats.provider_id.is_(None),
                    DailyStats.model_id.is_(None),
                )

                if filters["end_date"]:
                    query = query.filter(DailyStats.date <= filters["end_date"])

                if filters["aliases"]:
                    query = query.filter(DailyStats.alias.in_(filters["aliases"]))

                results = (
                    query.group_by(DailyStats.alias)
                    .order_by(func.sum(DailyStats.request_count).desc())
                    .all()
                )

                return jsonify(
                    [
                        {
                            "alias": r.alias,
                            "requests": r.requests or 0,
                            "input_tokens": r.input_tokens or 0,
                            "output_tokens": r.output_tokens or 0,
                            "total_tokens": (r.input_tokens or 0)
                            + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )

    @admin.route("/api/usage/timeseries", methods=["GET"])
    @require_auth_api
    def get_usage_timeseries():
        """Get time series data for charts with optional filters."""
        from sqlalchemy import func, or_

        filters = parse_usage_filters()

        with get_db_context() as db:
            # Use RequestLog for filtered queries, DailyStats for unfiltered
            has_filters = (
                filters["tags"]
                or filters["providers"]
                or filters["models"]
                or filters["clients"]
                or filters["aliases"]
            )
            if has_filters:
                # Query from RequestLog when filtering
                # Use func.date() for SQLite compatibility (works with datetime columns)
                date_expr = func.date(RequestLog.timestamp)
                query = db.query(
                    date_expr.label("date"),
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                    func.sum(RequestLog.cost).label("cost"),
                ).filter(RequestLog.timestamp >= filters["start_date"])

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["tags"]:
                    # Support multi-tag filtering
                    tag_conditions = []
                    for tag in filters["tags"]:
                        tag_conditions.append(RequestLog.tag == tag)
                        tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                    query = query.filter(or_(*tag_conditions))

                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                results = query.group_by(date_expr).order_by(date_expr).all()

                return jsonify(
                    [
                        {
                            # func.date() returns string in SQLite, date object in other DBs
                            "date": r.date
                            if isinstance(r.date, str)
                            else (r.date.strftime("%Y-%m-%d") if r.date else None),
                            "requests": r.requests or 0,
                            "tokens": (r.input_tokens or 0) + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )
            else:
                # Use pre-aggregated DailyStats for performance
                query = db.query(
                    DailyStats.date,
                    func.sum(DailyStats.request_count).label("requests"),
                    func.sum(DailyStats.input_tokens).label("input_tokens"),
                    func.sum(DailyStats.output_tokens).label("output_tokens"),
                    func.sum(DailyStats.estimated_cost).label("cost"),
                ).filter(
                    DailyStats.date >= filters["start_date"],
                    DailyStats.tag.is_(None),
                    DailyStats.provider_id.is_(None),
                    DailyStats.model_id.is_(None),
                )

                if filters["end_date"]:
                    query = query.filter(DailyStats.date <= filters["end_date"])

                results = (
                    query.group_by(DailyStats.date).order_by(DailyStats.date).all()
                )

                return jsonify(
                    [
                        {
                            "date": r.date.strftime("%Y-%m-%d") if r.date else None,
                            "requests": r.requests or 0,
                            "tokens": (r.input_tokens or 0) + (r.output_tokens or 0),
                            "cost": round(r.cost or 0, 4),
                        }
                        for r in results
                    ]
                )

    @admin.route("/api/usage/recent", methods=["GET"])
    @require_auth_api
    def get_recent_requests():
        """Get recent request logs."""
        limit = min(int(request.args.get("limit", 50)), 500)
        offset = int(request.args.get("offset", 0))

        with get_db_context() as db:
            logs = (
                db.query(RequestLog)
                .order_by(RequestLog.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            return jsonify([log.to_dict() for log in logs])

    @admin.route("/api/usage/export", methods=["GET"])
    @require_auth_api
    def export_request_logs():
        """Export request logs as CSV."""
        import csv
        import io
        from datetime import datetime, timedelta

        # Parse date filters
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")

        # Default to last 30 days if no dates specified
        if start_date_str:
            # Handle YYYY-MM-DD format - start of day
            start_date = datetime.strptime(start_date_str[:10], "%Y-%m-%d")
        else:
            start_date = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=30)

        if end_date_str:
            # Handle YYYY-MM-DD format - end of day (23:59:59)
            end_date = datetime.strptime(end_date_str[:10], "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, microsecond=999999
            )
        else:
            end_date = datetime.utcnow()

        # Optional filters
        tags = request.args.getlist("tag")
        providers = request.args.getlist("provider")
        models = request.args.getlist("model")

        with get_db_context() as db:
            query = db.query(RequestLog).filter(
                RequestLog.timestamp >= start_date,
                RequestLog.timestamp <= end_date,
            )

            # Apply optional filters
            if tags:
                from sqlalchemy import or_

                tag_conditions = []
                for tag in tags:
                    tag_conditions.append(RequestLog.tag == tag)
                    tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                    tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                    tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                query = query.filter(or_(*tag_conditions))

            if providers:
                query = query.filter(RequestLog.provider_id.in_(providers))

            if models:
                query = query.filter(RequestLog.model_id.in_(models))

            logs = query.order_by(RequestLog.timestamp.desc()).all()

            # Create CSV in memory
            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            writer.writerow(
                [
                    "timestamp",
                    "client_ip",
                    "hostname",
                    "tag",
                    "alias",
                    "router_name",
                    "is_designator",
                    "provider_id",
                    "model_id",
                    "endpoint",
                    "input_tokens",
                    "output_tokens",
                    "reasoning_tokens",
                    "cached_input_tokens",
                    "cache_creation_tokens",
                    "cache_read_tokens",
                    "cost",
                    "response_time_ms",
                    "status_code",
                    "is_streaming",
                    "error_message",
                ]
            )

            # Write data rows
            for log in logs:
                writer.writerow(
                    [
                        log.timestamp.isoformat() if log.timestamp else "",
                        log.client_ip or "",
                        log.hostname or "",
                        log.tag or "",
                        log.alias or "",
                        log.router_name or "",
                        "true" if log.is_designator else "false",
                        log.provider_id or "",
                        log.model_id or "",
                        log.endpoint or "",
                        log.input_tokens or 0,
                        log.output_tokens or 0,
                        log.reasoning_tokens or "",
                        log.cached_input_tokens or "",
                        log.cache_creation_tokens or "",
                        log.cache_read_tokens or "",
                        f"{log.cost:.6f}" if log.cost else "",
                        log.response_time_ms or 0,
                        log.status_code or "",
                        "true" if log.is_streaming else "false",
                        log.error_message or "",
                    ]
                )

            # Generate filename with date range
            filename = f"request_logs_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"

            # Return as downloadable CSV
            response = make_response(output.getvalue())
            response.headers["Content-Type"] = "text/csv"
            response.headers["Content-Disposition"] = f"attachment; filename={filename}"
            return response

    @admin.route("/api/usage/requests", methods=["GET"])
    @require_auth_api
    def get_paginated_requests():
        """Get paginated request logs with filtering and sorting."""
        from datetime import datetime, timedelta

        # Parse date filters
        start_date_str = request.args.get("start_date")
        end_date_str = request.args.get("end_date")

        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
        else:
            start_date = datetime.utcnow() - timedelta(days=30)

        if end_date_str:
            # Add a day to include the full end date
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
            end_date = end_date.replace(hour=23, minute=59, second=59)
        else:
            end_date = datetime.utcnow()

        # Pagination
        limit = min(int(request.args.get("limit", 100)), 500)
        offset = int(request.args.get("offset", 0))

        # Sorting
        sort_column = request.args.get("sort", "timestamp")
        sort_order = request.args.get("order", "desc")

        # Filters
        tags = request.args.getlist("tag")
        providers = request.args.getlist("provider")
        models = request.args.getlist("model")
        clients = request.args.getlist("client")
        aliases = request.args.getlist("alias")
        status_filter = request.args.get("status")

        with get_db_context() as db:
            from sqlalchemy import or_

            # Build base query
            query = db.query(RequestLog).filter(
                RequestLog.timestamp >= start_date,
                RequestLog.timestamp <= end_date,
            )

            # Apply filters
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append(RequestLog.tag == tag)
                    tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                    tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                    tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                query = query.filter(or_(*tag_conditions))

            if providers:
                query = query.filter(RequestLog.provider_id.in_(providers))

            if models:
                query = query.filter(RequestLog.model_id.in_(models))

            if clients:
                client_conditions = []
                for client in clients:
                    client_conditions.append(RequestLog.hostname == client)
                    client_conditions.append(RequestLog.client_ip == client)
                query = query.filter(or_(*client_conditions))

            if aliases:
                query = query.filter(RequestLog.alias.in_(aliases))

            if status_filter == "success":
                query = query.filter(RequestLog.status_code < 400)
            elif status_filter == "error":
                query = query.filter(RequestLog.status_code >= 400)

            # Get total count before pagination
            total_count = query.count()

            # Apply sorting
            sort_col_map = {
                "timestamp": RequestLog.timestamp,
                "input_tokens": RequestLog.input_tokens,
                "output_tokens": RequestLog.output_tokens,
                "cost": RequestLog.cost,
                "response_time_ms": RequestLog.response_time_ms,
            }
            sort_col = sort_col_map.get(sort_column, RequestLog.timestamp)
            if sort_order == "asc":
                query = query.order_by(sort_col.asc())
            else:
                query = query.order_by(sort_col.desc())

            # Apply pagination
            logs = query.offset(offset).limit(limit).all()

            return jsonify(
                {
                    "requests": [log.to_dict() for log in logs],
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                }
            )

    @admin.route("/api/usage/settings", methods=["GET"])
    @require_auth_api
    def get_usage_settings():
        """Get usage tracking settings."""
        with get_db_context() as db:
            tracking_enabled = (
                db.query(Setting)
                .filter(Setting.key == Setting.KEY_TRACKING_ENABLED)
                .first()
            )
            dns_resolution = (
                db.query(Setting)
                .filter(Setting.key == Setting.KEY_DNS_RESOLUTION_ENABLED)
                .first()
            )
            retention_days = (
                db.query(Setting)
                .filter(Setting.key == Setting.KEY_RETENTION_DAYS)
                .first()
            )

            return jsonify(
                {
                    "tracking_enabled": (
                        tracking_enabled.value.lower() == "true"
                        if tracking_enabled
                        else True
                    ),
                    "dns_resolution_enabled": (
                        dns_resolution.value.lower() == "true"
                        if dns_resolution
                        else True
                    ),
                    "retention_days": (
                        int(retention_days.value) if retention_days else 90
                    ),
                }
            )

    @admin.route("/api/usage/settings", methods=["PUT"])
    @require_auth_api
    def update_usage_settings():
        """Update usage tracking settings."""
        from datetime import datetime

        data = request.get_json() or {}

        with get_db_context() as db:
            settings_map = {
                "tracking_enabled": (
                    Setting.KEY_TRACKING_ENABLED,
                    lambda v: "true" if v else "false",
                ),
                "dns_resolution_enabled": (
                    Setting.KEY_DNS_RESOLUTION_ENABLED,
                    lambda v: "true" if v else "false",
                ),
                "retention_days": (Setting.KEY_RETENTION_DAYS, str),
            }

            for field, (key, transform) in settings_map.items():
                if field in data:
                    setting = db.query(Setting).filter(Setting.key == key).first()
                    value = transform(data[field])
                    if setting:
                        setting.value = value
                        setting.updated_at = datetime.utcnow()
                    else:
                        db.add(Setting(key=key, value=value))

            return jsonify({"success": True})

    # =========================================================================
    # Model Sync API (sync models/pricing from LiteLLM)
    # =========================================================================

    @admin.route("/api/models/sync", methods=["POST"])
    @require_auth_api
    def sync_models():
        """
        Sync models and pricing from LiteLLM and return a diff report.

        Request body (optional):
        {
            "provider": "openai",      // Optional: sync specific provider only
            "update_existing": false,  // Optional: update prices for existing models
            "add_new": false           // Optional: add new models as custom models
        }

        Compares LiteLLM data against current database models.
        When apply options are set, updates database models directly.
        """
        import sys
        from pathlib import Path

        # Add scripts directory to path for import
        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from sync_pricing import (
                PROVIDER_MAPPING,
                apply_pricing_overrides,
                compare_pricing,
                fetch_litellm_pricing,
                load_effective_pricing,
                parse_litellm_model,
            )
        except ImportError as e:
            return jsonify({"error": f"Failed to import sync module: {e}"}), 500

        data = request.get_json() or {}
        provider_filter = data.get("provider")
        update_existing = data.get("update_existing", False)
        add_new = data.get("add_new", False)
        disable_deprecated = data.get("disable_deprecated", False)
        # Optional: list of specific models to apply actions to
        selected_models = data.get(
            "selected_models", None
        )  # ["provider/model_id", ...]

        try:
            # Fetch remote pricing data
            litellm_data = fetch_litellm_pricing()

            # Parse all models
            remote_models = []
            for key, model_data in litellm_data.items():
                if key == "sample_spec":
                    continue
                model = parse_litellm_model(key, model_data)
                if model:
                    remote_models.append(model)

            # Determine which providers to sync
            if provider_filter:
                providers = [provider_filter]
            else:
                providers = list(set(PROVIDER_MAPPING.values()))

            # Compare with current database models
            all_diffs = []
            for provider in providers:
                # Use load_effective_pricing to include DB overrides
                local_config = load_effective_pricing(provider)
                provider_models = [m for m in remote_models if m.provider == provider]
                if provider_models:
                    diffs = compare_pricing(local_config, provider_models, provider)
                    all_diffs.extend(diffs)

            # Categorize diffs
            price_updates = [
                d
                for d in all_diffs
                if d.field not in ("(new model)", "(deprecated)")
                and d.change_type not in ("info", "removed")
            ]
            new_models = [d for d in all_diffs if d.field == "(new model)"]
            deprecated_models = [d for d in all_diffs if d.field == "(deprecated)"]

            # Count unique models with price updates
            unique_price_update_models = set(
                f"{d.provider}/{d.model_id}" for d in price_updates
            )

            # Format results
            results = {
                "total_remote_models": len(remote_models),
                "providers_checked": providers,
                "differences_found": len(all_diffs),
                "price_update_count": len(unique_price_update_models),
                "new_model_count": len(new_models),
                "deprecated_model_count": len(deprecated_models),
                "diffs": [
                    {
                        "model_id": d.model_id,
                        "provider": d.provider,
                        "field": d.field,
                        "local_value": d.local_value,
                        "remote_value": d.remote_value,
                        "change_type": d.change_type,
                    }
                    for d in all_diffs
                ],
            }

            # Filter diffs by selected models if provided
            diffs_to_apply = all_diffs
            if selected_models:
                selected_set = set(selected_models)
                diffs_to_apply = [
                    d for d in all_diffs if f"{d.provider}/{d.model_id}" in selected_set
                ]

            # Apply changes if requested
            if (update_existing or add_new) and diffs_to_apply:
                apply_result = apply_pricing_overrides(
                    diffs_to_apply,
                    dry_run=False,
                    update_existing=update_existing,
                    add_new=add_new,
                )
                results["overrides_created"] = apply_result["created"]
                results["overrides_updated"] = apply_result["updated"]
                results["new_models_added"] = apply_result["new_models_added"]

            # Handle disabling deprecated models
            if disable_deprecated and diffs_to_apply:
                deprecated_to_disable = [
                    d for d in diffs_to_apply if d.field == "(deprecated)"
                ]
                if deprecated_to_disable:
                    disabled_count = 0
                    with get_db_context() as db:
                        for diff in deprecated_to_disable:
                            model = (
                                db.query(Model)
                                .filter(
                                    Model.provider_id == diff.provider,
                                    Model.id == diff.model_id,
                                )
                                .first()
                            )
                            if model and model.enabled:
                                model.enabled = False
                                disabled_count += 1
                        db.commit()
                    results["models_disabled"] = disabled_count

            # Re-apply YAML overrides after sync to ensure quirks are preserved
            try:
                from config.override_loader import apply_yaml_overrides_to_db

                override_stats = apply_yaml_overrides_to_db()
                results["yaml_overrides_applied"] = (
                    override_stats["created"] + override_stats["updated"]
                )

                # Clear model cache so overrides take effect immediately
                clear_config_cache()
            except Exception as e:
                logger.warning(f"Could not apply YAML overrides after sync: {e}")

            return jsonify(results)

        except Exception as e:
            logger.exception("Failed to sync models")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/models/sync/providers", methods=["GET"])
    @require_auth_api
    def get_sync_providers():
        """Get list of providers that can be synced."""
        import sys
        from pathlib import Path

        scripts_dir = Path(__file__).parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        try:
            from sync_pricing import PROVIDER_MAPPING

            return jsonify({"providers": list(set(PROVIDER_MAPPING.values()))})
        except ImportError:
            return jsonify({"providers": []})

    # -------------------------------------------------------------------------
    # Debug Log Streaming API
    # -------------------------------------------------------------------------

    @admin.route("/api/debug/logs", methods=["GET"])
    @require_auth_api
    def get_debug_logs():
        """Get recent debug logs."""
        from .debug_logs import debug_log_buffer

        count = min(int(request.args.get("count", 100)), 500)
        entries = debug_log_buffer.get_recent(count)
        return jsonify(
            {
                "enabled": debug_log_buffer.is_enabled(),
                "logs": [e.to_dict() for e in entries],
            }
        )

    @admin.route("/api/debug/logs/stream", methods=["GET"])
    @require_auth_api
    def stream_debug_logs():
        """Stream debug logs via Server-Sent Events (SSE)."""
        import json

        from flask import Response, stream_with_context

        from .debug_logs import debug_log_buffer

        def generate():
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'enabled': debug_log_buffer.is_enabled()})}\n\n"

            # Stream new log entries
            for entry in debug_log_buffer.stream():
                data = {
                    "type": "log",
                    "entry": entry.to_dict(),
                }
                yield f"data: {json.dumps(data)}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @admin.route("/api/debug/logs/enable", methods=["POST"])
    @require_auth_api
    def enable_debug_logs():
        """Enable debug log capture."""
        from .debug_logs import debug_log_buffer

        debug_log_buffer.enable()
        return jsonify({"success": True, "enabled": True})

    @admin.route("/api/debug/logs/disable", methods=["POST"])
    @require_auth_api
    def disable_debug_logs():
        """Disable debug log capture."""
        from .debug_logs import debug_log_buffer

        debug_log_buffer.disable()
        return jsonify({"success": True, "enabled": False})

    @admin.route("/api/debug/logs/clear", methods=["POST"])
    @require_auth_api
    def clear_debug_logs():
        """Clear all captured debug logs."""
        from .debug_logs import debug_log_buffer

        debug_log_buffer.clear()
        return jsonify({"success": True})

    # -------------------------------------------------------------------------
    # Aliases API (v3.1)
    # -------------------------------------------------------------------------

    @admin.route("/api/aliases", methods=["GET"])
    @require_auth_api
    def list_aliases():
        """List all aliases."""
        from db import get_all_aliases

        aliases = get_all_aliases()
        return jsonify([a.to_dict() for a in aliases])

    @admin.route("/api/aliases/<int:alias_id>", methods=["GET"])
    @require_auth_api
    def get_alias(alias_id: int):
        """Get a single alias by ID."""
        from db import get_alias_by_id

        alias = get_alias_by_id(alias_id)
        if not alias:
            return jsonify({"error": "Alias not found"}), 404
        return jsonify(alias.to_dict())

    @admin.route("/api/aliases", methods=["POST"])
    @require_auth_api
    def create_alias_endpoint():
        """Create a new alias."""
        from db import alias_name_available, create_alias

        data = request.get_json() or {}

        # Validate required fields
        if not data.get("name"):
            return jsonify({"error": "Name is required"}), 400
        if not data.get("target_model"):
            return jsonify({"error": "Target model is required"}), 400

        name = data["name"].lower().strip()

        # Check if name is available
        if not alias_name_available(name):
            return jsonify({"error": f"Alias name '{name}' is already in use"}), 409

        # Check if name conflicts with an existing model
        try:
            from providers import registry

            resolved = registry.resolve_model(name)
            # If we get here without using an alias or default fallback,
            # the name matches a real model
            if not resolved.has_alias and not resolved.is_default_fallback:
                return jsonify(
                    {"error": f"Name '{name}' conflicts with an existing model"}
                ), 409
        except ValueError:
            # Model not found - that's fine, the name is available
            pass

        try:
            alias = create_alias(
                name=name,
                target_model=data["target_model"],
                tags=data.get("tags", []),
                description=data.get("description"),
                enabled=data.get("enabled", True),
            )
            return jsonify(alias.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/aliases/<int:alias_id>", methods=["PUT"])
    @require_auth_api
    def update_alias_endpoint(alias_id: int):
        """Update an existing alias."""
        from db import alias_name_available, get_alias_by_id, update_alias

        data = request.get_json() or {}

        # Check alias exists
        existing = get_alias_by_id(alias_id)
        if not existing:
            return jsonify({"error": "Alias not found"}), 404

        # If name is being changed, validate it
        new_name = data.get("name")
        if new_name and new_name.lower().strip() != existing.name:
            new_name = new_name.lower().strip()
            if not alias_name_available(new_name, exclude_id=alias_id):
                return jsonify(
                    {"error": f"Alias name '{new_name}' is already in use"}
                ), 409

            # Check if new name conflicts with an existing model
            try:
                from providers import registry

                resolved = registry.resolve_model(new_name)
                if not resolved.has_alias and not resolved.is_default_fallback:
                    return jsonify(
                        {"error": f"Name '{new_name}' conflicts with an existing model"}
                    ), 409
            except ValueError:
                pass

        try:
            alias = update_alias(
                alias_id=alias_id,
                name=data.get("name"),
                target_model=data.get("target_model"),
                tags=data.get("tags"),
                description=data.get("description"),
                enabled=data.get("enabled"),
            )
            if not alias:
                return jsonify({"error": "Alias not found"}), 404
            return jsonify(alias.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/aliases/<int:alias_id>", methods=["DELETE"])
    @require_auth_api
    def delete_alias_endpoint(alias_id: int):
        """Delete an alias."""
        from db import delete_alias

        if delete_alias(alias_id):
            return jsonify({"success": True})
        return jsonify({"error": "Alias not found"}), 404

    @admin.route("/api/aliases/validate/<name>", methods=["GET"])
    @require_auth_api
    def validate_alias_name(name: str):
        """Check if an alias name is available."""
        from db import alias_name_available

        name = name.lower().strip()
        exclude_id = request.args.get("exclude_id", type=int)

        # Check if name is taken by another alias
        if not alias_name_available(name, exclude_id=exclude_id):
            return jsonify(
                {"available": False, "reason": "Name is already used by another alias"}
            )

        # Check if name conflicts with an existing model
        try:
            from providers import registry

            resolved = registry.resolve_model(name)
            if not resolved.has_alias and not resolved.is_default_fallback:
                return jsonify(
                    {
                        "available": False,
                        "reason": "Name conflicts with an existing model",
                    }
                )
        except ValueError:
            pass

        return jsonify({"available": True})

    @admin.route("/api/aliases/models", methods=["GET"])
    @require_auth_api
    def list_available_models_for_alias():
        """List all available models that can be used as alias targets."""
        models = []
        for provider in registry.get_available_providers():
            for model_id, info in provider.get_models().items():
                models.append(
                    {
                        "id": f"{provider.name}/{model_id}",
                        "provider": provider.name,
                        "model_id": model_id,
                        "family": info.family,
                        "description": info.description,
                    }
                )
        return jsonify(models)

    # -------------------------------------------------------------------------
    # Smart Router API (v3.2)
    # -------------------------------------------------------------------------

    @admin.route("/routers")
    @require_auth
    def routers_page():
        """Smart Routers management page."""
        return render_template("routers.html")

    @admin.route("/api/routers", methods=["GET"])
    @require_auth_api
    def list_routers():
        """List all smart routers."""
        from db import get_all_smart_routers

        routers = get_all_smart_routers()
        return jsonify([r.to_dict() for r in routers])

    @admin.route("/api/routers/<int:router_id>", methods=["GET"])
    @require_auth_api
    def get_router(router_id: int):
        """Get a single smart router by ID."""
        from db import get_smart_router_by_id

        router = get_smart_router_by_id(router_id)
        if not router:
            return jsonify({"error": "Router not found"}), 404
        return jsonify(router.to_dict())

    @admin.route("/api/routers", methods=["POST"])
    @require_auth_api
    def create_router_endpoint():
        """Create a new smart router."""
        from db import create_smart_router, router_name_available

        data = request.get_json() or {}

        # Validate required fields
        if not data.get("name"):
            return jsonify({"error": "Name is required"}), 400
        if not data.get("designator_model"):
            return jsonify({"error": "Designator model is required"}), 400
        if not data.get("purpose"):
            return jsonify({"error": "Purpose is required"}), 400
        if not data.get("candidates") or len(data["candidates"]) < 2:
            return jsonify({"error": "At least 2 candidates are required"}), 400
        if not data.get("fallback_model"):
            return jsonify({"error": "Fallback model is required"}), 400

        name = data["name"].lower().strip()

        # Check if name is available
        if not router_name_available(name):
            return jsonify({"error": f"Router name '{name}' is already in use"}), 409

        # Check if name conflicts with an existing model or alias
        try:
            from providers import registry

            resolved = registry.resolve_model(name)
            # If we get here without using default fallback, the name matches something
            if not resolved.is_default_fallback:
                return jsonify(
                    {
                        "error": f"Name '{name}' conflicts with an existing model or alias"
                    }
                ), 409
        except ValueError:
            # Model not found - that's fine, the name is available
            pass

        try:
            router = create_smart_router(
                name=name,
                designator_model=data["designator_model"],
                purpose=data["purpose"],
                candidates=data["candidates"],
                fallback_model=data["fallback_model"],
                strategy=data.get("strategy", "per_request"),
                session_ttl=data.get("session_ttl", 3600),
                tags=data.get("tags", []),
                description=data.get("description"),
                enabled=data.get("enabled", True),
            )
            return jsonify(router.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/routers/<int:router_id>", methods=["PUT"])
    @require_auth_api
    def update_router_endpoint(router_id: int):
        """Update an existing smart router."""
        from db import (
            get_smart_router_by_id,
            router_name_available,
            update_smart_router,
        )

        data = request.get_json() or {}

        # Check router exists
        existing = get_smart_router_by_id(router_id)
        if not existing:
            return jsonify({"error": "Router not found"}), 404

        # If name is being changed, validate it
        new_name = data.get("name")
        if new_name and new_name.lower().strip() != existing.name:
            new_name = new_name.lower().strip()
            if not router_name_available(new_name, exclude_id=router_id):
                return jsonify(
                    {"error": f"Router name '{new_name}' is already in use"}
                ), 409

            # Check if new name conflicts with an existing model or alias
            try:
                from providers import registry

                resolved = registry.resolve_model(new_name)
                if not resolved.is_default_fallback:
                    return jsonify(
                        {
                            "error": f"Name '{new_name}' conflicts with an existing model or alias"
                        }
                    ), 409
            except ValueError:
                pass

        try:
            router = update_smart_router(
                router_id=router_id,
                name=data.get("name"),
                designator_model=data.get("designator_model"),
                purpose=data.get("purpose"),
                candidates=data.get("candidates"),
                fallback_model=data.get("fallback_model"),
                strategy=data.get("strategy"),
                session_ttl=data.get("session_ttl"),
                tags=data.get("tags"),
                description=data.get("description"),
                enabled=data.get("enabled"),
            )
            if not router:
                return jsonify({"error": "Router not found"}), 404
            return jsonify(router.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/routers/<int:router_id>", methods=["DELETE"])
    @require_auth_api
    def delete_router_endpoint(router_id: int):
        """Delete a smart router."""
        from db import delete_smart_router

        if delete_smart_router(router_id):
            return jsonify({"success": True})
        return jsonify({"error": "Router not found"}), 404

    @admin.route("/api/routers/validate/<name>", methods=["GET"])
    @require_auth_api
    def validate_router_name(name: str):
        """Check if a router name is available."""
        from db import router_name_available

        name = name.lower().strip()
        exclude_id = request.args.get("exclude_id", type=int)

        # Check if name is taken by another router
        if not router_name_available(name, exclude_id=exclude_id):
            return jsonify(
                {"available": False, "reason": "Name is already used by another router"}
            )

        # Check if name conflicts with an existing model or alias
        try:
            from providers import registry

            resolved = registry.resolve_model(name)
            if not resolved.is_default_fallback:
                return jsonify(
                    {
                        "available": False,
                        "reason": "Name conflicts with an existing model or alias",
                    }
                )
        except ValueError:
            pass

        return jsonify({"available": True})

    @admin.route("/api/routers/models", methods=["GET"])
    @require_auth_api
    def list_available_models_for_router():
        """List all available models that can be used as router candidates/designators."""
        models = []
        for provider in registry.get_available_providers():
            for model_id, info in provider.get_models().items():
                models.append(
                    {
                        "id": f"{provider.name}/{model_id}",
                        "provider": provider.name,
                        "model_id": model_id,
                        "family": info.family,
                        "description": info.description,
                        "context_length": info.context_length,
                        "capabilities": info.capabilities,
                        "input_cost": info.input_cost,
                        "output_cost": info.output_cost,
                    }
                )
        return jsonify(models)

    @admin.route("/api/usage/by-router", methods=["GET"])
    @require_auth_api
    def usage_by_router():
        """Get usage statistics grouped by smart router."""
        from datetime import datetime, timedelta

        from sqlalchemy import func

        # Parse date range
        days = request.args.get("days", 7, type=int)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        with get_db_context() as db:
            # Query daily stats grouped by router
            stats = (
                db.query(
                    DailyStats.router_name,
                    func.sum(DailyStats.request_count).label("requests"),
                    func.sum(DailyStats.input_tokens).label("input_tokens"),
                    func.sum(DailyStats.output_tokens).label("output_tokens"),
                    func.sum(DailyStats.estimated_cost).label("cost"),
                    func.sum(DailyStats.success_count).label("success"),
                    func.sum(DailyStats.error_count).label("errors"),
                )
                .filter(
                    DailyStats.date >= start_date,
                    DailyStats.router_name.isnot(None),
                    # Only get router-level totals (no other dimensions)
                    DailyStats.tag.is_(None),
                    DailyStats.provider_id.is_(None),
                    DailyStats.model_id.is_(None),
                    DailyStats.alias.is_(None),
                )
                .group_by(DailyStats.router_name)
                .all()
            )

            result = []
            for stat in stats:
                result.append(
                    {
                        "router_name": stat.router_name,
                        "requests": stat.requests or 0,
                        "input_tokens": stat.input_tokens or 0,
                        "output_tokens": stat.output_tokens or 0,
                        "cost": float(stat.cost or 0),
                        "success": stat.success or 0,
                        "errors": stat.errors or 0,
                    }
                )

            return jsonify(result)

    return admin
