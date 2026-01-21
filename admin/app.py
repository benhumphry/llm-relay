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
    session,
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
    require_widget_api_key,
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
    # Web Search & Scraping Settings
    # -------------------------------------------------------------------------

    @admin.route("/api/settings/web", methods=["GET"])
    @require_auth_api
    def get_web_settings():
        """Get web search and scraping settings."""
        import os

        # Check GPU availability for local models
        gpu_available = False
        gpu_name = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_available = True
                gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        keys = [
            Setting.KEY_WEB_SEARCH_PROVIDER,
            Setting.KEY_WEB_SEARCH_URL,
            Setting.KEY_WEB_SCRAPER_PROVIDER,
            Setting.KEY_WEB_CRAWL_PROVIDER,
            Setting.KEY_WEB_PDF_PARSER,
            Setting.KEY_RAG_PDF_PARSER,
            Setting.KEY_EMBEDDING_PROVIDER,
            Setting.KEY_EMBEDDING_MODEL,
            Setting.KEY_EMBEDDING_OLLAMA_URL,
            Setting.KEY_WEB_RERANK_PROVIDER,
            Setting.KEY_VISION_PROVIDER,
            Setting.KEY_VISION_MODEL,
            Setting.KEY_VISION_OLLAMA_URL,
            Setting.KEY_DOCSTORE_INTELLIGENCE_MODEL,
        ]

        with get_db_context() as db:
            settings = db.query(Setting).filter(Setting.key.in_(keys)).all()
            settings_dict = {s.key: s.value for s in settings}

        # Check if JINA_API_KEY env var is set
        jina_api_configured = bool(os.environ.get("JINA_API_KEY", ""))

        return jsonify(
            {
                "search_provider": settings_dict.get(Setting.KEY_WEB_SEARCH_PROVIDER)
                or "",
                "search_url": settings_dict.get(Setting.KEY_WEB_SEARCH_URL) or "",
                "scraper_provider": settings_dict.get(Setting.KEY_WEB_SCRAPER_PROVIDER)
                or "builtin",
                "crawl_provider": settings_dict.get(Setting.KEY_WEB_CRAWL_PROVIDER)
                or "builtin",
                "pdf_parser": settings_dict.get(Setting.KEY_WEB_PDF_PARSER) or "pypdf",
                "rag_pdf_parser": settings_dict.get(Setting.KEY_RAG_PDF_PARSER)
                or "docling",
                "embedding_provider": settings_dict.get(Setting.KEY_EMBEDDING_PROVIDER)
                or "local",
                "embedding_model": settings_dict.get(Setting.KEY_EMBEDDING_MODEL) or "",
                "embedding_ollama_url": settings_dict.get(
                    Setting.KEY_EMBEDDING_OLLAMA_URL
                )
                or "",
                "rerank_provider": settings_dict.get(Setting.KEY_WEB_RERANK_PROVIDER)
                or "local",
                "jina_api_configured": jina_api_configured,
                "vision_provider": settings_dict.get(Setting.KEY_VISION_PROVIDER)
                or "local",
                "vision_model": settings_dict.get(Setting.KEY_VISION_MODEL) or "",
                "vision_ollama_url": settings_dict.get(Setting.KEY_VISION_OLLAMA_URL)
                or "",
                "docstore_intelligence_model": settings_dict.get(
                    Setting.KEY_DOCSTORE_INTELLIGENCE_MODEL
                )
                or "",
                "gpu_available": gpu_available,
                "gpu_name": gpu_name,
            }
        )

    @admin.route("/api/settings/web", methods=["PUT"])
    @require_auth_api
    def save_web_settings():
        """Save web search and scraping settings."""
        data = request.get_json() or {}

        # Validate search provider
        search_provider = data.get("search_provider", "")
        if search_provider and search_provider not in ("searxng", "perplexity", "jina"):
            return jsonify({"error": "Invalid search provider"}), 400

        # Validate scraper provider
        # "jina" = free tier, "jina-api" = with API key
        scraper_provider = data.get("scraper_provider", "builtin")
        if scraper_provider not in ("builtin", "jina", "jina-api"):
            return jsonify({"error": "Invalid scraper provider"}), 400

        # Validate crawl provider (for website indexing)
        # "jina" = free tier, "jina-api" = with API key
        crawl_provider = data.get("crawl_provider", "builtin")
        if crawl_provider not in ("builtin", "jina", "jina-api"):
            return jsonify({"error": "Invalid crawl provider"}), 400

        # Validate PDF parser (for web scraping)
        pdf_parser = data.get("pdf_parser", "pypdf")
        if pdf_parser not in ("docling", "pypdf", "jina"):
            return jsonify({"error": "Invalid PDF parser"}), 400

        # Validate RAG PDF parser (for document indexing)
        rag_pdf_parser = data.get("rag_pdf_parser", "docling")
        if rag_pdf_parser not in ("docling", "pypdf"):
            return jsonify({"error": "Invalid RAG PDF parser"}), 400

        # Embedding provider can be "local", "ollama:<instance>", or any provider name
        embedding_provider = data.get("embedding_provider", "local")
        embedding_model = data.get("embedding_model", "")
        embedding_ollama_url = data.get("embedding_ollama_url", "")

        # Validate rerank provider
        rerank_provider = data.get("rerank_provider", "local")
        if rerank_provider not in ("local", "jina"):
            return jsonify({"error": "Invalid rerank provider"}), 400

        # Vision provider can be "local", "ollama:<instance>", or any provider name
        vision_provider = data.get("vision_provider", "local")
        vision_model = data.get("vision_model", "")
        vision_ollama_url = data.get("vision_ollama_url", "")

        # Save settings (jina_api_key is now via env var JINA_API_KEY)
        # Only save fields that are explicitly provided in the request
        settings_to_save = {}

        if "search_provider" in data:
            settings_to_save[Setting.KEY_WEB_SEARCH_PROVIDER] = search_provider
        if "search_url" in data:
            settings_to_save[Setting.KEY_WEB_SEARCH_URL] = data.get("search_url", "")
        if "scraper_provider" in data:
            settings_to_save[Setting.KEY_WEB_SCRAPER_PROVIDER] = scraper_provider
        if "crawl_provider" in data:
            settings_to_save[Setting.KEY_WEB_CRAWL_PROVIDER] = crawl_provider
        if "pdf_parser" in data:
            settings_to_save[Setting.KEY_WEB_PDF_PARSER] = pdf_parser
        if "rag_pdf_parser" in data:
            settings_to_save[Setting.KEY_RAG_PDF_PARSER] = rag_pdf_parser
        if "embedding_provider" in data:
            settings_to_save[Setting.KEY_EMBEDDING_PROVIDER] = embedding_provider
        if "embedding_model" in data:
            settings_to_save[Setting.KEY_EMBEDDING_MODEL] = embedding_model
        if "embedding_ollama_url" in data:
            settings_to_save[Setting.KEY_EMBEDDING_OLLAMA_URL] = embedding_ollama_url
        if "rerank_provider" in data:
            settings_to_save[Setting.KEY_WEB_RERANK_PROVIDER] = rerank_provider
        if "vision_provider" in data:
            settings_to_save[Setting.KEY_VISION_PROVIDER] = vision_provider
        if "vision_model" in data:
            settings_to_save[Setting.KEY_VISION_MODEL] = vision_model
        if "vision_ollama_url" in data:
            settings_to_save[Setting.KEY_VISION_OLLAMA_URL] = vision_ollama_url
        if "docstore_intelligence_model" in data:
            settings_to_save[Setting.KEY_DOCSTORE_INTELLIGENCE_MODEL] = data.get(
                "docstore_intelligence_model", ""
            )

        with get_db_context() as db:
            for key, value in settings_to_save.items():
                setting = db.query(Setting).filter(Setting.key == key).first()
                if setting:
                    setting.value = value
                else:
                    db.add(Setting(key=key, value=value))
            db.commit()

        return jsonify({"success": True})

    @admin.route("/api/settings/web", methods=["POST"])
    @require_auth_api
    def save_web_settings_post():
        """Save web settings (POST alias for PUT)."""
        return save_web_settings()

    @admin.route("/api/settings/embedding-models", methods=["GET"])
    @require_auth_api
    def get_embedding_models():
        """Get available embedding models from Ollama instances and providers."""
        from providers.ollama_provider import OllamaProvider

        result = {
            "ollama": {"available": False, "instances": []},
            "providers": [],
        }

        embedding_patterns = ["embed", "nomic", "mxbai", "bge", "e5", "gte"]

        # Get Ollama instances with embedding models
        try:
            ollama_providers = [
                p
                for p in registry.get_all_providers()
                if isinstance(p, OllamaProvider) and p.is_configured()
            ]

            for provider in ollama_providers:
                try:
                    raw_models = provider.get_raw_models()
                    embedding_models = [
                        m.get("name", "")
                        for m in raw_models
                        if any(
                            kw in m.get("name", "").lower() for kw in embedding_patterns
                        )
                    ]
                    if embedding_models:
                        result["ollama"]["instances"].append(
                            {
                                "name": provider.name,
                                "url": provider.base_url,
                                "embedding_models": embedding_models,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get models from {provider.name}: {e}")

            if result["ollama"]["instances"]:
                result["ollama"]["available"] = True
        except Exception as e:
            logger.warning(f"Failed to enumerate Ollama providers: {e}")

        # Get providers with embedding support (OpenAI)
        try:
            for provider in registry.get_available_providers():
                if provider.name.lower() == "openai":
                    result["providers"].append(
                        {
                            "name": provider.name,
                            "models": [
                                "text-embedding-3-small",
                                "text-embedding-3-large",
                                "text-embedding-ada-002",
                            ],
                        }
                    )
        except Exception as e:
            logger.warning(f"Failed to enumerate embedding providers: {e}")

        return jsonify(result)

    @admin.route("/api/settings/vision-models", methods=["GET"])
    @require_auth_api
    def get_vision_models():
        """Get available vision models from Ollama instances and providers."""
        from providers.ollama_provider import OllamaProvider

        PROVIDER_VISION_MODELS = {
            "openai": [
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            ],
            "anthropic": [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4"},
                {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet"},
                {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku"},
            ],
            "openrouter": [
                {"id": "openai/gpt-4o", "name": "OpenAI GPT-4o"},
                {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
                {"id": "google/gemini-pro-1.5", "name": "Gemini Pro 1.5"},
            ],
            "google": [
                {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash"},
                {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro"},
                {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash"},
            ],
            "together": [
                {
                    "id": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
                    "name": "Llama 3.2 11B Vision",
                },
                {
                    "id": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "name": "Llama 3.2 90B Vision",
                },
            ],
            "fireworks": [
                {
                    "id": "accounts/fireworks/models/llama-v3p2-11b-vision-instruct",
                    "name": "Llama 3.2 11B Vision",
                },
            ],
        }

        result = {
            "local": {
                "available": True,
                "description": "Bundled Docling models (CPU/GPU)",
                "models": [
                    {"id": "default", "name": "Docling Default (local processing)"}
                ],
            },
            "ollama": {"available": False, "instances": []},
            "providers": [],
        }

        vision_patterns = [
            "vision",
            "llava",
            "granite3.2-vision",
            "granite-vision",
            "granite-docling",
            "moondream",
            "bakllava",
            "llama3.2-vision",
            "minicpm-v",
        ]

        # Get Ollama instances with vision models
        try:
            ollama_providers = [
                p
                for p in registry.get_all_providers()
                if isinstance(p, OllamaProvider) and p.is_configured()
            ]

            for provider in ollama_providers:
                try:
                    raw_models = provider.get_raw_models()
                    vision_models = [
                        {
                            "id": m.get("name", ""),
                            "name": m.get("name", ""),
                            "size": m.get("size"),
                        }
                        for m in raw_models
                        if any(
                            pattern in m.get("name", "").lower()
                            for pattern in vision_patterns
                        )
                    ]
                    if vision_models:
                        result["ollama"]["instances"].append(
                            {
                                "name": provider.name,
                                "url": provider.base_url,
                                "vision_models": vision_models,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Failed to get models from {provider.name}: {e}")

            if result["ollama"]["instances"]:
                result["ollama"]["available"] = True
        except Exception as e:
            logger.warning(f"Failed to enumerate Ollama providers: {e}")

        # Include ALL configured providers
        try:
            for provider in registry.get_available_providers():
                if provider.name.startswith("ollama"):
                    continue
                provider_name = provider.name.lower()
                suggested_models = PROVIDER_VISION_MODELS.get(provider_name, [])
                result["providers"].append(
                    {
                        "name": provider.name,
                        "models": suggested_models,
                        "supports_custom": True,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to enumerate vision providers: {e}")

        return jsonify(result)

    # -------------------------------------------------------------------------
    # ChromaDB Status
    # -------------------------------------------------------------------------

    @admin.route("/api/chroma/status", methods=["GET"])
    @require_auth_api
    def chroma_status():
        """Check ChromaDB connection status and list collections."""
        try:
            from context import (
                get_chroma_url,
                get_collection_prefix,
                is_chroma_available,
                is_chroma_configured,
                list_collections,
            )
        except ImportError:
            return jsonify(
                {
                    "configured": False,
                    "available": False,
                    "error": "context module not available",
                }
            )

        # Check if CHROMA_URL is set
        configured = is_chroma_configured()
        url = get_chroma_url()
        prefix = get_collection_prefix()

        result = {
            "configured": configured,
            "url": url,
            "prefix": prefix,
            "available": False,
        }

        if not configured:
            result["message"] = (
                "ChromaDB not configured. Set CHROMA_URL environment variable "
                "to enable Smart Cache, Smart Augmentor, and Model Intelligence features."
            )
            return jsonify(result)

        if is_chroma_available():
            result["available"] = True
            try:
                collections = list_collections()
                result["collections"] = collections
                result["collection_count"] = len(collections)

                # Get document counts per collection
                from context import CollectionWrapper

                collection_stats = []
                for coll_name in collections:
                    try:
                        wrapper = CollectionWrapper(coll_name)
                        collection_stats.append(
                            {
                                "name": coll_name,
                                "count": wrapper.count(),
                            }
                        )
                    except Exception:
                        collection_stats.append(
                            {
                                "name": coll_name,
                                "count": -1,
                                "error": "Could not get count",
                            }
                        )
                result["collection_stats"] = collection_stats
            except Exception as e:
                result["error"] = str(e)
        else:
            result["error"] = f"ChromaDB not reachable at {url}"

        return jsonify(result)

    @admin.route("/api/feature-status", methods=["GET"])
    @require_auth_api
    def feature_status():
        """
        Get status of optional features and their dependencies.

        Used by dashboard and sidebar to show available features.
        """
        import os

        # Check ChromaDB
        chroma_configured = False
        chroma_available = False
        chroma_url = os.environ.get("CHROMA_URL", "")

        try:
            from context import is_chroma_available, is_chroma_configured

            chroma_configured = is_chroma_configured()
            if chroma_configured:
                chroma_available = is_chroma_available()
        except ImportError:
            pass

        # Check search providers
        search_providers = []
        has_search_provider = False

        try:
            from augmentation.search import list_search_providers

            search_providers = list_search_providers()
            has_search_provider = any(p["configured"] for p in search_providers)
        except ImportError:
            pass

        # Add URL info to search providers
        searxng_url = os.environ.get("SEARXNG_URL", "")
        perplexity_configured = bool(os.environ.get("PERPLEXITY_API_KEY", ""))

        for p in search_providers:
            if p["name"] == "searxng":
                p["url"] = searxng_url
            elif p["name"] == "perplexity":
                p["url"] = "api.perplexity.ai"

        # Build feature status
        features = {
            "smart_enrichers": {
                "available": chroma_available,
                "enabled": chroma_configured,
                "reason": None
                if chroma_available
                else (
                    "CHROMA_URL not set"
                    if not chroma_configured
                    else "ChromaDB not reachable"
                ),
            },
            "model_intelligence": {
                "available": chroma_available and has_search_provider,
                "enabled": chroma_configured and has_search_provider,
                "reason": None
                if (chroma_available and has_search_provider)
                else (
                    "Requires ChromaDB and a search provider"
                    if not chroma_configured
                    else "No search provider configured"
                    if not has_search_provider
                    else "ChromaDB not reachable"
                ),
            },
        }

        # Get database info
        database_url = os.environ.get("DATABASE_URL", "")
        if database_url and (
            "postgresql" in database_url or "postgres" in database_url
        ):
            db_type = "PostgreSQL"
            # Extract host from URL (hide credentials)
            try:
                from urllib.parse import urlparse

                parsed = urlparse(database_url)
                db_display = (
                    f"{parsed.hostname}:{parsed.port or 5432}/{parsed.path.lstrip('/')}"
                )
            except Exception:
                db_display = "configured"
        else:
            db_type = "SQLite"
            db_display = "built-in"

        # Build dependencies status
        dependencies = {
            "database": {
                "type": db_type,
                "display": db_display,
            },
            "chromadb": {
                "configured": chroma_configured,
                "available": chroma_available,
                "url": chroma_url or None,
                "reason": None
                if chroma_available
                else (
                    "CHROMA_URL not set" if not chroma_configured else "Not reachable"
                ),
            },
            "search_providers": search_providers,
            "has_search_provider": has_search_provider,
            "document_sources": {
                "paperless": {
                    "configured": bool(
                        os.environ.get("PAPERLESS_URL")
                        and os.environ.get("PAPERLESS_TOKEN")
                    ),
                },
                "nextcloud": {
                    "configured": bool(
                        os.environ.get("NEXTCLOUD_URL")
                        and os.environ.get("NEXTCLOUD_USER")
                        and os.environ.get("NEXTCLOUD_PASSWORD")
                    ),
                },
                "notion": {
                    "configured": bool(
                        os.environ.get("NOTION_TOKEN")
                        or os.environ.get("NOTION_API_KEY")
                    ),
                },
                "github": {
                    "configured": bool(
                        os.environ.get("GITHUB_TOKEN")
                        or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
                    ),
                },
                "slack": {
                    "configured": bool(os.environ.get("SLACK_BOT_TOKEN")),
                },
                "todoist": {
                    "configured": bool(
                        os.environ.get("TODOIST_API_TOKEN")
                        or os.environ.get("TODOIST_API_KEY")
                    ),
                },
            },
        }

        return jsonify(
            {
                "features": features,
                "dependencies": dependencies,
            }
        )

    # -------------------------------------------------------------------------
    # Config Import/Export
    # -------------------------------------------------------------------------

    @admin.route("/api/config/export", methods=["GET"])
    @require_auth_api
    def export_config():
        """Export database configuration as JSON for backup/migration."""
        from db.models import (
            CustomModel,
            CustomProvider,
            DailyStats,
            ModelOverride,
            OllamaInstance,
            Provider,
            RequestLog,
            Setting,
            SmartAlias,
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
                "smart_aliases": [
                    {
                        "name": a.name,
                        "target_model": a.target_model,
                        "use_routing": a.use_routing,
                        "use_rag": a.use_rag,
                        "use_web": a.use_web,
                        "use_cache": a.use_cache,
                        "designator_model": a.designator_model,
                        "purpose": a.purpose,
                        "candidates": a.candidates,
                        "fallback_model": a.fallback_model,
                        "routing_strategy": a.routing_strategy,
                        "session_ttl": a.session_ttl,
                        "use_model_intelligence": a.use_model_intelligence,
                        "search_provider": a.search_provider,
                        "intelligence_model": a.intelligence_model,
                        "max_results": a.max_results,
                        "similarity_threshold": a.similarity_threshold,
                        "max_search_results": a.max_search_results,
                        "max_scrape_urls": a.max_scrape_urls,
                        "max_context_tokens": a.max_context_tokens,
                        "rerank_provider": a.rerank_provider,
                        "rerank_model": a.rerank_model,
                        "rerank_top_n": a.rerank_top_n,
                        "cache_similarity_threshold": a.cache_similarity_threshold,
                        "cache_match_system_prompt": a.cache_match_system_prompt,
                        "cache_match_last_message_only": a.cache_match_last_message_only,
                        "cache_ttl_hours": a.cache_ttl_hours,
                        "cache_min_tokens": a.cache_min_tokens,
                        "cache_max_tokens": a.cache_max_tokens,
                        "cache_collection": a.cache_collection,
                        "tags": a.tags,
                        "description": a.description,
                        "system_prompt": a.system_prompt,
                        "enabled": a.enabled,
                        "document_store_ids": [s.id for s in a.document_stores],
                    }
                    for a in db.query(SmartAlias).all()
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
            CustomModel,
            CustomProvider,
            DailyStats,
            DocumentStore,
            ModelOverride,
            OllamaInstance,
            Provider,
            RequestLog,
            Setting,
            SmartAlias,
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
            "smart_aliases": 0,
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

                # Import smart aliases (new format)
                for a in data.get("smart_aliases", []):
                    existing = (
                        db.query(SmartAlias)
                        .filter(SmartAlias.name == a["name"])
                        .first()
                    )
                    if existing:
                        existing.target_model = a["target_model"]
                        existing.use_routing = a.get("use_routing", False)
                        existing.use_rag = a.get("use_rag", False)
                        existing.use_web = a.get("use_web", False)
                        existing.use_cache = a.get("use_cache", False)
                        existing.designator_model = a.get("designator_model")
                        existing.purpose = a.get("purpose")
                        existing.candidates_json = json.dumps(a.get("candidates", []))
                        existing.fallback_model = a.get("fallback_model")
                        existing.routing_strategy = a.get(
                            "routing_strategy", "per_request"
                        )
                        existing.session_ttl = a.get("session_ttl", 3600)
                        existing.use_model_intelligence = a.get(
                            "use_model_intelligence", False
                        )
                        existing.search_provider = a.get("search_provider")
                        existing.intelligence_model = a.get("intelligence_model")
                        existing.max_results = a.get("max_results", 5)
                        existing.similarity_threshold = a.get(
                            "similarity_threshold", 0.7
                        )
                        existing.max_search_results = a.get("max_search_results", 5)
                        existing.max_scrape_urls = a.get("max_scrape_urls", 3)
                        existing.max_context_tokens = a.get("max_context_tokens", 4000)
                        existing.rerank_provider = a.get("rerank_provider", "local")
                        existing.rerank_model = a.get(
                            "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                        )
                        existing.rerank_top_n = a.get("rerank_top_n", 20)
                        existing.cache_similarity_threshold = a.get(
                            "cache_similarity_threshold", 0.95
                        )
                        existing.cache_match_system_prompt = a.get(
                            "cache_match_system_prompt", True
                        )
                        existing.cache_match_last_message_only = a.get(
                            "cache_match_last_message_only", False
                        )
                        existing.cache_ttl_hours = a.get("cache_ttl_hours", 168)
                        existing.cache_min_tokens = a.get("cache_min_tokens", 50)
                        existing.cache_max_tokens = a.get("cache_max_tokens", 4000)
                        existing.cache_collection = a.get("cache_collection")
                        existing.tags = a.get("tags", [])
                        existing.description = a.get("description")
                        existing.system_prompt = a.get("system_prompt")
                        existing.enabled = a.get("enabled", True)
                    else:
                        alias = SmartAlias(
                            name=a["name"],
                            target_model=a["target_model"],
                            use_routing=a.get("use_routing", False),
                            use_rag=a.get("use_rag", False),
                            use_web=a.get("use_web", False),
                            use_cache=a.get("use_cache", False),
                            designator_model=a.get("designator_model"),
                            purpose=a.get("purpose"),
                            candidates_json=json.dumps(a.get("candidates", [])),
                            fallback_model=a.get("fallback_model"),
                            routing_strategy=a.get("routing_strategy", "per_request"),
                            session_ttl=a.get("session_ttl", 3600),
                            use_model_intelligence=a.get(
                                "use_model_intelligence", False
                            ),
                            search_provider=a.get("search_provider"),
                            intelligence_model=a.get("intelligence_model"),
                            max_results=a.get("max_results", 5),
                            similarity_threshold=a.get("similarity_threshold", 0.7),
                            max_search_results=a.get("max_search_results", 5),
                            max_scrape_urls=a.get("max_scrape_urls", 3),
                            max_context_tokens=a.get("max_context_tokens", 4000),
                            rerank_provider=a.get("rerank_provider", "local"),
                            rerank_model=a.get(
                                "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                            ),
                            rerank_top_n=a.get("rerank_top_n", 20),
                            cache_similarity_threshold=a.get(
                                "cache_similarity_threshold", 0.95
                            ),
                            cache_match_system_prompt=a.get(
                                "cache_match_system_prompt", True
                            ),
                            cache_match_last_message_only=a.get(
                                "cache_match_last_message_only", False
                            ),
                            cache_ttl_hours=a.get("cache_ttl_hours", 168),
                            cache_min_tokens=a.get("cache_min_tokens", 50),
                            cache_max_tokens=a.get("cache_max_tokens", 4000),
                            cache_collection=a.get("cache_collection"),
                            description=a.get("description"),
                            system_prompt=a.get("system_prompt"),
                            enabled=a.get("enabled", True),
                        )
                        alias.tags = a.get("tags", [])
                        db.add(alias)
                        db.flush()  # Get the ID for document store linking

                    # Link document stores if specified
                    store_ids = a.get("document_store_ids", [])
                    if store_ids and existing:
                        existing.document_stores = [
                            db.get(DocumentStore, sid)
                            for sid in store_ids
                            if db.get(DocumentStore, sid)
                        ]
                    elif store_ids:
                        alias.document_stores = [
                            db.get(DocumentStore, sid)
                            for sid in store_ids
                            if db.get(DocumentStore, sid)
                        ]
                    stats["smart_aliases"] += 1

                # Import legacy aliases (convert to SmartAlias)
                for a in data.get("aliases", []):
                    existing = (
                        db.query(SmartAlias)
                        .filter(SmartAlias.name == a["name"])
                        .first()
                    )
                    if existing:
                        # Skip - already exists as SmartAlias
                        continue
                    alias = SmartAlias(
                        name=a["name"],
                        target_model=a["target_model"],
                        description=a.get("description"),
                        enabled=a.get("enabled", True),
                    )
                    alias.tags = a.get("tags", [])
                    db.add(alias)
                    stats["smart_aliases"] += 1

                # Import legacy smart routers (convert to SmartAlias with routing)
                for r in data.get("smart_routers", []):
                    existing = (
                        db.query(SmartAlias)
                        .filter(SmartAlias.name == r["name"])
                        .first()
                    )
                    if existing:
                        # Skip - already exists as SmartAlias
                        continue
                    alias = SmartAlias(
                        name=r["name"],
                        target_model=r["fallback_model"],
                        use_routing=True,
                        designator_model=r["designator_model"],
                        purpose=r["purpose"],
                        candidates_json=json.dumps(r.get("candidates", [])),
                        fallback_model=r["fallback_model"],
                        routing_strategy=r.get("strategy", "per_request"),
                        session_ttl=r.get("session_ttl", 3600),
                        description=r.get("description"),
                        enabled=r.get("enabled", True),
                    )
                    alias.tags = r.get("tags", [])
                    db.add(alias)
                    stats["smart_aliases"] += 1

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

    @admin.route("/api/dashboard/smart-stats", methods=["GET"])
    @require_auth_api
    def get_smart_stats():
        """Get Smart Alias and Data Source statistics for the dashboard."""
        from sqlalchemy import func

        from db.models import DocumentStore, SmartAlias

        with get_db_context() as db:
            # Smart Alias statistics
            aliases = db.query(SmartAlias).filter(SmartAlias.enabled == True).all()

            # Aggregate stats across all aliases
            total_requests = sum(a.total_requests for a in aliases)
            total_routing = sum(a.routing_decisions for a in aliases)
            total_enrichments = sum(a.context_injections for a in aliases)
            total_searches = sum(a.search_requests for a in aliases)
            total_scrapes = sum(a.scrape_requests for a in aliases)
            total_cache_hits = sum(a.cache_hits for a in aliases)
            total_cache_saved = sum(a.cache_cost_saved for a in aliases)

            # Count aliases by feature
            aliases_with_routing = sum(1 for a in aliases if a.use_routing)
            aliases_with_rag = sum(1 for a in aliases if a.use_rag)
            aliases_with_web = sum(1 for a in aliases if a.use_web)
            aliases_with_cache = sum(1 for a in aliases if a.use_cache)
            aliases_with_memory = sum(1 for a in aliases if a.use_memory)

            # Top aliases by requests
            top_aliases = sorted(aliases, key=lambda a: a.total_requests, reverse=True)[
                :5
            ]
            top_aliases_data = [
                {
                    "id": a.id,
                    "name": a.name,
                    "requests": a.total_requests,
                    "features": {
                        "routing": a.use_routing,
                        "rag": a.use_rag,
                        "web": a.use_web,
                        "cache": a.use_cache,
                        "memory": a.use_memory,
                    },
                }
                for a in top_aliases
            ]

            # Document Store statistics
            stores = db.query(DocumentStore).all()
            total_documents = sum(s.document_count for s in stores)
            total_chunks = sum(s.chunk_count for s in stores)

            stores_by_status = {
                "ready": sum(1 for s in stores if s.index_status == "ready"),
                "indexing": sum(1 for s in stores if s.index_status == "indexing"),
                "error": sum(1 for s in stores if s.index_status == "error"),
                "pending": sum(1 for s in stores if s.index_status == "pending"),
            }

            # Stores with errors or stale indexes (not indexed in 7 days if scheduled)
            from datetime import datetime, timedelta

            stale_threshold = datetime.utcnow() - timedelta(days=7)
            stale_stores = [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": s.index_status,
                    "last_indexed": s.last_indexed.isoformat()
                    if s.last_indexed
                    else None,
                    "error": s.index_error[:100] if s.index_error else None,
                }
                for s in stores
                if s.index_status == "error"
                or (
                    s.index_schedule
                    and s.last_indexed
                    and s.last_indexed < stale_threshold
                )
            ][:5]

            return jsonify(
                {
                    "smart_aliases": {
                        "total": len(aliases),
                        "by_feature": {
                            "routing": aliases_with_routing,
                            "rag": aliases_with_rag,
                            "web": aliases_with_web,
                            "cache": aliases_with_cache,
                            "memory": aliases_with_memory,
                        },
                        "stats": {
                            "total_requests": total_requests,
                            "routing_decisions": total_routing,
                            "context_injections": total_enrichments,
                            "search_requests": total_searches,
                            "scrape_requests": total_scrapes,
                            "cache_hits": total_cache_hits,
                            "cache_cost_saved": total_cache_saved,
                        },
                        "top_aliases": top_aliases_data,
                    },
                    "document_stores": {
                        "total": len(stores),
                        "total_documents": total_documents,
                        "total_chunks": total_chunks,
                        "by_status": stores_by_status,
                        "issues": stale_stores,
                    },
                }
            )

    @admin.route("/api/dashboard/live-data-stats", methods=["GET"])
    @require_auth_api
    def get_live_data_stats():
        """Get Live Data Source statistics for the dashboard."""
        from db.live_data_sources import get_top_live_data_sources

        top_sources = get_top_live_data_sources(limit=5)

        # Calculate summary stats
        total_calls = sum(s["total_calls"] for s in top_sources)
        successful_calls = sum(s["successful_calls"] for s in top_sources)
        total_latency = sum(s["avg_latency_ms"] * s["total_calls"] for s in top_sources)

        summary = {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": total_calls - successful_calls,
            "success_rate": (
                round(successful_calls / total_calls * 100, 1) if total_calls > 0 else 0
            ),
            "avg_latency_ms": (
                round(total_latency / total_calls) if total_calls > 0 else 0
            ),
        }

        return jsonify({"top_sources": top_sources, "summary": summary})

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

    def build_model_filter_conditions(models: list[str], model_class):
        """Build SQLAlchemy filter conditions for model list.

        Handles models that may include provider prefix (e.g., "perplexity/sonar-small-chat").

        Args:
            models: List of model identifiers (may include provider prefix)
            model_class: SQLAlchemy model class (RequestLog or DailyStats)

        Returns:
            List of SQLAlchemy filter conditions to be used with or_()
        """
        from sqlalchemy import and_

        conditions = []
        for model in models:
            if "/" in model:
                # Split provider/model and match both
                provider_part, model_part = model.split("/", 1)
                conditions.append(
                    and_(
                        model_class.provider_id == provider_part,
                        model_class.model_id == model_part,
                    )
                )
            else:
                conditions.append(model_class.model_id == model)
        return conditions

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

        # Request type filter (v3.11)
        request_types_param = request.args.get("request_types")
        request_types = (
            [rt.strip() for rt in request_types_param.split(",") if rt.strip()]
            if request_types_param
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
            "request_types": request_types,
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

            # Get distinct request types (v3.11)
            request_types = [
                r[0]
                for r in db.query(distinct(RequestLog.request_type))
                .filter(RequestLog.request_type.isnot(None))
                .order_by(RequestLog.request_type)
                .all()
            ]

            return jsonify(
                {
                    "tags": tags,
                    "providers": providers,
                    "models": models,
                    "clients": clients,
                    "aliases": aliases,
                    "request_types": request_types,
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
                or filters["request_types"]
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
                    # Cache stats
                    func.count(RequestLog.id).filter(RequestLog.is_cache_hit == True),
                    func.sum(RequestLog.cache_tokens_saved),
                    func.sum(RequestLog.cache_cost_saved),
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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

                if filters["clients"]:
                    # Filter by client (hostname or IP)
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                if filters["request_types"]:
                    query = query.filter(
                        RequestLog.request_type.in_(filters["request_types"])
                    )

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
                        # Cache stats
                        "cache_hits": stats[6] or 0,
                        "cache_tokens_saved": stats[7] or 0,
                        "cache_cost_saved": round(stats[8] or 0, 4),
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

                # Get cache stats from RequestLog (not in DailyStats)
                cache_query = db.query(
                    func.count(RequestLog.id).filter(RequestLog.is_cache_hit == True),
                    func.sum(RequestLog.cache_tokens_saved),
                    func.sum(RequestLog.cache_cost_saved),
                ).filter(RequestLog.timestamp >= filters["start_date"])

                if filters["end_date"]:
                    cache_query = cache_query.filter(
                        RequestLog.timestamp <= filters["end_date"]
                    )

                cache_stats = cache_query.first()

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
                        # Cache stats
                        "cache_hits": cache_stats[0] or 0,
                        "cache_tokens_saved": cache_stats[1] or 0,
                        "cache_cost_saved": round(cache_stats[2] or 0, 4),
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
                or filters["request_types"]
            )
            if has_filters:
                # Query from RequestLog when filtering
                # Note: RequestLog.tag may contain comma-separated tags like "ben,testing"
                # We need to split and aggregate by individual tags
                query = db.query(
                    RequestLog.tag,
                    RequestLog.input_tokens,
                    RequestLog.output_tokens,
                    RequestLog.cost,
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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                if filters["request_types"]:
                    query = query.filter(
                        RequestLog.request_type.in_(filters["request_types"])
                    )

                rows = query.all()

                # Split comma-separated tags and aggregate by individual tag
                tag_stats: dict[str, dict] = {}
                for row in rows:
                    # Split "ben,testing" into ["ben", "testing"]
                    individual_tags = [
                        t.strip() for t in (row.tag or "").split(",") if t.strip()
                    ]
                    for tag in individual_tags:
                        if tag not in tag_stats:
                            tag_stats[tag] = {
                                "requests": 0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "cost": 0.0,
                            }
                        tag_stats[tag]["requests"] += 1
                        tag_stats[tag]["input_tokens"] += row.input_tokens or 0
                        tag_stats[tag]["output_tokens"] += row.output_tokens or 0
                        tag_stats[tag]["cost"] += row.cost or 0.0

                # Sort by request count descending
                sorted_tags = sorted(
                    tag_stats.items(), key=lambda x: x[1]["requests"], reverse=True
                )

                return jsonify(
                    [
                        {
                            "tag": tag,
                            "requests": stats["requests"],
                            "input_tokens": stats["input_tokens"],
                            "output_tokens": stats["output_tokens"],
                            "total_tokens": stats["input_tokens"]
                            + stats["output_tokens"],
                            "cost": round(stats["cost"], 4),
                        }
                        for tag, stats in sorted_tags
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
                or filters["request_types"]
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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                if filters["request_types"]:
                    query = query.filter(
                        RequestLog.request_type.in_(filters["request_types"])
                    )

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
            has_filters = (
                filters["tags"]
                or filters["clients"]
                or filters["aliases"]
                or filters["request_types"]
            )
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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

                if filters["request_types"]:
                    query = query.filter(
                        RequestLog.request_type.in_(filters["request_types"])
                    )

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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], DailyStats
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

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
                model_conditions = build_model_filter_conditions(
                    filters["models"], RequestLog
                )
                if model_conditions:
                    query = query.filter(or_(*model_conditions))

            if filters["clients"]:
                client_conditions = []
                for client in filters["clients"]:
                    client_conditions.append(RequestLog.hostname == client)
                    client_conditions.append(RequestLog.client_ip == client)
                query = query.filter(or_(*client_conditions))

            if filters["aliases"]:
                query = query.filter(RequestLog.alias.in_(filters["aliases"]))

            if filters["request_types"]:
                query = query.filter(
                    RequestLog.request_type.in_(filters["request_types"])
                )

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
                or filters["request_types"]
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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                if filters["request_types"]:
                    query = query.filter(
                        RequestLog.request_type.in_(filters["request_types"])
                    )

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
                or filters["request_types"]
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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

                if filters["clients"]:
                    client_conditions = []
                    for client in filters["clients"]:
                        client_conditions.append(RequestLog.hostname == client)
                        client_conditions.append(RequestLog.client_ip == client)
                    query = query.filter(or_(*client_conditions))

                if filters["aliases"]:
                    query = query.filter(RequestLog.alias.in_(filters["aliases"]))

                if filters["request_types"]:
                    query = query.filter(
                        RequestLog.request_type.in_(filters["request_types"])
                    )

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
                model_conditions = build_model_filter_conditions(models, RequestLog)
                if model_conditions:
                    query = query.filter(or_(*model_conditions))

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
        request_types = request.args.getlist("request_type")

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
                model_conditions = build_model_filter_conditions(models, RequestLog)
                if model_conditions:
                    query = query.filter(or_(*model_conditions))

            if clients:
                client_conditions = []
                for client in clients:
                    client_conditions.append(RequestLog.hostname == client)
                    client_conditions.append(RequestLog.client_ip == client)
                query = query.filter(or_(*client_conditions))

            if aliases:
                query = query.filter(RequestLog.alias.in_(aliases))

            if request_types:
                query = query.filter(RequestLog.request_type.in_(request_types))

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
    # GPU Model Cache API
    # =========================================================================

    @admin.route("/api/gpu-cache", methods=["GET"])
    @require_auth_api
    def get_gpu_cache_stats():
        """Get GPU model cache statistics."""
        try:
            from rag.model_cache import get_model_cache

            cache = get_model_cache()
            stats = cache.get_stats()

            # Add GPU memory info if available
            try:
                import torch

                if torch.cuda.is_available():
                    stats["gpu_memory_used_mb"] = round(
                        torch.cuda.memory_allocated() / 1024 / 1024, 1
                    )
                    stats["gpu_memory_reserved_mb"] = round(
                        torch.cuda.memory_reserved() / 1024 / 1024, 1
                    )
            except ImportError:
                pass

            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/gpu-cache/clear", methods=["POST"])
    @require_auth_api
    def clear_gpu_cache():
        """Clear all cached GPU models to free memory."""
        try:
            from rag.model_cache import get_model_cache

            cache = get_model_cache()
            cache.clear()
            return jsonify({"success": True, "message": "GPU model cache cleared"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Session Cache API (in-memory session-scoped caches)
    # =========================================================================

    @admin.route("/api/session-cache", methods=["GET"])
    @require_auth_api
    def get_session_cache_stats():
        """Get session cache statistics for live data context and email lookups."""
        try:
            from live.sources import get_session_cache_stats

            stats = get_session_cache_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/session-cache/reset-stats", methods=["POST"])
    @require_auth_api
    def reset_session_cache_stats():
        """Reset session cache statistics (keeps cache data, just resets counters)."""
        try:
            from live.sources import clear_session_cache_stats

            clear_session_cache_stats()
            return jsonify({"success": True, "message": "Session cache stats reset"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Live Data Cache API (database-persisted cache)
    # =========================================================================

    @admin.route("/api/live-cache", methods=["GET"])
    @require_auth_api
    def get_live_cache_stats():
        """Get live data cache statistics."""
        try:
            from db.live_cache import get_all_cache_stats

            stats = get_all_cache_stats()
            return jsonify(stats)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/live-cache/clear", methods=["POST"])
    @require_auth_api
    def clear_live_cache():
        """
        Clear live data cache entries.

        Request body (optional):
        {
            "cache_type": "data" | "entity" | "all",  // Default: "all"
            "expired_only": true | false,             // Default: false
            "source_type": "stocks"                   // Optional: filter by source
        }
        """
        try:
            from db.live_cache import (
                clear_all_caches,
                clear_data_cache,
                clear_entity_cache,
            )

            data = request.get_json() or {}
            cache_type = data.get("cache_type", "all")
            expired_only = data.get("expired_only", False)
            source_type = data.get("source_type")

            result = {}

            if cache_type in ("data", "all"):
                result["data_cache_cleared"] = clear_data_cache(
                    source_type=source_type, expired_only=expired_only
                )

            if cache_type in ("entity", "all"):
                result["entity_cache_cleared"] = clear_entity_cache(
                    source_type=source_type, expired_only=expired_only
                )

            return jsonify({"success": True, **result})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/live-cache/settings", methods=["GET"])
    @require_auth_api
    def get_live_cache_settings():
        """Get live data cache settings."""
        from db.live_cache import DEFAULT_TTLS
        from db.settings import get_setting

        # Get current settings or defaults
        settings = {
            "enabled": get_setting("live_cache_enabled") or "true",
            "ttl_price": get_setting("live_cache_ttl_price")
            or str(DEFAULT_TTLS["price"]),
            "ttl_weather_current": get_setting("live_cache_ttl_weather_current")
            or str(DEFAULT_TTLS["weather_current"]),
            "ttl_weather_forecast": get_setting("live_cache_ttl_weather_forecast")
            or str(DEFAULT_TTLS["weather_forecast"]),
            "ttl_score_live": get_setting("live_cache_ttl_score_live")
            or str(DEFAULT_TTLS["score_live"]),
            "ttl_default": get_setting("live_cache_ttl_default")
            or str(DEFAULT_TTLS["default"]),
            "entity_ttl_days": get_setting("live_cache_entity_ttl_days") or "90",
            "max_size_mb": get_setting("live_cache_max_size_mb") or "100",
        }

        # Add defaults for reference
        settings["defaults"] = {
            "ttl_price": DEFAULT_TTLS["price"],
            "ttl_weather_current": DEFAULT_TTLS["weather_current"],
            "ttl_weather_forecast": DEFAULT_TTLS["weather_forecast"],
            "ttl_score_live": DEFAULT_TTLS["score_live"],
            "ttl_default": DEFAULT_TTLS["default"],
            "entity_ttl_days": 90,
        }

        return jsonify(settings)

    @admin.route("/api/live-cache/settings", methods=["PUT"])
    @require_auth_api
    def update_live_cache_settings():
        """Update live data cache settings."""
        from db.settings import set_setting

        data = request.get_json() or {}

        # Map of allowed settings
        allowed_keys = {
            "enabled": "live_cache_enabled",
            "ttl_price": "live_cache_ttl_price",
            "ttl_weather_current": "live_cache_ttl_weather_current",
            "ttl_weather_forecast": "live_cache_ttl_weather_forecast",
            "ttl_score_live": "live_cache_ttl_score_live",
            "ttl_default": "live_cache_ttl_default",
            "entity_ttl_days": "live_cache_entity_ttl_days",
            "max_size_mb": "live_cache_max_size_mb",
        }

        updated = []
        for key, db_key in allowed_keys.items():
            if key in data:
                set_setting(db_key, str(data[key]))
                updated.append(key)

        return jsonify({"success": True, "updated": updated})

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

    # =========================================================================
    # Model Description Sync API
    # =========================================================================

    @admin.route("/api/models/sync-descriptions", methods=["POST"])
    @require_auth_api
    def sync_descriptions():
        """
        Sync model descriptions from provider APIs and OpenRouter.

        Fetches descriptions from:
        - Provider APIs (Google, Mistral, Cohere have descriptions)
        - OpenRouter public API (rich descriptions, no auth needed)

        Request body (optional):
        {
            "provider": null,         // Sync specific provider only, or "openrouter"
            "update_existing": false  // If true, overwrite existing descriptions
        }
        """
        from db import sync_model_descriptions

        data = request.get_json() or {}
        provider = data.get("provider")  # None = all providers
        update_existing = data.get("update_existing", False)

        try:
            stats = sync_model_descriptions(
                update_existing=update_existing,
                provider=provider,
            )
            return jsonify(
                {
                    "success": True,
                    "updated": stats["updated"],
                    "skipped": stats["skipped"],
                    "providers_synced": stats["providers_synced"],
                    "providers_attempted": stats.get("providers_attempted", {}),
                }
            )
        except Exception as e:
            logger.exception("Failed to sync model descriptions")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/models/sync-descriptions/providers", methods=["GET"])
    @require_auth_api
    def get_description_sync_providers():
        """Get list of providers available for description sync."""
        from db import get_available_description_providers

        return jsonify({"providers": get_available_description_providers()})

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
    # Redirects API
    # -------------------------------------------------------------------------

    @admin.route("/redirects")
    @require_auth
    def redirects_page():
        """Redirects management page."""
        return render_template("redirects.html")

    @admin.route("/api/redirects", methods=["GET"])
    @require_auth_api
    def list_redirects():
        """List all redirects."""
        from db import get_all_redirects

        redirects = get_all_redirects()
        return jsonify([r.to_dict() for r in redirects])

    @admin.route("/api/redirects/<int:redirect_id>", methods=["GET"])
    @require_auth_api
    def get_redirect(redirect_id: int):
        """Get a single redirect by ID."""
        from db import get_redirect_by_id

        redirect = get_redirect_by_id(redirect_id)
        if not redirect:
            return jsonify({"error": "Redirect not found"}), 404
        return jsonify(redirect.to_dict())

    @admin.route("/api/redirects", methods=["POST"])
    @require_auth_api
    def create_redirect_endpoint():
        """Create a new redirect."""
        from db import create_redirect, get_redirect_by_source

        data = request.get_json() or {}

        # Validate required fields
        if not data.get("source"):
            return jsonify({"error": "Source pattern is required"}), 400
        if not data.get("target"):
            return jsonify({"error": "Target is required"}), 400

        source = data["source"].lower().strip()
        target = data["target"].lower().strip()

        # Check if source already exists
        existing = get_redirect_by_source(source)
        if existing:
            return jsonify(
                {"error": f"Redirect for source '{source}' already exists"}
            ), 409

        # Parse tags
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip().lower() for t in tags.split(",") if t.strip()]

        try:
            redirect = create_redirect(
                source=source,
                target=target,
                description=data.get("description"),
                enabled=data.get("enabled", True),
                tags=tags if tags else None,
                # Caching
                use_cache=data.get("use_cache", False),
                cache_similarity_threshold=data.get("cache_similarity_threshold", 0.95),
                cache_match_system_prompt=data.get("cache_match_system_prompt", True),
                cache_match_last_message_only=data.get(
                    "cache_match_last_message_only", False
                ),
                cache_ttl_hours=data.get("cache_ttl_hours", 168),
                cache_min_tokens=data.get("cache_min_tokens", 50),
                cache_max_tokens=data.get("cache_max_tokens", 4000),
            )
            return jsonify(redirect.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/redirects/<int:redirect_id>", methods=["PUT"])
    @require_auth_api
    def update_redirect_endpoint(redirect_id: int):
        """Update an existing redirect."""
        from db import get_redirect_by_id, get_redirect_by_source, update_redirect

        data = request.get_json() or {}

        # Check redirect exists
        existing = get_redirect_by_id(redirect_id)
        if not existing:
            return jsonify({"error": "Redirect not found"}), 404

        # If source is being changed, check for conflicts
        new_source = data.get("source")
        if new_source and new_source.lower().strip() != existing.source:
            new_source = new_source.lower().strip()
            conflict = get_redirect_by_source(new_source)
            if conflict and conflict.id != redirect_id:
                return jsonify(
                    {"error": f"Redirect for source '{new_source}' already exists"}
                ), 409

        # Parse tags if provided
        tags = None
        if "tags" in data:
            tags = data["tags"]
            if isinstance(tags, str):
                tags = [t.strip().lower() for t in tags.split(",") if t.strip()]

        try:
            redirect = update_redirect(
                redirect_id=redirect_id,
                source=data.get("source"),
                target=data.get("target"),
                description=data.get("description"),
                enabled=data.get("enabled"),
                tags=tags,
                # Caching
                use_cache=data.get("use_cache"),
                cache_similarity_threshold=data.get("cache_similarity_threshold"),
                cache_match_system_prompt=data.get("cache_match_system_prompt"),
                cache_match_last_message_only=data.get("cache_match_last_message_only"),
                cache_ttl_hours=data.get("cache_ttl_hours"),
                cache_min_tokens=data.get("cache_min_tokens"),
                cache_max_tokens=data.get("cache_max_tokens"),
            )
            if not redirect:
                return jsonify({"error": "Redirect not found"}), 404
            return jsonify(redirect.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/redirects/<int:redirect_id>", methods=["DELETE"])
    @require_auth_api
    def delete_redirect_endpoint(redirect_id: int):
        """Delete a redirect."""
        from db import delete_redirect

        if delete_redirect(redirect_id):
            return jsonify({"success": True})
        return jsonify({"error": "Redirect not found"}), 404

    @admin.route("/api/redirects/check", methods=["POST"])
    @require_auth_api
    def check_redirects():
        """Check if any model names would be affected by redirects.

        Used by Smart Routers UI to warn when candidates/designator are redirected.
        """
        from db import find_matching_redirect

        data = request.get_json() or {}
        models = data.get("models", [])

        results = {}
        for model in models:
            match = find_matching_redirect(model)
            if match:
                redirect, target = match
                results[model] = {
                    "redirected": True,
                    "target": target,
                    "redirect_id": redirect.id,
                }
            else:
                results[model] = {"redirected": False}

        return jsonify(results)

    # -------------------------------------------------------------------------
    # Alerts API (System Health Monitoring)
    # -------------------------------------------------------------------------

    def _categorize_error(status_code: int, error_message: str | None = None) -> str:
        """Categorize HTTP status code into error type."""
        # First check if error message indicates a model not found error
        # (regardless of status code, since some providers wrap errors)
        if error_message:
            msg_lower = error_message.lower()
            if any(
                phrase in msg_lower
                for phrase in [
                    "invalid model",
                    "model not found",
                    "model does not exist",
                    "unknown model",
                    "no such model",
                ]
            ):
                return "not_found"

        # Then categorize by status code
        if status_code == 404:
            return "not_found"
        elif status_code in (401, 403):
            return "auth_error"
        elif status_code == 429:
            return "rate_limit"
        elif 500 <= status_code < 600:
            return "server_error"
        elif status_code == 400:
            return "bad_request"
        else:
            return "error"

    def _get_error_description(error_type: str) -> str:
        """Get human-readable description for error type."""
        descriptions = {
            "not_found": "Model not found",
            "auth_error": "Authentication failed",
            "rate_limit": "Rate limited",
            "server_error": "Server error",
            "bad_request": "Bad request",
            "error": "Request failed",
        }
        return descriptions.get(error_type, "Unknown error")

    @admin.route("/alerts")
    @require_auth
    def alerts_page():
        """Alerts management page."""
        return render_template("alerts.html")

    def _get_dismissed_alerts() -> dict:
        """Get dismissed alerts from settings. Returns {key: expiry_timestamp}."""
        with get_db_context() as db:
            setting = (
                db.query(Setting).filter(Setting.key == "dismissed_alerts").first()
            )
            if setting and setting.value:
                try:
                    dismissed = json.loads(setting.value)
                    # Clean up expired dismissals
                    now = datetime.utcnow().timestamp()
                    return {k: v for k, v in dismissed.items() if v > now}
                except (json.JSONDecodeError, TypeError):
                    pass
            return {}

    def _set_dismissed_alerts(dismissed: dict):
        """Save dismissed alerts to settings."""
        with get_db_context() as db:
            setting = (
                db.query(Setting).filter(Setting.key == "dismissed_alerts").first()
            )
            if setting:
                setting.value = json.dumps(dismissed)
            else:
                db.add(Setting(key="dismissed_alerts", value=json.dumps(dismissed)))
            db.commit()

    @admin.route("/api/alerts", methods=["GET"])
    @require_auth_api
    def get_alerts():
        """Get model alerts based on recent errors in request logs."""
        from datetime import timedelta

        from sqlalchemy import func

        hours = int(request.args.get("hours", 24))
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        # Get dismissed alerts
        dismissed = _get_dismissed_alerts()

        with get_db_context() as db:
            # Query errors grouped by provider, model, and status code
            errors = (
                db.query(
                    RequestLog.provider_id,
                    RequestLog.model_id,
                    RequestLog.status_code,
                    func.count().label("count"),
                    func.max(RequestLog.timestamp).label("last_seen"),
                    func.max(RequestLog.error_message).label("error_message"),
                )
                .filter(
                    RequestLog.timestamp >= cutoff,
                    RequestLog.status_code >= 400,
                    RequestLog.provider_id.isnot(None),
                    RequestLog.model_id.isnot(None),
                )
                .group_by(
                    RequestLog.provider_id,
                    RequestLog.model_id,
                    RequestLog.status_code,
                )
                .all()
            )

            alerts = []
            critical_count = 0
            warning_count = 0

            for error in errors:
                # Check if this alert is dismissed
                alert_key = f"{error.provider_id}/{error.model_id}/{error.status_code}"
                if alert_key in dismissed:
                    continue

                error_type = _categorize_error(error.status_code, error.error_message)
                # Model not found and auth errors are critical
                severity = (
                    "critical"
                    if error_type in ("not_found", "auth_error")
                    else "warning"
                )

                if severity == "critical":
                    critical_count += 1
                else:
                    warning_count += 1

                alerts.append(
                    {
                        "provider_id": error.provider_id,
                        "model_id": error.model_id,
                        "status_code": error.status_code,
                        "error_type": error_type,
                        "error_description": _get_error_description(error_type),
                        "error_message": error.error_message,
                        "count": error.count,
                        "last_seen": error.last_seen.isoformat()
                        if error.last_seen
                        else None,
                        "severity": severity,
                    }
                )

            # Sort by severity (critical first) then by count
            alerts.sort(
                key=lambda x: (0 if x["severity"] == "critical" else 1, -x["count"])
            )

            return jsonify(
                {
                    "alerts": alerts,
                    "summary": {
                        "critical": critical_count,
                        "warning": warning_count,
                        "total": len(alerts),
                    },
                    "period_hours": hours,
                }
            )

    @admin.route("/api/alerts/dismiss", methods=["POST"])
    @require_auth_api
    def dismiss_alert():
        """Dismiss an alert for 24 hours."""
        data = request.get_json() or {}

        provider_id = data.get("provider_id")
        model_id = data.get("model_id")
        status_code = data.get("status_code")

        if not all([provider_id, model_id, status_code]):
            return jsonify({"error": "Missing required fields"}), 400

        alert_key = f"{provider_id}/{model_id}/{status_code}"

        # Get current dismissals and add new one
        dismissed = _get_dismissed_alerts()
        # Dismiss for 24 hours
        dismissed[alert_key] = (datetime.utcnow().timestamp()) + (24 * 60 * 60)
        _set_dismissed_alerts(dismissed)

        return jsonify({"success": True, "dismissed_until": dismissed[alert_key]})

    @admin.route("/api/alerts/count", methods=["GET"])
    @require_auth_api
    def get_alerts_count():
        """Get count of alerts for badge display (lightweight query)."""
        from datetime import timedelta

        from sqlalchemy import func

        hours = int(request.args.get("hours", 24))
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        with get_db_context() as db:
            # Count distinct model/status combinations with errors
            critical_count = (
                db.query(func.count(func.distinct(RequestLog.model_id)))
                .filter(
                    RequestLog.timestamp >= cutoff,
                    RequestLog.status_code.in_([404, 401, 403]),
                    RequestLog.model_id.isnot(None),
                )
                .scalar()
                or 0
            )

            warning_count = (
                db.query(func.count(func.distinct(RequestLog.model_id)))
                .filter(
                    RequestLog.timestamp >= cutoff,
                    RequestLog.status_code >= 400,
                    ~RequestLog.status_code.in_([404, 401, 403]),
                    RequestLog.model_id.isnot(None),
                )
                .scalar()
                or 0
            )

            return jsonify(
                {
                    "critical": critical_count,
                    "warning": warning_count,
                    "total": critical_count + warning_count,
                }
            )

    # -------------------------------------------------------------------------
    # Model Intelligence API
    # -------------------------------------------------------------------------

    @admin.route("/api/model-intelligence", methods=["GET"])
    @require_auth_api
    def list_model_intelligence():
        """List all cached model intelligence entries."""
        from context import is_chroma_available

        if not is_chroma_available():
            return jsonify({"error": "ChromaDB not available", "entries": []}), 200

        try:
            from context.model_intelligence import ModelIntelligence

            mi = ModelIntelligence()
            entries = mi.get_all_cached()

            return jsonify(
                {
                    "entries": [
                        {
                            "model_id": e.model_id,
                            "intelligence": e.intelligence,
                            "strengths": e.strengths,
                            "weaknesses": e.weaknesses,
                            "best_for": e.best_for,
                            "avoid_for": e.avoid_for,
                            "sources": e.sources,
                            "generated_at": e.generated_at.isoformat(),
                            "expires_at": e.expires_at.isoformat(),
                            "expired": e.is_expired(),
                        }
                        for e in entries
                    ]
                }
            )
        except Exception as e:
            logger.error(f"Error listing model intelligence: {e}")
            return jsonify({"error": str(e), "entries": []}), 500

    @admin.route("/api/model-intelligence/<path:model_id>", methods=["GET"])
    @require_auth_api
    def get_model_intelligence(model_id: str):
        """Get cached intelligence for a specific model."""
        from context import is_chroma_available

        if not is_chroma_available():
            return jsonify({"error": "ChromaDB not available"}), 503

        try:
            from context.model_intelligence import ModelIntelligence

            mi = ModelIntelligence()
            intel = mi.get_intelligence(model_id)

            if not intel:
                return jsonify({"error": "No intelligence found for model"}), 404

            return jsonify(
                {
                    "model_id": intel.model_id,
                    "intelligence": intel.intelligence,
                    "strengths": intel.strengths,
                    "weaknesses": intel.weaknesses,
                    "best_for": intel.best_for,
                    "avoid_for": intel.avoid_for,
                    "sources": intel.sources,
                    "generated_at": intel.generated_at.isoformat(),
                    "expires_at": intel.expires_at.isoformat(),
                    "expired": intel.is_expired(),
                }
            )
        except Exception as e:
            logger.error(f"Error getting model intelligence: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/model-intelligence/refresh", methods=["POST"])
    @require_auth_api
    def refresh_model_intelligence():
        """Refresh intelligence for specified models."""
        from context import is_chroma_available

        if not is_chroma_available():
            return jsonify({"error": "ChromaDB not available"}), 503

        data = request.get_json() or {}
        model_ids = data.get("model_ids", [])
        summarizer_model = data.get("summarizer_model")
        force = data.get("force", False)

        if not model_ids:
            return jsonify({"error": "model_ids is required"}), 400

        if not summarizer_model:
            return jsonify({"error": "summarizer_model is required"}), 400

        try:
            from context.model_intelligence import ModelIntelligence

            mi = ModelIntelligence(summarizer_model=summarizer_model)
            results = mi.refresh_models(model_ids, force=force)

            return jsonify(
                {
                    "refreshed": list(results.keys()),
                    "count": len(results),
                    "details": {
                        model_id: {
                            "intelligence": intel.intelligence[:200] + "..."
                            if len(intel.intelligence) > 200
                            else intel.intelligence,
                            "strengths": intel.strengths,
                            "weaknesses": intel.weaknesses,
                        }
                        for model_id, intel in results.items()
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error refreshing model intelligence: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route(
        "/api/model-intelligence/refresh-router/<int:router_id>", methods=["POST"]
    )
    @require_auth_api
    def refresh_router_model_intelligence(router_id: int):
        """Refresh intelligence for all candidates in a smart router."""
        from context import is_chroma_available
        from db import get_smart_router_by_id

        if not is_chroma_available():
            return jsonify({"error": "ChromaDB not available"}), 503

        router = get_smart_router_by_id(router_id)
        if not router:
            return jsonify({"error": "Router not found"}), 404

        # Get all candidate model IDs
        model_ids = [c.get("model") for c in router.candidates if c.get("model")]

        if not model_ids:
            return jsonify({"error": "Router has no candidates"}), 400

        if not router.intelligence_model:
            return jsonify(
                {
                    "error": "No summarizer model configured. Please select a model in the Intelligence Settings."
                }
            ), 400

        data = request.get_json() or {}
        force = data.get("force", False)

        try:
            from context.model_intelligence import ModelIntelligence

            mi = ModelIntelligence(
                summarizer_model=router.intelligence_model,
            )
            # Use comparative refresh to get relative assessments between candidates
            results = mi.refresh_models_comparative(model_ids, force=force)

            return jsonify(
                {
                    "router": router.name,
                    "refreshed": list(results.keys()),
                    "count": len(results),
                    "total_candidates": len(model_ids),
                }
            )
        except Exception as e:
            logger.error(f"Error refreshing router model intelligence: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/model-intelligence/<path:model_id>", methods=["DELETE"])
    @require_auth_api
    def delete_model_intelligence(model_id: str):
        """Delete cached intelligence for a specific model."""
        from context import is_chroma_available

        if not is_chroma_available():
            return jsonify({"error": "ChromaDB not available"}), 503

        try:
            from context.model_intelligence import ModelIntelligence

            mi = ModelIntelligence()
            success = mi.delete_intelligence(model_id)

            if success:
                return jsonify({"success": True, "deleted": model_id})
            return jsonify({"error": "Failed to delete intelligence"}), 500
        except Exception as e:
            logger.error(f"Error deleting model intelligence: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/model-intelligence/clear", methods=["DELETE"])
    @require_auth_api
    def clear_all_model_intelligence():
        """Clear all cached model intelligence."""
        from context import is_chroma_available

        if not is_chroma_available():
            return jsonify({"error": "ChromaDB not available"}), 503

        try:
            from context.model_intelligence import ModelIntelligence

            mi = ModelIntelligence()
            success = mi.clear_all()

            if success:
                return jsonify({"success": True})
            return jsonify({"error": "Failed to clear intelligence"}), 500
        except Exception as e:
            logger.error(f"Error clearing model intelligence: {e}")
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Document Stores (v3.9)
    # =========================================================================

    @admin.route("/document-stores")
    @require_auth
    def document_stores_page():
        """Document Stores management page."""
        return render_template("document_stores.html")

    @admin.route("/rag-config")
    @require_auth
    def rag_config_page():
        """RAG Configuration page (embeddings, reranking, vision settings)."""
        return render_template("rag_config.html")

    @admin.route("/web-sources")
    @require_auth
    def web_sources_page():
        """Web Configuration page (search and scraping settings)."""
        return render_template("web_sources.html")

    @admin.route("/api/mcp-integrations/status", methods=["GET"])
    @require_auth_api
    def get_mcp_integrations_status():
        """Check which MCP integrations are available based on environment variables."""
        import os

        # Check for required tokens/credentials
        has_notion = bool(
            os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
        )
        has_github = bool(
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN")
        )
        has_slack = bool(
            os.environ.get("SLACK_BOT_TOKEN") and os.environ.get("SLACK_TEAM_ID")
        )
        has_google = bool(
            os.environ.get("GOOGLE_CLIENT_ID")
            and os.environ.get("GOOGLE_CLIENT_SECRET")
        )
        has_microsoft = bool(
            os.environ.get("MICROSOFT_CLIENT_ID")
            and os.environ.get("MICROSOFT_CLIENT_SECRET")
        )

        return jsonify(
            {
                "google": {
                    "available": has_google,
                    "reason": None
                    if has_google
                    else "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET required",
                },
                "microsoft": {
                    "available": has_microsoft,
                    "reason": None
                    if has_microsoft
                    else "MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET required",
                },
                "notion": {
                    "available": has_notion,
                    "reason": None
                    if has_notion
                    else "NOTION_TOKEN or NOTION_API_KEY required",
                },
                "github": {
                    "available": has_github,
                    "reason": None if has_github else "GITHUB_TOKEN required",
                },
                "slack": {
                    "available": has_slack,
                    "reason": None
                    if has_slack
                    else "SLACK_BOT_TOKEN and SLACK_TEAM_ID required",
                },
                "postgres": {
                    "available": True,
                    "reason": None,
                },  # Always available (connection string in args)
                "custom": {"available": True, "reason": None},  # Always available
            }
        )

    @admin.route("/api/notion/search", methods=["GET"])
    @require_auth_api
    def notion_search():
        """Search Notion for databases and pages."""
        import os

        import requests

        token = os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
        if not token:
            return jsonify({"error": "NOTION_TOKEN not configured"}), 400

        query = request.args.get("query", "")
        filter_type = request.args.get("filter", "")  # "database" or "page" or ""

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        # Build search request
        search_body = {}
        if query:
            search_body["query"] = query
        if filter_type in ("database", "page"):
            search_body["filter"] = {"property": "object", "value": filter_type}

        try:
            resp = requests.post(
                "https://api.notion.com/v1/search",
                headers=headers,
                json=search_body,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", []):
                obj_type = item.get("object")
                item_id = item.get("id", "")

                # Extract title
                title = "Untitled"
                if obj_type == "database":
                    title_arr = item.get("title", [])
                    if title_arr:
                        title = title_arr[0].get("plain_text", "Untitled")
                elif obj_type == "page":
                    props = item.get("properties", {})
                    # Try common title property names
                    for key in ["title", "Title", "Name", "name"]:
                        if key in props:
                            title_prop = props[key]
                            if title_prop.get("type") == "title":
                                title_arr = title_prop.get("title", [])
                                if title_arr:
                                    title = title_arr[0].get("plain_text", "Untitled")
                                    break
                results.append(
                    {
                        "id": item_id,
                        "type": obj_type,
                        "title": title,
                        "url": item.get("url", ""),
                    }
                )

            return jsonify({"results": results})

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Notion API error: {str(e)}"}), 500

    @admin.route("/api/notion/databases", methods=["GET"])
    @require_auth_api
    def notion_list_databases():
        """List all accessible Notion databases."""
        import os

        import requests

        token = os.environ.get("NOTION_TOKEN") or os.environ.get("NOTION_API_KEY")
        if not token:
            return jsonify({"error": "NOTION_TOKEN not configured"}), 400

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        try:
            resp = requests.post(
                "https://api.notion.com/v1/search",
                headers=headers,
                json={"filter": {"property": "object", "value": "database"}},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            databases = []
            for item in data.get("results", []):
                title_arr = item.get("title", [])
                title = (
                    title_arr[0].get("plain_text", "Untitled")
                    if title_arr
                    else "Untitled"
                )
                databases.append(
                    {
                        "id": item.get("id", ""),
                        "title": title,
                        "url": item.get("url", ""),
                    }
                )

            return jsonify({"databases": databases})

        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Notion API error: {str(e)}"}), 500

    @admin.route("/api/document-stores", methods=["GET"])
    @require_auth_api
    def list_document_stores():
        """List all document stores with Google account info and unified source status."""
        from db import get_all_document_stores
        from db.oauth_tokens import get_oauth_token_info

        # Get unified source info dynamically from registry
        unified_plugins = {}
        try:
            from plugin_base.loader import (
                get_unified_source_for_doc_type,
                unified_source_registry,
            )

            for source_type, plugin_class in unified_source_registry.get_all().items():
                unified_plugins[source_type] = {
                    "display_name": getattr(plugin_class, "display_name", source_type),
                    "supports_rag": getattr(plugin_class, "supports_rag", True),
                    "supports_live": getattr(plugin_class, "supports_live", False),
                    "supports_actions": getattr(
                        plugin_class, "supports_actions", False
                    ),
                    "icon": getattr(plugin_class, "icon", "📦"),
                }
        except Exception:
            get_unified_source_for_doc_type = None

        stores = get_all_document_stores()
        result = []
        for s in stores:
            store_dict = s.to_dict()
            # Add Google account email if applicable
            if s.google_account_id:
                token_info = get_oauth_token_info(s.google_account_id)
                if token_info:
                    store_dict["google_account_email"] = token_info.get(
                        "account_email", ""
                    )

            # Add unified source info if available (dynamic lookup)
            store_dict["unified_source"] = None
            if get_unified_source_for_doc_type:
                plugin_class = get_unified_source_for_doc_type(s.source_type)
                if plugin_class:
                    unified_type = plugin_class.source_type
                    store_dict["unified_source"] = {
                        "type": unified_type,
                        **unified_plugins.get(unified_type, {}),
                    }

            result.append(store_dict)
        return jsonify(result)

    @admin.route("/api/document-stores/<int:store_id>", methods=["GET"])
    @require_auth_api
    def get_document_store(store_id: int):
        """Get a specific document store."""
        from db import get_document_store_by_id

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404
        return jsonify(store.to_dict())

    @admin.route("/api/document-stores", methods=["POST"])
    @require_auth_api
    def create_document_store():
        """Create a new document store."""
        from db import create_document_store as db_create_store

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Name is required"}), 400

        source_type = data.get("source_type", "local")
        source_path = (
            data.get("source_path", "").strip() if data.get("source_path") else None
        )
        mcp_server_config = data.get("mcp_server_config")

        # Validate source configuration
        google_account_id = data.get("google_account_id")
        if source_type == "local":
            if not source_path:
                return jsonify(
                    {"error": "Source path is required for local sources"}
                ), 400
        elif source_type in (
            "mcp:gdrive",
            "mcp:gmail",
            "mcp:gcalendar",
            "mcp:gtasks",
            "mcp:gcontacts",
        ):
            # Google sources require OAuth account
            if not google_account_id:
                return jsonify(
                    {"error": "Google account is required for Google sources"}
                ), 400
        elif source_type == "mcp":
            if not mcp_server_config or not mcp_server_config.get("name"):
                return jsonify({"error": "MCP server configuration is required"}), 400
        elif source_type == "paperless":
            # Credentials come from PAPERLESS_URL and PAPERLESS_TOKEN env vars
            if not os.environ.get("PAPERLESS_URL") or not os.environ.get(
                "PAPERLESS_TOKEN"
            ):
                return jsonify(
                    {
                        "error": "PAPERLESS_URL and PAPERLESS_TOKEN environment variables are required"
                    }
                ), 400
        elif source_type == "mcp:github":
            # Credentials come from GITHUB_TOKEN env var
            if not os.environ.get("GITHUB_TOKEN") and not os.environ.get(
                "GITHUB_PERSONAL_ACCESS_TOKEN"
            ):
                return jsonify(
                    {"error": "GITHUB_TOKEN environment variable is required"}
                ), 400
            if not data.get("github_repo"):
                return jsonify(
                    {"error": "Repository is required for GitHub source"}
                ), 400
        elif source_type == "notion":
            # Credentials come from NOTION_TOKEN or NOTION_API_KEY env var
            if not os.environ.get("NOTION_TOKEN") and not os.environ.get(
                "NOTION_API_KEY"
            ):
                return jsonify(
                    {
                        "error": "NOTION_TOKEN or NOTION_API_KEY environment variable is required"
                    }
                ), 400
        elif source_type == "nextcloud":
            # Credentials come from NEXTCLOUD_URL, NEXTCLOUD_USER, NEXTCLOUD_PASSWORD env vars
            if (
                not os.environ.get("NEXTCLOUD_URL")
                or not os.environ.get("NEXTCLOUD_USER")
                or not os.environ.get("NEXTCLOUD_PASSWORD")
            ):
                return jsonify(
                    {
                        "error": "NEXTCLOUD_URL, NEXTCLOUD_USER, and NEXTCLOUD_PASSWORD environment variables are required"
                    }
                ), 400
        elif source_type == "website":
            # Website crawler - requires URL
            if not data.get("website_url"):
                return jsonify(
                    {"error": "Website URL is required for website sources"}
                ), 400
        elif source_type == "slack":
            # Slack - requires bot token
            if not os.environ.get("SLACK_BOT_TOKEN"):
                return jsonify(
                    {"error": "SLACK_BOT_TOKEN environment variable is required"}
                ), 400
        elif source_type == "todoist":
            # Todoist - requires API token
            if not os.environ.get("TODOIST_API_TOKEN") and not os.environ.get(
                "TODOIST_API_KEY"
            ):
                return jsonify(
                    {
                        "error": "TODOIST_API_TOKEN or TODOIST_API_KEY environment variable is required"
                    }
                ), 400
        elif source_type in ("mcp:onedrive", "mcp:outlook", "mcp:onenote", "mcp:teams"):
            # Microsoft sources require OAuth account
            microsoft_account_id = data.get("microsoft_account_id")
            if not microsoft_account_id:
                return jsonify(
                    {"error": "Microsoft account is required for Microsoft sources"}
                ), 400
        elif source_type == "websearch":
            # Web search - requires query
            if not data.get("websearch_query"):
                return jsonify(
                    {"error": "Search query is required for web search sources"}
                ), 400
        else:
            return jsonify({"error": f"Invalid source type: {source_type}"}), 400

        try:
            store = db_create_store(
                name=name,
                source_type=source_type,
                source_path=source_path,
                mcp_server_config=mcp_server_config,
                google_account_id=google_account_id,
                gdrive_folder_id=data.get("gdrive_folder_id"),
                gdrive_folder_name=data.get("gdrive_folder_name"),
                gmail_label_id=data.get("gmail_label_id"),
                gmail_label_name=data.get("gmail_label_name"),
                gcalendar_calendar_id=data.get("gcalendar_calendar_id"),
                gcalendar_calendar_name=data.get("gcalendar_calendar_name"),
                gtasks_tasklist_id=data.get("gtasks_tasklist_id"),
                gtasks_tasklist_name=data.get("gtasks_tasklist_name"),
                gcontacts_group_id=data.get("gcontacts_group_id"),
                gcontacts_group_name=data.get("gcontacts_group_name"),
                microsoft_account_id=data.get("microsoft_account_id"),
                onedrive_folder_id=data.get("onedrive_folder_id"),
                onedrive_folder_name=data.get("onedrive_folder_name"),
                outlook_folder_id=data.get("outlook_folder_id"),
                outlook_folder_name=data.get("outlook_folder_name"),
                outlook_days_back=data.get("outlook_days_back", 90),
                onenote_notebook_id=data.get("onenote_notebook_id"),
                onenote_notebook_name=data.get("onenote_notebook_name"),
                teams_team_id=data.get("teams_team_id"),
                teams_team_name=data.get("teams_team_name"),
                teams_channel_id=data.get("teams_channel_id"),
                teams_channel_name=data.get("teams_channel_name"),
                teams_days_back=data.get("teams_days_back", 90),
                paperless_url=data.get("paperless_url"),
                paperless_token=data.get("paperless_token"),
                paperless_tag_id=data.get("paperless_tag_id"),
                paperless_tag_name=data.get("paperless_tag_name"),
                github_repo=data.get("github_repo"),
                github_branch=data.get("github_branch"),
                github_path=data.get("github_path"),
                notion_database_id=data.get("notion_database_id"),
                notion_page_id=data.get("notion_page_id"),
                nextcloud_folder=data.get("nextcloud_folder"),
                website_url=data.get("website_url"),
                website_crawl_depth=data.get("website_crawl_depth", 1),
                website_max_pages=data.get("website_max_pages", 50),
                website_include_pattern=data.get("website_include_pattern"),
                website_exclude_pattern=data.get("website_exclude_pattern"),
                website_crawler_override=data.get("website_crawler_override"),
                slack_channel_id=data.get("slack_channel_id"),
                slack_channel_types=data.get("slack_channel_types", "public_channel"),
                slack_days_back=data.get("slack_days_back", 90),
                todoist_project_id=data.get("todoist_project_id"),
                todoist_project_name=data.get("todoist_project_name"),
                todoist_filter=data.get("todoist_filter"),
                todoist_include_completed=data.get("todoist_include_completed", False),
                websearch_query=data.get("websearch_query"),
                websearch_max_results=data.get("websearch_max_results", 10),
                websearch_pages_to_scrape=data.get("websearch_pages_to_scrape", 5),
                websearch_time_range=data.get("websearch_time_range"),
                websearch_category=data.get("websearch_category"),
                embedding_provider=data.get("embedding_provider", "local"),
                embedding_model=data.get("embedding_model"),
                ollama_url=data.get("ollama_url"),
                vision_provider=data.get("vision_provider", "local"),
                vision_model=data.get("vision_model"),
                vision_ollama_url=data.get("vision_ollama_url"),
                index_schedule=data.get("index_schedule"),
                max_documents=data.get("max_documents"),
                chunk_size=data.get("chunk_size", 512),
                chunk_overlap=data.get("chunk_overlap", 50),
                description=data.get("description"),
                enabled=data.get("enabled", True),
                use_temporal_filtering=data.get("use_temporal_filtering", False),
            )

            # Auto-create live source for Google Calendar/Tasks/Gmail document stores
            if (
                source_type in ("mcp:gcalendar", "mcp:gtasks", "mcp:gmail")
                and google_account_id
            ):
                try:
                    from db.live_data_sources import sync_live_source_for_document_store
                    from db.oauth_tokens import get_oauth_token_info

                    oauth_token = get_oauth_token_info(google_account_id)
                    if oauth_token:
                        sync_live_source_for_document_store(
                            doc_store_name=name,
                            doc_store_source_type=source_type,
                            google_account_id=google_account_id,
                            google_account_email=oauth_token.get("account_email")
                            or "unknown",
                            enabled=data.get("enabled", True),
                        )
                except Exception as e:
                    logger.warning(f"Failed to auto-create live source for {name}: {e}")

            return jsonify(store.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/document-stores/<int:store_id>", methods=["PUT"])
    @require_auth_api
    def update_document_store_endpoint(store_id: int):
        """Update a document store."""
        from db import update_document_store

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        try:
            store = update_document_store(
                store_id=store_id,
                name=data.get("name"),
                source_type=data.get("source_type"),
                source_path=data.get("source_path"),
                mcp_server_config=data.get("mcp_server_config"),
                google_account_id=data.get("google_account_id"),
                gdrive_folder_id=data.get("gdrive_folder_id"),
                gdrive_folder_name=data.get("gdrive_folder_name"),
                gmail_label_id=data.get("gmail_label_id"),
                gmail_label_name=data.get("gmail_label_name"),
                gcalendar_calendar_id=data.get("gcalendar_calendar_id"),
                gcalendar_calendar_name=data.get("gcalendar_calendar_name"),
                microsoft_account_id=data.get("microsoft_account_id"),
                onedrive_folder_id=data.get("onedrive_folder_id"),
                onedrive_folder_name=data.get("onedrive_folder_name"),
                outlook_folder_id=data.get("outlook_folder_id"),
                outlook_folder_name=data.get("outlook_folder_name"),
                outlook_days_back=data.get("outlook_days_back"),
                onenote_notebook_id=data.get("onenote_notebook_id"),
                onenote_notebook_name=data.get("onenote_notebook_name"),
                teams_team_id=data.get("teams_team_id"),
                teams_team_name=data.get("teams_team_name"),
                teams_channel_id=data.get("teams_channel_id"),
                teams_channel_name=data.get("teams_channel_name"),
                teams_days_back=data.get("teams_days_back"),
                paperless_url=data.get("paperless_url"),
                paperless_token=data.get("paperless_token"),
                paperless_tag_id=data.get("paperless_tag_id"),
                paperless_tag_name=data.get("paperless_tag_name"),
                github_repo=data.get("github_repo"),
                github_branch=data.get("github_branch"),
                github_path=data.get("github_path"),
                notion_database_id=data.get("notion_database_id"),
                notion_page_id=data.get("notion_page_id"),
                nextcloud_folder=data.get("nextcloud_folder"),
                website_url=data.get("website_url"),
                website_crawl_depth=data.get("website_crawl_depth"),
                website_max_pages=data.get("website_max_pages"),
                website_include_pattern=data.get("website_include_pattern"),
                website_exclude_pattern=data.get("website_exclude_pattern"),
                website_crawler_override=data.get("website_crawler_override"),
                slack_channel_id=data.get("slack_channel_id"),
                slack_channel_types=data.get("slack_channel_types"),
                slack_days_back=data.get("slack_days_back"),
                todoist_project_id=data.get("todoist_project_id"),
                todoist_project_name=data.get("todoist_project_name"),
                todoist_filter=data.get("todoist_filter"),
                todoist_include_completed=data.get("todoist_include_completed"),
                websearch_query=data.get("websearch_query"),
                websearch_max_results=data.get("websearch_max_results"),
                websearch_pages_to_scrape=data.get("websearch_pages_to_scrape"),
                websearch_time_range=data.get("websearch_time_range"),
                websearch_category=data.get("websearch_category"),
                embedding_provider=data.get("embedding_provider"),
                embedding_model=data.get("embedding_model"),
                ollama_url=data.get("ollama_url"),
                vision_provider=data.get("vision_provider"),
                vision_model=data.get("vision_model"),
                vision_ollama_url=data.get("vision_ollama_url"),
                index_schedule=data.get("index_schedule"),
                max_documents=data.get("max_documents"),
                chunk_size=data.get("chunk_size"),
                chunk_overlap=data.get("chunk_overlap"),
                description=data.get("description"),
                enabled=data.get("enabled"),
                use_temporal_filtering=data.get("use_temporal_filtering"),
            )
            if not store:
                return jsonify({"error": "Document store not found"}), 404

            # Sync live source for Google Calendar/Tasks/Gmail document stores
            if (
                store.source_type in ("mcp:gcalendar", "mcp:gtasks", "mcp:gmail")
                and store.google_account_id
            ):
                try:
                    from db.live_data_sources import sync_live_source_for_document_store
                    from db.oauth_tokens import get_oauth_token_info

                    oauth_token = get_oauth_token_info(store.google_account_id)
                    if oauth_token:
                        sync_live_source_for_document_store(
                            doc_store_name=store.name,
                            doc_store_source_type=store.source_type,
                            google_account_id=store.google_account_id,
                            google_account_email=oauth_token.get("account_email")
                            or "unknown",
                            enabled=store.enabled,
                        )
                except Exception as e:
                    logger.warning(f"Failed to sync live source for {store.name}: {e}")

            return jsonify(store.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/document-stores/<int:store_id>", methods=["DELETE"])
    @require_auth_api
    def delete_document_store_endpoint(store_id: int):
        """Delete a document store and its ChromaDB collection."""
        from db import delete_document_store, get_document_store_by_id

        # Check if any RAGs are using this store
        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        if store.smart_aliases:
            alias_names = [a.name for a in store.smart_aliases]
            return jsonify(
                {
                    "error": f"Cannot delete store - used by smart aliases: {', '.join(alias_names)}"
                }
            ), 409

        # Try to delete the ChromaDB collection first
        if store.collection_name:
            try:
                from rag import get_indexer

                indexer = get_indexer()
                indexer.delete_store_collection(store_id)
            except Exception as e:
                logger.warning(f"Failed to delete ChromaDB collection: {e}")

        # Delete associated live source for Google Calendar/Tasks/Gmail
        if store.source_type in ("mcp:gcalendar", "mcp:gtasks", "mcp:gmail"):
            try:
                from db.live_data_sources import delete_live_source_for_document_store

                delete_live_source_for_document_store(store.name)
            except Exception as e:
                logger.warning(f"Failed to delete live source for {store.name}: {e}")

        if delete_document_store(store_id):
            return jsonify({"success": True})
        return jsonify({"error": "Failed to delete document store"}), 500

    @admin.route("/api/document-stores/<int:store_id>/index", methods=["POST"])
    @require_auth_api
    def index_store_now(store_id: int):
        """Trigger immediate indexing for a document store."""
        from db import get_document_store_by_id
        from rag import get_indexer

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        try:
            indexer = get_indexer()
            if indexer.index_store(store_id, background=True):
                return jsonify({"success": True, "message": "Indexing started"})
            else:
                return jsonify({"error": "Indexing already in progress"}), 409
        except Exception as e:
            logger.error(f"Failed to start indexing: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/document-stores/<int:store_id>/cancel-index", methods=["POST"])
    @require_auth_api
    def cancel_store_indexing(store_id: int):
        """Cancel/reset a stuck indexing job for a document store."""
        from db import get_document_store_by_id
        from rag import get_indexer

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        try:
            indexer = get_indexer()
            indexer.cancel_store_indexing(store_id)
            return jsonify({"success": True, "message": "Indexing cancelled"})
        except Exception as e:
            logger.error(f"Failed to cancel indexing: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/document-stores/<int:store_id>/preview", methods=["GET"])
    @require_auth_api
    def preview_store_contents(store_id: int):
        """Preview indexed content from a document store."""
        from db import get_document_store_by_id

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        if not store.collection_name:
            return jsonify({"error": "Store has not been indexed yet"}), 400

        try:
            import chromadb

            chroma_url = os.environ.get("CHROMA_URL")
            if chroma_url:
                chroma_client = chromadb.HttpClient(host=chroma_url)
            else:
                chroma_path = os.environ.get("CHROMA_PATH", "./chroma_data")
                chroma_client = chromadb.PersistentClient(path=chroma_path)

            try:
                collection = chroma_client.get_collection(store.collection_name)
            except Exception:
                return jsonify(
                    {"error": "Collection not found - store may need reindexing"}
                ), 404

            # Get limit from query params (default 20, max 100)
            limit = min(int(request.args.get("limit", 20)), 100)
            offset = int(request.args.get("offset", 0))

            # Get documents with metadata
            results = collection.get(
                limit=limit, offset=offset, include=["documents", "metadatas"]
            )

            documents = []
            if results and results.get("documents"):
                for i, (doc, meta) in enumerate(
                    zip(results["documents"], results["metadatas"])
                ):
                    documents.append(
                        {
                            "id": results["ids"][i] if results.get("ids") else i,
                            "content": doc[:500] + "..." if len(doc) > 500 else doc,
                            "full_length": len(doc),
                            "source_uri": meta.get("source_uri", ""),
                            "source_name": meta.get("source_name", ""),
                            "chunk_index": meta.get("chunk_index", 0),
                            "timestamp": meta.get("timestamp", ""),
                        }
                    )

            return jsonify(
                {
                    "store_id": store_id,
                    "store_name": store.name,
                    "total_chunks": store.chunk_count or 0,
                    "total_documents": store.document_count or 0,
                    "showing": len(documents),
                    "offset": offset,
                    "documents": documents,
                }
            )

        except Exception as e:
            logger.error(f"Failed to preview store contents: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/document-stores/<int:store_id>/clear", methods=["POST"])
    @require_auth_api
    def clear_store_contents(store_id: int):
        """Clear all indexed content from a document store without deleting it."""
        from db import get_document_store_by_id, update_document_store_index_status
        from rag import get_indexer

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        try:
            indexer = get_indexer()
            # Cancel any running indexing first
            indexer.cancel_store_indexing(store_id)
            # Delete the ChromaDB collection
            indexer.delete_store_collection(store_id)
            # Reset the store stats
            update_document_store_index_status(
                store_id, "pending", document_count=0, chunk_count=0
            )
            return jsonify({"success": True, "message": "Store contents cleared"})
        except Exception as e:
            logger.error(f"Failed to clear store contents: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/document-stores/<int:store_id>/test-unified", methods=["POST"])
    @require_auth_api
    def test_unified_source(store_id: int):
        """
        Test the unified source for a document store.

        Tests both the RAG (document listing) and Live (API query) sides
        if the store has a unified source available.
        """
        from db import get_document_store_by_id

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        try:
            from plugin_base.loader import get_unified_source_for_doc_type

            # Dynamic lookup - find unified source that handles this doc store type
            plugin_class = get_unified_source_for_doc_type(store.source_type)
            if not plugin_class:
                return jsonify(
                    {"error": f"No unified source available for {store.source_type}"}
                ), 400

            unified_type = plugin_class.source_type

            # Build config using the plugin's own method
            # Uses PluginConfig if available, falls back to legacy columns
            config = plugin_class.get_config_for_store(store)

            # Instantiate and test the plugin
            plugin = plugin_class(config)
            success, message = plugin.test_connection()

            # Get capability info
            supports_rag = getattr(plugin_class, "supports_rag", True)
            supports_live = getattr(plugin_class, "supports_live", False)

            return jsonify(
                {
                    "success": success,
                    "message": message,
                    "unified_type": unified_type,
                    "capabilities": {
                        "rag": supports_rag,
                        "live": supports_live,
                        "actions": getattr(plugin_class, "supports_actions", False),
                    },
                }
            )

        except Exception as e:
            logger.error(f"Failed to test unified source: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route(
        "/api/document-stores/<int:store_id>/refresh-intelligence", methods=["POST"]
    )
    @require_auth_api
    def refresh_store_intelligence(store_id: int):
        """Regenerate intelligence (themes, best_for, summary) for a document store."""
        from db import get_document_store_by_id
        from rag import get_indexer

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        if store.index_status != "ready" or store.chunk_count == 0:
            return jsonify(
                {"error": "Store must be indexed before generating intelligence"}
            ), 400

        try:
            indexer = get_indexer()
            # Get model from request or use default
            data = request.get_json(silent=True) or {}
            model = data.get("model")

            success = indexer.generate_store_intelligence(store_id, model)
            if success:
                # Reload store to get updated data
                store = get_document_store_by_id(store_id)
                return jsonify(
                    {
                        "success": True,
                        "themes": store.themes,
                        "best_for": store.best_for,
                        "content_summary": store.content_summary,
                    }
                )
            else:
                return jsonify(
                    {
                        "error": "Intelligence generation failed - check if model is configured"
                    }
                ), 500
        except Exception as e:
            logger.error(f"Failed to generate intelligence: {e}")
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------------
    # Plugin Config Routes for Document Stores
    # -------------------------------------------------------------------------

    @admin.route("/api/document-stores/<int:store_id>/plugin-config", methods=["GET"])
    @require_auth_api
    def get_store_plugin_config(store_id: int):
        """Get the PluginConfig for a document store."""
        from db import get_document_store_by_id
        from db.plugin_configs import get_plugin_config

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        if not store.plugin_config_id:
            return jsonify({"plugin_config": None})

        plugin_config = get_plugin_config(store.plugin_config_id)
        if not plugin_config:
            return jsonify({"plugin_config": None})

        return jsonify({"plugin_config": plugin_config.to_dict()})

    @admin.route("/api/document-stores/<int:store_id>/plugin-config", methods=["POST"])
    @require_auth_api
    def create_store_plugin_config(store_id: int):
        """
        Create a PluginConfig for a document store.

        This creates a new PluginConfig with the provided config and links it
        to the document store. If the store already has a PluginConfig, this
        will fail - use PUT to update instead.

        Request body:
        {
            "config": { ... plugin-specific config ... }
        }
        """
        from db import get_document_store_by_id, update_document_store
        from db.plugin_configs import create_plugin_config

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        if store.plugin_config_id:
            return jsonify(
                {"error": "Store already has a PluginConfig - use PUT to update"}
            ), 400

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        config = data.get("config", {})

        # Create the PluginConfig
        plugin_config = create_plugin_config(
            plugin_type="unified_source",
            source_type=store.source_type,
            name=f"{store.name}_config",
            config=config,
            enabled=True,
        )

        if not plugin_config:
            return jsonify({"error": "Failed to create PluginConfig"}), 500

        # Link it to the document store
        updated_store = update_document_store(
            store_id=store_id,
            plugin_config_id=plugin_config.id,
        )

        if not updated_store:
            return jsonify({"error": "Failed to link PluginConfig to store"}), 500

        return jsonify(
            {
                "plugin_config": plugin_config.to_dict(),
                "store": updated_store.to_dict(),
            }
        ), 201

    @admin.route("/api/document-stores/<int:store_id>/plugin-config", methods=["PUT"])
    @require_auth_api
    def update_store_plugin_config(store_id: int):
        """
        Update the PluginConfig for a document store.

        If the store doesn't have a PluginConfig, one will be created.

        Request body:
        {
            "config": { ... plugin-specific config ... }
        }
        """
        from db import get_document_store_by_id, update_document_store
        from db.plugin_configs import (
            create_plugin_config,
            get_plugin_config,
            update_plugin_config,
        )

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        config = data.get("config", {})

        if store.plugin_config_id:
            # Update existing PluginConfig
            plugin_config = update_plugin_config(
                config_id=store.plugin_config_id,
                config=config,
            )
            if not plugin_config:
                return jsonify({"error": "Failed to update PluginConfig"}), 500
        else:
            # Create new PluginConfig and link it
            plugin_config = create_plugin_config(
                plugin_type="unified_source",
                source_type=store.source_type,
                name=f"{store.name}_config",
                config=config,
                enabled=True,
            )
            if not plugin_config:
                return jsonify({"error": "Failed to create PluginConfig"}), 500

            # Link it to the document store
            update_document_store(
                store_id=store_id,
                plugin_config_id=plugin_config.id,
            )

        # Reload store to get updated data
        store = get_document_store_by_id(store_id)

        return jsonify(
            {
                "plugin_config": plugin_config.to_dict(),
                "store": store.to_dict(),
            }
        )

    @admin.route(
        "/api/document-stores/<int:store_id>/plugin-config", methods=["DELETE"]
    )
    @require_auth_api
    def delete_store_plugin_config(store_id: int):
        """
        Delete the PluginConfig for a document store.

        This unlinks and deletes the PluginConfig. The store will fall back
        to using legacy column-based configuration.
        """
        from db import get_document_store_by_id, update_document_store
        from db.plugin_configs import delete_plugin_config

        store = get_document_store_by_id(store_id)
        if not store:
            return jsonify({"error": "Document store not found"}), 404

        if not store.plugin_config_id:
            return jsonify({"error": "Store has no PluginConfig"}), 404

        plugin_config_id = store.plugin_config_id

        # Unlink from store first
        update_document_store(
            store_id=store_id,
            plugin_config_id=0,  # Will be treated as None
        )

        # Delete the PluginConfig
        deleted = delete_plugin_config(plugin_config_id)
        if not deleted:
            return jsonify({"error": "Failed to delete PluginConfig"}), 500

        return jsonify({"success": True})

    # -------------------------------------------------------------------------
    # OAuth Routes (Google, etc.)
    # -------------------------------------------------------------------------

    # Google OAuth scopes for different services
    # All include userinfo.email to identify the account
    # Write scopes are included for Smart Actions (draft creation, event creation, etc.)
    GOOGLE_SCOPES = {
        "drive": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",  # Create/edit files created by app
        ],
        "gmail": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",  # Create drafts and send emails
            "https://www.googleapis.com/auth/gmail.modify",  # Mark read/unread, archive, label
        ],
        "calendar": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",  # Create/edit calendar events
            "https://www.googleapis.com/auth/calendar",  # Full access including delete
        ],
        "tasks": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/tasks.readonly",
            "https://www.googleapis.com/auth/tasks",  # Create/edit tasks
        ],
        "contacts": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/contacts.readonly",
            "https://www.googleapis.com/auth/contacts",  # Create/edit contacts
        ],
        "workspace": [
            "https://www.googleapis.com/auth/userinfo.email",
            # Drive
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.file",
            # Gmail
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.modify",  # Mark read/unread, archive, label
            # Calendar
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/calendar",  # Full access including delete
            # Tasks
            "https://www.googleapis.com/auth/tasks.readonly",
            "https://www.googleapis.com/auth/tasks",
            # Contacts
            "https://www.googleapis.com/auth/contacts.readonly",
            "https://www.googleapis.com/auth/contacts",
        ],
        "places": [
            "https://www.googleapis.com/auth/userinfo.email",
        ],
    }

    @admin.route("/api/oauth/tokens", methods=["GET"])
    @require_auth_api
    def list_oauth_tokens():
        """List all stored OAuth tokens (metadata only)."""
        from db.oauth_tokens import list_oauth_tokens

        provider = request.args.get("provider")
        tokens = list_oauth_tokens(provider=provider)
        return jsonify(tokens)

    @admin.route("/api/oauth/tokens/<int:token_id>", methods=["DELETE"])
    @require_auth_api
    def delete_oauth_token(token_id: int):
        """Delete an OAuth token."""
        from db.oauth_tokens import delete_oauth_token

        if delete_oauth_token(token_id):
            return jsonify({"success": True})
        return jsonify({"error": "Token not found"}), 404

    @admin.route("/api/oauth/google/start", methods=["GET"])
    @require_auth_api
    def google_oauth_start():
        """
        Start Google OAuth flow.

        Query params:
            - scope: "drive", "gmail", "calendar", or "workspace" (default: workspace)
            - redirect_uri: Where to redirect after auth (default: /rags)

        Returns redirect URL to Google consent screen.
        """
        import secrets
        import urllib.parse

        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        if not client_id:
            return jsonify(
                {
                    "error": "GOOGLE_CLIENT_ID not configured",
                    "setup_required": True,
                }
            ), 400

        # Always request all scopes (workspace) so one account works for Drive, Gmail, Calendar
        # This avoids confusing UX where users need separate accounts per service
        scopes = GOOGLE_SCOPES["workspace"]

        # Generate state token for CSRF protection
        # Store in database instead of session to avoid cookie issues with OAuth redirects
        import json

        from db import set_setting

        state = secrets.token_urlsafe(32)
        oauth_data = {
            "redirect": request.args.get("redirect_uri", "/document-stores"),
            "scopes": scopes,
        }
        set_setting(f"oauth_state_{state}", json.dumps(oauth_data))

        # Build the OAuth URL
        # Determine our callback URL - use EXTERNAL_URL if set (for reverse proxy setups)
        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.google_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.google_oauth_callback"
            )

        params = {
            "client_id": client_id,
            "redirect_uri": callback_url,
            "scope": " ".join(scopes),
            "response_type": "code",
            "state": state,
            "access_type": "offline",  # Get refresh token
            "prompt": "consent",  # Always show consent to get refresh token
        }

        auth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth?"
            + urllib.parse.urlencode(params)
        )

        return jsonify({"auth_url": auth_url})

    @admin.route("/api/document-stores/oauth/callback", methods=["GET"])
    def google_oauth_callback():
        """
        Handle Google OAuth callback.

        Google redirects here after user consents. We exchange the code for tokens.
        """
        import requests as http_requests

        error = request.args.get("error")
        if error:
            logger.error(f"Google OAuth error: {error}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Google authentication failed: {error}",
            )

        # Verify state - retrieve from database
        import json

        from db import delete_setting, get_setting

        state = request.args.get("state")
        oauth_data_json = get_setting(f"oauth_state_{state}") if state else None

        if not state or not oauth_data_json:
            logger.error("OAuth state mismatch or not found")
            return render_template(
                "oauth_result.html",
                success=False,
                error="Invalid OAuth state. Please try again.",
            )

        # Parse stored OAuth data and clean up
        oauth_data = json.loads(oauth_data_json)
        delete_setting(f"oauth_state_{state}")  # One-time use

        code = request.args.get("code")
        if not code:
            return render_template(
                "oauth_result.html",
                success=False,
                error="No authorization code received.",
            )

        # Exchange code for tokens
        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

        # Use EXTERNAL_URL if set (for reverse proxy setups)
        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.google_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.google_oauth_callback"
            )

        try:
            token_response = http_requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": callback_url,
                    "grant_type": "authorization_code",
                },
                timeout=30,
            )
            token_response.raise_for_status()
            token_data = token_response.json()
        except Exception as e:
            logger.error(f"Failed to exchange OAuth code: {e}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Failed to get tokens: {e}",
            )

        if "refresh_token" not in token_data:
            logger.warning(
                "No refresh_token in response - user may have already authorized"
            )

        # Get user info to identify the account
        try:
            userinfo_response = http_requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
                timeout=30,
            )
            userinfo_response.raise_for_status()
            userinfo = userinfo_response.json()
            account_email = userinfo.get("email", "unknown")
            account_name = userinfo.get("name")
        except Exception as e:
            logger.warning(f"Failed to get user info: {e}")
            account_email = "unknown"
            account_name = None

        # Store the tokens
        from db.oauth_tokens import store_oauth_token

        scopes = oauth_data.get("scopes", [])

        # Add client credentials to token data for later refresh
        token_data["client_id"] = client_id
        token_data["client_secret"] = client_secret
        token_data["token_uri"] = "https://oauth2.googleapis.com/token"

        token_id = store_oauth_token(
            provider="google",
            account_email=account_email,
            token_data=token_data,
            scopes=scopes,
            account_name=account_name,
        )

        redirect_url = oauth_data.get("redirect", "/document-stores")

        return render_template(
            "oauth_result.html",
            success=True,
            account_id=token_id,
            account_email=account_email,
            account_name=account_name,
            redirect_url=redirect_url,
        )

    @admin.route("/api/oauth/google/status", methods=["GET"])
    @require_auth_api
    def google_oauth_status():
        """
        Check Google OAuth configuration status.

        Returns whether client credentials are configured and list of connected accounts.
        """
        from db.oauth_tokens import list_oauth_tokens

        client_id = os.environ.get("GOOGLE_CLIENT_ID")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

        configured = bool(client_id and client_secret)
        accounts = list_oauth_tokens(provider="google") if configured else []

        return jsonify(
            {
                "configured": configured,
                "accounts": accounts,
            }
        )

    @admin.route("/api/oauth/accounts/<int:account_id>", methods=["DELETE"])
    @require_auth_api
    def delete_oauth_account(account_id):
        """Delete an OAuth account by ID."""
        from db.oauth_tokens import delete_oauth_token

        if delete_oauth_token(account_id):
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Account not found"}), 404

    @admin.route("/api/oauth/google/<int:account_id>/folders", methods=["GET"])
    @require_auth_api
    def list_google_drive_folders(account_id):
        """
        List folders from a Google Drive account using the Drive API directly.

        Query params:
            parent: Optional parent folder ID to list subfolders of.
                    If not provided or 'root', lists root-level folders.

        Returns a list of folders the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id, list_oauth_tokens

        parent_id = request.args.get("parent", "root") or "root"

        # Get the OAuth token
        tokens = list_oauth_tokens(provider="google")
        token_meta = next((t for t in tokens if t["id"] == account_id), None)
        if not token_meta:
            return jsonify({"error": "OAuth account not found"}), 404

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "Failed to get OAuth token"}), 400

        access_token = token_data.get("access_token")
        if not access_token:
            return jsonify({"error": "No access token available"}), 400

        folders = []
        try:
            # Use Google Drive API directly
            # Query: folders that have the specified parent
            query = f"'{parent_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"

            page_token = None
            max_pages = 10

            for _ in range(max_pages):
                params = {
                    "q": query,
                    "fields": "nextPageToken, files(id, name)",
                    "pageSize": 100,
                    "orderBy": "name",
                }
                if page_token:
                    params["pageToken"] = page_token

                response = http_requests.get(
                    "https://www.googleapis.com/drive/v3/files",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=30,
                )

                if response.status_code == 401:
                    # Token expired - try to refresh
                    refresh_token = token_data.get("refresh_token")
                    client_id = token_data.get("client_id")
                    client_secret = token_data.get("client_secret")

                    if refresh_token and client_id and client_secret:
                        refresh_response = http_requests.post(
                            "https://oauth2.googleapis.com/token",
                            data={
                                "client_id": client_id,
                                "client_secret": client_secret,
                                "refresh_token": refresh_token,
                                "grant_type": "refresh_token",
                            },
                            timeout=30,
                        )
                        if refresh_response.status_code == 200:
                            new_token_data = refresh_response.json()
                            access_token = new_token_data.get("access_token")
                            # Update stored token
                            from db.oauth_tokens import update_oauth_token_data

                            token_data["access_token"] = access_token
                            update_oauth_token_data(account_id, token_data)
                            # Retry the request
                            response = http_requests.get(
                                "https://www.googleapis.com/drive/v3/files",
                                headers={"Authorization": f"Bearer {access_token}"},
                                params=params,
                                timeout=30,
                            )
                        else:
                            return jsonify(
                                {"error": "Token expired and refresh failed"}
                            ), 401
                    else:
                        return jsonify(
                            {"error": "Token expired and no refresh token available"}
                        ), 401

                if response.status_code != 200:
                    logger.error(
                        f"Drive API error: {response.status_code} - {response.text}"
                    )
                    return jsonify(
                        {"error": f"Drive API error: {response.status_code}"}
                    ), 500

                data = response.json()
                for file in data.get("files", []):
                    folders.append({"id": file["id"], "name": file["name"]})

                page_token = data.get("nextPageToken")
                if not page_token:
                    break

        except Exception as e:
            logger.error(f"Failed to list Google Drive folders: {e}")
            return jsonify({"error": str(e)}), 500

        return jsonify(
            {
                "folders": folders,
                "parent_id": parent_id if parent_id != "root" else "",
            }
        )

    @admin.route("/api/local-folders", methods=["GET"])
    @require_auth_api
    def list_local_folders():
        """
        List folders from the local filesystem for document store configuration.

        Query params:
            path: The path to list folders from. Defaults to /.

        Returns a list of folders the user can select for indexing.
        Excludes system directories for cleaner browsing.
        """
        from pathlib import Path

        # Directories to exclude from browsing (system/uninteresting dirs)
        EXCLUDED_DIRS = {
            "proc",
            "sys",
            "dev",
            "run",
            "snap",
            "boot",
            "lib",
            "lib64",
            "sbin",
            "bin",
            "usr",
            "etc",
            "var",
            "tmp",
            "lost+found",
            "__pycache__",
            "node_modules",
            ".git",
            ".venv",
            "venv",
        }

        # Get requested path, default to /
        requested_path = request.args.get("path", "/")

        try:
            target_path = Path(requested_path).resolve()
        except Exception:
            return jsonify({"error": "Invalid path"}), 400

        # Check if path exists
        if not target_path.exists():
            return jsonify({"error": f"Path does not exist: {requested_path}"}), 404

        if not target_path.is_dir():
            return jsonify({"error": f"Path is not a directory: {requested_path}"}), 400

        folders = []
        root = Path("/")
        try:
            for item in sorted(target_path.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    # Skip excluded directories at root level
                    if target_path == root and item.name in EXCLUDED_DIRS:
                        continue
                    folders.append(
                        {
                            "name": item.name,
                            "path": str(item),
                        }
                    )
        except PermissionError:
            return jsonify({"error": "Permission denied"}), 403
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        # Build breadcrumb path from root
        breadcrumbs = []
        current = target_path
        while current != root and current != current.parent:
            breadcrumbs.insert(0, {"name": current.name, "path": str(current)})
            current = current.parent

        return jsonify(
            {
                "folders": folders,
                "current_path": str(target_path),
                "parent_path": str(target_path.parent) if target_path != root else None,
                "breadcrumbs": breadcrumbs,
                "is_root": target_path == root,
            }
        )

    @admin.route("/api/nextcloud-folders", methods=["GET"])
    @require_auth_api
    def list_nextcloud_folders():
        """
        List folders from Nextcloud using WebDAV API.

        Query params:
            path: The path to list folders from. Defaults to root.

        Returns a list of folders the user can select for indexing.
        """
        import xml.etree.ElementTree as ET

        import requests as http_requests

        base_url = os.environ.get("NEXTCLOUD_URL", "").rstrip("/")
        username = os.environ.get("NEXTCLOUD_USER", "")
        password = os.environ.get("NEXTCLOUD_PASSWORD", "")

        if not base_url or not username or not password:
            return jsonify({"error": "Nextcloud credentials not configured"}), 400

        # Get requested path, default to root
        requested_path = request.args.get("path", "").strip("/")

        # Build WebDAV URL
        webdav_url = f"{base_url}/remote.php/dav/files/{username}"
        if requested_path:
            webdav_url = f"{webdav_url}/{requested_path}"

        # PROPFIND request to list contents (Depth: 1 for immediate children only)
        propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
<d:propfind xmlns:d="DAV:">
    <d:prop>
        <d:resourcetype/>
        <d:displayname/>
    </d:prop>
</d:propfind>"""

        try:
            response = http_requests.request(
                "PROPFIND",
                webdav_url,
                auth=(username, password),
                headers={
                    "Depth": "1",
                    "Content-Type": "application/xml",
                },
                data=propfind_body,
                timeout=15,
            )

            if response.status_code not in (200, 207):
                return (
                    jsonify({"error": f"Nextcloud error: {response.status_code}"}),
                    response.status_code,
                )

            # Check for XML content type - if HTML, the URL is probably wrong
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                return (
                    jsonify(
                        {
                            "error": "Nextcloud returned HTML instead of XML. Check that NEXTCLOUD_URL points to a valid Nextcloud instance."
                        }
                    ),
                    502,
                )

            # Parse XML response
            root = ET.fromstring(response.content)
            ns = {"d": "DAV:"}

            folders = []
            base_path = f"/remote.php/dav/files/{username}/"

            for response_elem in root.findall("d:response", ns):
                href = response_elem.find("d:href", ns)
                if href is None:
                    continue

                # Extract path from href
                path = href.text
                if path.startswith(base_path):
                    path = path[len(base_path) :]
                path = path.strip("/")

                # Skip the current directory (first result is always the requested dir)
                if path == requested_path:
                    continue

                # Check if it's a collection (folder)
                propstat = response_elem.find("d:propstat", ns)
                if propstat is None:
                    continue

                prop = propstat.find("d:prop", ns)
                if prop is None:
                    continue

                resourcetype = prop.find("d:resourcetype", ns)
                if resourcetype is None:
                    continue

                # Only include folders (collections)
                collection = resourcetype.find("d:collection", ns)
                if collection is None:
                    continue

                # Get display name or use path segment
                displayname = prop.find("d:displayname", ns)
                name = (
                    displayname.text
                    if displayname is not None and displayname.text
                    else path.split("/")[-1]
                )

                folders.append({"name": name, "path": f"/{path}"})

            # Sort folders by name
            folders.sort(key=lambda x: x["name"].lower())

            # Build breadcrumb path
            breadcrumbs = []
            if requested_path:
                parts = requested_path.split("/")
                for i, part in enumerate(parts):
                    crumb_path = "/" + "/".join(parts[: i + 1])
                    breadcrumbs.append({"name": part, "path": crumb_path})

            return jsonify(
                {
                    "folders": folders,
                    "current_path": f"/{requested_path}" if requested_path else "/",
                    "breadcrumbs": breadcrumbs,
                    "is_root": not requested_path,
                }
            )

        except http_requests.exceptions.Timeout:
            return jsonify({"error": "Nextcloud request timed out"}), 504
        except http_requests.exceptions.ConnectionError:
            return jsonify({"error": "Could not connect to Nextcloud"}), 503
        except ET.ParseError as e:
            return jsonify({"error": f"Invalid XML response: {e}"}), 500
        except Exception as e:
            logger.error(f"Error listing Nextcloud folders: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/paperless-tags", methods=["GET"])
    @require_auth_api
    def list_paperless_tags():
        """
        List tags from Paperless-ngx.

        Returns a list of tags the user can select to filter document indexing.
        """
        import requests as http_requests

        base_url = os.environ.get("PAPERLESS_URL", "").rstrip("/")
        api_token = os.environ.get("PAPERLESS_TOKEN", "")

        if not base_url or not api_token:
            return jsonify({"error": "Paperless credentials not configured"}), 400

        try:
            # Fetch all tags (handle pagination)
            all_tags = []
            url = f"{base_url}/api/tags/"

            while url:
                response = http_requests.get(
                    url,
                    headers={"Authorization": f"Token {api_token}"},
                    timeout=15,
                )

                if response.status_code != 200:
                    return (
                        jsonify({"error": f"Paperless error: {response.status_code}"}),
                        response.status_code,
                    )

                data = response.json()
                results = data.get("results", [])

                for tag in results:
                    all_tags.append(
                        {
                            "id": tag.get("id"),
                            "name": tag.get("name", ""),
                            "color": tag.get("color", "#a6cee3"),
                            "document_count": tag.get("document_count", 0),
                        }
                    )

                # Get next page if exists
                url = data.get("next")

            # Sort by name
            all_tags.sort(key=lambda x: x["name"].lower())

            return jsonify({"tags": all_tags})

        except http_requests.exceptions.Timeout:
            return jsonify({"error": "Paperless request timed out"}), 504
        except http_requests.exceptions.ConnectionError:
            return jsonify({"error": "Could not connect to Paperless"}), 503
        except Exception as e:
            logger.error(f"Error listing Paperless tags: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/slack/channels", methods=["GET"])
    @require_auth_api
    def list_slack_channels():
        """
        List channels from Slack.

        Returns a list of channels the bot can access.
        """
        import requests as http_requests

        bot_token = os.environ.get("SLACK_BOT_TOKEN", "")

        if not bot_token:
            return jsonify({"error": "Slack bot token not configured"}), 400

        try:
            # Fetch conversations list
            all_channels = []
            cursor = None

            while True:
                params = {
                    "types": "public_channel,private_channel",
                    "exclude_archived": "true",
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor

                response = http_requests.get(
                    "https://slack.com/api/conversations.list",
                    headers={"Authorization": f"Bearer {bot_token}"},
                    params=params,
                    timeout=15,
                )

                if response.status_code != 200:
                    return (
                        jsonify({"error": f"Slack error: {response.status_code}"}),
                        response.status_code,
                    )

                data = response.json()

                if not data.get("ok"):
                    error = data.get("error", "Unknown error")
                    return jsonify({"error": f"Slack API error: {error}"}), 400

                channels = data.get("channels", [])
                for channel in channels:
                    all_channels.append(
                        {
                            "id": channel.get("id"),
                            "name": channel.get("name", ""),
                            "is_private": channel.get("is_private", False),
                            "num_members": channel.get("num_members", 0),
                        }
                    )

                # Check for pagination
                cursor = data.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break

            # Sort by name
            all_channels.sort(key=lambda x: x["name"].lower())

            return jsonify({"channels": all_channels})

        except http_requests.exceptions.Timeout:
            return jsonify({"error": "Slack request timed out"}), 504
        except http_requests.exceptions.ConnectionError:
            return jsonify({"error": "Could not connect to Slack"}), 503
        except Exception as e:
            logger.error(f"Error listing Slack channels: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/todoist/projects", methods=["GET"])
    @require_auth_api
    def list_todoist_projects():
        """
        List projects from Todoist.

        Returns a list of projects the user can select to limit indexing scope.
        """
        import requests as http_requests

        # Support both TODOIST_API_TOKEN and TODOIST_API_KEY for compatibility
        api_token = os.environ.get("TODOIST_API_TOKEN") or os.environ.get(
            "TODOIST_API_KEY", ""
        )

        if not api_token:
            return jsonify({"error": "Todoist API token not configured"}), 400

        try:
            response = http_requests.get(
                "https://api.todoist.com/rest/v2/projects",
                headers={"Authorization": f"Bearer {api_token}"},
                timeout=15,
            )

            if response.status_code != 200:
                return (
                    jsonify({"error": f"Todoist error: {response.status_code}"}),
                    response.status_code,
                )

            projects = response.json()

            # Format projects for the dropdown
            formatted_projects = []
            for project in projects:
                formatted_projects.append(
                    {
                        "id": project.get("id"),
                        "name": project.get("name", ""),
                        "color": project.get("color", ""),
                        "is_favorite": project.get("is_favorite", False),
                    }
                )

            # Sort by name
            formatted_projects.sort(key=lambda x: x["name"].lower())

            return jsonify({"projects": formatted_projects})

        except http_requests.exceptions.Timeout:
            return jsonify({"error": "Todoist request timed out"}), 504
        except http_requests.exceptions.ConnectionError:
            return jsonify({"error": "Could not connect to Todoist"}), 503
        except Exception as e:
            logger.error(f"Error listing Todoist projects: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/oauth/google/<int:account_id>/labels", methods=["GET"])
    @require_auth_api
    def list_gmail_labels(account_id):
        """
        List standard Gmail labels.

        Returns a list of labels the user can select to limit indexing scope.
        Gmail labels include system labels (INBOX, SENT, etc.).
        """
        # Provide standard Gmail labels
        # Gmail API doesn't easily enumerate custom labels without full OAuth,
        # so we provide the standard system labels
        standard_labels = [
            {"id": "", "name": "All emails (no filter)"},
            {"id": "INBOX", "name": "Inbox"},
            {"id": "SENT", "name": "Sent"},
            {"id": "DRAFT", "name": "Drafts"},
            {"id": "STARRED", "name": "Starred"},
            {"id": "IMPORTANT", "name": "Important"},
            {"id": "UNREAD", "name": "Unread"},
            {"id": "SPAM", "name": "Spam"},
            {"id": "TRASH", "name": "Trash"},
            {"id": "CATEGORY_PERSONAL", "name": "Category: Personal"},
            {"id": "CATEGORY_SOCIAL", "name": "Category: Social"},
            {"id": "CATEGORY_PROMOTIONS", "name": "Category: Promotions"},
            {"id": "CATEGORY_UPDATES", "name": "Category: Updates"},
            {"id": "CATEGORY_FORUMS", "name": "Category: Forums"},
        ]

        return jsonify({"labels": standard_labels})

    @admin.route("/api/oauth/google/<int:account_id>/calendars", methods=["GET"])
    @require_auth_api
    def list_google_calendars(account_id):
        """
        List calendars from a Google Calendar account using direct API.

        Returns a list of calendars the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        # Try the access token first
        response = http_requests.get(
            "https://www.googleapis.com/calendar/v3/users/me/calendarList",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                # Update stored token
                from db.oauth_tokens import update_oauth_token_data

                update_oauth_token_data(account_id, {"access_token": access_token})

                # Retry the request
                response = http_requests.get(
                    "https://www.googleapis.com/calendar/v3/users/me/calendarList",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list calendars: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list calendars: {response.status_code}"}
            ), 500

        data = response.json()
        calendars = [{"id": "", "name": "Primary calendar only"}]

        for cal in data.get("items", []):
            cal_id = cal.get("id", "")
            summary = cal.get("summary", cal_id)
            primary = cal.get("primary", False)

            if primary:
                # Put primary first (after the "all" option)
                calendars.insert(1, {"id": cal_id, "name": f"{summary} (Primary)"})
            else:
                calendars.append({"id": cal_id, "name": summary})

        return jsonify({"calendars": calendars})

    @admin.route("/api/oauth/google/<int:account_id>/tasklists", methods=["GET"])
    @require_auth_api
    def list_google_tasklists(account_id):
        """
        List task lists from a Google Tasks account using direct API.

        Returns a list of task lists the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        # Try the access token first
        response = http_requests.get(
            "https://tasks.googleapis.com/tasks/v1/users/@me/lists",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                # Update stored token
                from db.oauth_tokens import update_oauth_token_data

                update_oauth_token_data(account_id, {"access_token": access_token})

                # Retry the request
                response = http_requests.get(
                    "https://tasks.googleapis.com/tasks/v1/users/@me/lists",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list task lists: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list task lists: {response.status_code}"}
            ), 500

        data = response.json()
        tasklists = []

        for tasklist in data.get("items", []):
            tasklist_id = tasklist.get("id", "")
            title = tasklist.get("title", tasklist_id)
            tasklists.append({"id": tasklist_id, "name": title})

        return jsonify({"tasklists": tasklists})

    @admin.route("/api/oauth/google/<int:account_id>/contactgroups", methods=["GET"])
    @require_auth_api
    def list_google_contactgroups(account_id):
        """
        List contact groups from a Google Contacts account using People API.

        Returns a list of contact groups the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        # Try the access token first
        response = http_requests.get(
            "https://people.googleapis.com/v1/contactGroups",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"groupFields": "name,memberCount"},
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                # Update stored token
                from db.oauth_tokens import update_oauth_token_data

                update_oauth_token_data(account_id, {"access_token": access_token})

                # Retry the request
                response = http_requests.get(
                    "https://people.googleapis.com/v1/contactGroups",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params={"groupFields": "name,memberCount"},
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list contact groups: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list contact groups: {response.status_code}"}
            ), 500

        data = response.json()
        groups = []

        for group in data.get("contactGroups", []):
            resource_name = group.get("resourceName", "")
            name = group.get("name", resource_name)
            member_count = group.get("memberCount", 0)
            group_type = group.get("groupType", "")

            # Skip system groups except "myContacts" (all contacts with data)
            if group_type == "SYSTEM_CONTACT_GROUP" and name not in [
                "myContacts",
                "starred",
            ]:
                continue

            # Make names more user-friendly
            display_name = name
            if name == "myContacts":
                display_name = "All Contacts"
            elif name == "starred":
                display_name = "Starred"

            groups.append(
                {
                    "id": resource_name,
                    "name": display_name,
                    "memberCount": member_count,
                }
            )

        return jsonify({"groups": groups})

    # =========================================================================
    # Microsoft OAuth
    # =========================================================================

    # Microsoft Graph API scopes for different services
    # Personal accounts use "consumers" endpoint, work/school use "common" or tenant ID
    # Write scopes are included for Smart Actions (draft creation, event creation, etc.)
    MICROSOFT_SCOPES = {
        "onedrive": [
            "User.Read",  # Basic profile info
            "Files.Read",  # OneDrive files (read-only)
            "Files.ReadWrite",  # Create/edit files
        ],
        "outlook": [
            "User.Read",
            "Mail.Read",  # Outlook mail (read)
            "Mail.ReadWrite",  # Create drafts
            "Mail.Send",  # Send emails
            "Calendars.Read",  # Outlook calendar (read)
            "Calendars.ReadWrite",  # Create/edit calendar events
        ],
        "onenote": [
            "User.Read",
            "Notes.Read",  # OneNote notebooks (read)
            "Notes.ReadWrite",  # Create/edit notes
        ],
        "tasks": [
            "User.Read",
            "Tasks.Read",  # To Do tasks (read)
            "Tasks.ReadWrite",  # Create/edit tasks
        ],
        "full": [
            "User.Read",
            # OneDrive
            "Files.Read",
            "Files.ReadWrite",
            # Outlook Mail
            "Mail.Read",
            "Mail.ReadWrite",
            "Mail.Send",
            # Calendar
            "Calendars.Read",
            "Calendars.ReadWrite",
            # OneNote
            "Notes.Read",
            "Notes.ReadWrite",
            # Tasks (To Do)
            "Tasks.Read",
            "Tasks.ReadWrite",
            # Contacts
            "Contacts.Read",
            "Contacts.ReadWrite",
            "offline_access",  # Required for refresh tokens
        ],
    }

    @admin.route("/api/oauth/microsoft/start", methods=["GET"])
    @require_auth_api
    def microsoft_oauth_start():
        """
        Start Microsoft OAuth flow.

        Query params:
            - redirect_uri: Where to redirect after auth (default: /document-stores)

        Returns redirect URL to Microsoft consent screen.
        """
        import secrets
        import urllib.parse

        client_id = os.environ.get("MICROSOFT_CLIENT_ID")
        if not client_id:
            return jsonify(
                {
                    "error": "MICROSOFT_CLIENT_ID not configured",
                    "setup_required": True,
                }
            ), 400

        # Request all scopes so one account works for OneDrive, Outlook, OneNote
        scopes = MICROSOFT_SCOPES["full"]

        # Generate state token for CSRF protection
        import json

        from db import set_setting

        state = secrets.token_urlsafe(32)
        oauth_data = {
            "redirect": request.args.get("redirect_uri", "/document-stores"),
            "scopes": scopes,
        }
        set_setting(f"oauth_state_{state}", json.dumps(oauth_data))

        # Build the OAuth URL
        # Use "consumers" for personal Microsoft accounts
        # Use "common" for both personal and work/school accounts
        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.microsoft_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.microsoft_oauth_callback"
            )

        params = {
            "client_id": client_id,
            "redirect_uri": callback_url,
            "scope": " ".join(scopes),
            "response_type": "code",
            "state": state,
            "response_mode": "query",
        }

        # Use "consumers" endpoint for personal accounts only
        auth_url = (
            "https://login.microsoftonline.com/consumers/oauth2/v2.0/authorize?"
            + urllib.parse.urlencode(params)
        )

        return jsonify({"auth_url": auth_url})

    @admin.route("/api/oauth/microsoft/callback", methods=["GET"])
    def microsoft_oauth_callback():
        """
        Handle Microsoft OAuth callback.

        Microsoft redirects here after user consents. We exchange the code for tokens.
        """
        import requests as http_requests

        error = request.args.get("error")
        error_description = request.args.get("error_description", "")
        if error:
            logger.error(f"Microsoft OAuth error: {error} - {error_description}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Microsoft authentication failed: {error_description or error}",
            )

        # Verify state - retrieve from database
        import json

        from db import delete_setting, get_setting

        state = request.args.get("state")
        oauth_data_json = get_setting(f"oauth_state_{state}") if state else None

        if not state or not oauth_data_json:
            logger.error("OAuth state mismatch or not found")
            return render_template(
                "oauth_result.html",
                success=False,
                error="Invalid OAuth state. Please try again.",
            )

        # Parse stored OAuth data and clean up
        oauth_data = json.loads(oauth_data_json)
        delete_setting(f"oauth_state_{state}")  # One-time use

        code = request.args.get("code")
        if not code:
            return render_template(
                "oauth_result.html",
                success=False,
                error="No authorization code received.",
            )

        # Exchange code for tokens
        client_id = os.environ.get("MICROSOFT_CLIENT_ID")
        client_secret = os.environ.get("MICROSOFT_CLIENT_SECRET")

        # Use EXTERNAL_URL if set (for reverse proxy setups)
        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.microsoft_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.microsoft_oauth_callback"
            )

        try:
            token_response = http_requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": callback_url,
                    "grant_type": "authorization_code",
                },
                timeout=30,
            )
            token_response.raise_for_status()
            token_data = token_response.json()
        except Exception as e:
            logger.error(f"Failed to exchange Microsoft OAuth code: {e}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Failed to get tokens: {e}",
            )

        if "refresh_token" not in token_data:
            logger.warning(
                "No refresh_token in response - ensure offline_access scope is requested"
            )

        # Get user info from Microsoft Graph API
        try:
            userinfo_response = http_requests.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
                timeout=30,
            )
            userinfo_response.raise_for_status()
            userinfo = userinfo_response.json()
            account_email = userinfo.get("mail") or userinfo.get(
                "userPrincipalName", "unknown"
            )
            account_name = userinfo.get("displayName")
        except Exception as e:
            logger.warning(f"Failed to get Microsoft user info: {e}")
            account_email = "unknown"
            account_name = None

        # Store the tokens
        from db.oauth_tokens import store_oauth_token

        scopes = oauth_data.get("scopes", [])

        # Add client credentials to token data for later refresh
        token_data["client_id"] = client_id
        token_data["client_secret"] = client_secret
        token_data["token_uri"] = (
            "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
        )

        token_id = store_oauth_token(
            provider="microsoft",
            account_email=account_email,
            token_data=token_data,
            scopes=scopes,
            account_name=account_name,
        )

        redirect_url = oauth_data.get("redirect", "/document-stores")

        return render_template(
            "oauth_result.html",
            success=True,
            account_id=token_id,
            account_email=account_email,
            account_name=account_name,
            redirect_url=redirect_url,
            provider="Microsoft",
        )

    @admin.route("/api/oauth/microsoft/status", methods=["GET"])
    @require_auth_api
    def microsoft_oauth_status():
        """
        Check Microsoft OAuth configuration status.

        Returns whether client credentials are configured and list of connected accounts.
        """
        from db.oauth_tokens import list_oauth_tokens

        client_id = os.environ.get("MICROSOFT_CLIENT_ID")
        client_secret = os.environ.get("MICROSOFT_CLIENT_SECRET")

        configured = bool(client_id and client_secret)
        accounts = list_oauth_tokens(provider="microsoft") if configured else []

        return jsonify(
            {
                "configured": configured,
                "accounts": accounts,
            }
        )

    @admin.route("/api/oauth/microsoft/<int:account_id>/folders", methods=["GET"])
    @require_auth_api
    def list_onedrive_folders(account_id):
        """
        List folders from a OneDrive account using the Graph API.

        Query params:
            parent: Optional parent folder ID to list subfolders of.
                    If not provided or 'root', lists root-level folders.

        Returns a list of folders the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        parent_id = request.args.get("parent", "root") or "root"

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        # Build the Graph API URL for listing folder children
        if parent_id == "root":
            url = "https://graph.microsoft.com/v1.0/me/drive/root/children"
        else:
            url = (
                f"https://graph.microsoft.com/v1.0/me/drive/items/{parent_id}/children"
            )

        # Filter to only folders
        params = {"$filter": "folder ne null", "$select": "id,name,folder"}

        response = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Microsoft access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                # Update stored token (Microsoft may return a new refresh token)
                token_data["access_token"] = access_token
                if new_tokens.get("refresh_token"):
                    token_data["refresh_token"] = new_tokens["refresh_token"]
                update_oauth_token_data(account_id, token_data)

                # Retry the request
                response = http_requests.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list OneDrive folders: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list folders: {response.status_code}"}
            ), 500

        data = response.json()

        # Build folder list - include option for root/all
        folders = [{"id": "", "name": "All files (root)"}]

        for item in data.get("value", []):
            if "folder" in item:
                folders.append({"id": item["id"], "name": item["name"]})

        return jsonify({"folders": folders, "parent": parent_id})

    @admin.route("/api/oauth/microsoft/<int:account_id>/mail-folders", methods=["GET"])
    @require_auth_api
    def list_outlook_folders(account_id):
        """
        List mail folders from an Outlook account using the Graph API.

        Returns a list of mail folders the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        url = "https://graph.microsoft.com/v1.0/me/mailFolders"
        params = {"$select": "id,displayName,totalItemCount"}

        response = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Microsoft access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                token_data["access_token"] = access_token
                if new_tokens.get("refresh_token"):
                    token_data["refresh_token"] = new_tokens["refresh_token"]
                update_oauth_token_data(account_id, token_data)

                response = http_requests.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list Outlook folders: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list folders: {response.status_code}"}
            ), 500

        data = response.json()

        folders = []
        for item in data.get("value", []):
            folders.append(
                {
                    "id": item["id"],
                    "name": f"{item['displayName']} ({item.get('totalItemCount', 0)})",
                }
            )

        return jsonify({"folders": folders})

    @admin.route("/api/oauth/microsoft/<int:account_id>/notebooks", methods=["GET"])
    @require_auth_api
    def list_onenote_notebooks(account_id):
        """
        List OneNote notebooks from a Microsoft account using the Graph API.

        Returns a list of notebooks the user can select to limit indexing scope.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        url = "https://graph.microsoft.com/v1.0/me/onenote/notebooks"
        params = {"$select": "id,displayName,createdDateTime"}

        response = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Microsoft access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                token_data["access_token"] = access_token
                if new_tokens.get("refresh_token"):
                    token_data["refresh_token"] = new_tokens["refresh_token"]
                update_oauth_token_data(account_id, token_data)

                response = http_requests.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list OneNote notebooks: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list notebooks: {response.status_code}"}
            ), 500

        data = response.json()

        notebooks = []
        for item in data.get("value", []):
            notebooks.append({"id": item["id"], "name": item["displayName"]})

        return jsonify({"notebooks": notebooks})

    @admin.route("/api/oauth/microsoft/<int:account_id>/teams", methods=["GET"])
    @require_auth_api
    def list_microsoft_teams(account_id):
        """
        List Teams from a Microsoft account using the Graph API.

        Returns a list of teams the user is a member of.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        url = "https://graph.microsoft.com/v1.0/me/joinedTeams"
        params = {"$select": "id,displayName,description"}

        response = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Microsoft access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                token_data["access_token"] = access_token
                if new_tokens.get("refresh_token"):
                    token_data["refresh_token"] = new_tokens["refresh_token"]
                update_oauth_token_data(account_id, token_data)

                response = http_requests.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list Teams: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list teams: {response.status_code}"}
            ), 500

        data = response.json()

        teams = []
        for item in data.get("value", []):
            teams.append({"id": item["id"], "name": item["displayName"]})

        return jsonify({"teams": teams})

    @admin.route(
        "/api/oauth/microsoft/<int:account_id>/teams/<team_id>/channels",
        methods=["GET"],
    )
    @require_auth_api
    def list_teams_channels(account_id, team_id):
        """
        List channels from a Teams team using the Graph API.

        Returns a list of channels in the specified team.
        """
        import requests as http_requests

        from db.oauth_tokens import get_oauth_token_by_id, update_oauth_token_data

        token_data = get_oauth_token_by_id(account_id)
        if not token_data:
            return jsonify({"error": "OAuth token not found"}), 404

        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        client_id = token_data.get("client_id")
        client_secret = token_data.get("client_secret")

        if not access_token:
            return jsonify({"error": "No access token in stored credentials"}), 400

        url = f"https://graph.microsoft.com/v1.0/teams/{team_id}/channels"
        params = {"$select": "id,displayName,description"}

        response = http_requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=10,
        )

        # Token expired, try to refresh
        if (
            response.status_code == 401
            and refresh_token
            and client_id
            and client_secret
        ):
            logger.info("Microsoft access token expired, refreshing...")
            refresh_response = http_requests.post(
                "https://login.microsoftonline.com/consumers/oauth2/v2.0/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=10,
            )

            if refresh_response.status_code == 200:
                new_tokens = refresh_response.json()
                access_token = new_tokens.get("access_token")

                token_data["access_token"] = access_token
                if new_tokens.get("refresh_token"):
                    token_data["refresh_token"] = new_tokens["refresh_token"]
                update_oauth_token_data(account_id, token_data)

                response = http_requests.get(
                    url,
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params,
                    timeout=10,
                )

        if response.status_code != 200:
            logger.error(
                f"Failed to list Teams channels: {response.status_code} {response.text}"
            )
            return jsonify(
                {"error": f"Failed to list channels: {response.status_code}"}
            ), 500

        data = response.json()

        channels = []
        for item in data.get("value", []):
            channels.append({"id": item["id"], "name": item["displayName"]})

        return jsonify({"channels": channels})

    # =========================================================================
    # Oura OAuth
    # =========================================================================

    # Oura OAuth scopes
    # See: https://cloud.ouraring.com/docs/authentication
    OURA_SCOPES = [
        "email",  # User's email address
        "personal",  # Gender, age, height, weight
        "daily",  # Sleep, activity, and readiness summaries
        "heartrate",  # Heart rate time series data
        "workout",  # Workout summaries
        "session",  # Guided and unguided sessions
        "spo2",  # SpO2 data
    ]

    @admin.route("/api/oauth/oura/start", methods=["GET"])
    @require_auth_api
    def oura_oauth_start():
        """
        Start Oura OAuth flow.

        Query params:
            - redirect_uri: Where to redirect after auth (default: /live-data-sources)

        Returns redirect URL to Oura consent screen.
        """
        import secrets
        import urllib.parse

        client_id = os.environ.get("OURA_CLIENT_ID")
        if not client_id:
            return jsonify(
                {
                    "error": "OURA_CLIENT_ID not configured",
                    "setup_required": True,
                }
            ), 400

        # Generate state token for CSRF protection
        import json

        from db import set_setting

        state = secrets.token_urlsafe(32)
        oauth_data = {
            "redirect": request.args.get("redirect_uri", "/live-data-sources"),
            "scopes": OURA_SCOPES,
        }
        set_setting(f"oauth_state_{state}", json.dumps(oauth_data))

        # Build the OAuth URL
        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.oura_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.oura_oauth_callback"
            )

        params = {
            "client_id": client_id,
            "redirect_uri": callback_url,
            "scope": " ".join(OURA_SCOPES),
            "response_type": "code",
            "state": state,
        }

        auth_url = (
            "https://cloud.ouraring.com/oauth/authorize?"
            + urllib.parse.urlencode(params)
        )

        return jsonify({"auth_url": auth_url})

    @admin.route("/api/oauth/oura/callback", methods=["GET"])
    def oura_oauth_callback():
        """
        Handle Oura OAuth callback.

        Oura redirects here after user consents. We exchange the code for tokens.
        """
        import requests as http_requests

        error = request.args.get("error")
        if error:
            logger.error(f"Oura OAuth error: {error}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Oura authentication failed: {error}",
            )

        # Verify state
        import json

        from db import delete_setting, get_setting

        state = request.args.get("state")
        oauth_data_json = get_setting(f"oauth_state_{state}") if state else None

        if not state or not oauth_data_json:
            logger.error("OAuth state mismatch or not found")
            return render_template(
                "oauth_result.html",
                success=False,
                error="Invalid OAuth state. Please try again.",
            )

        # Parse stored OAuth data and clean up
        oauth_data = json.loads(oauth_data_json)
        delete_setting(f"oauth_state_{state}")

        code = request.args.get("code")
        if not code:
            return render_template(
                "oauth_result.html",
                success=False,
                error="No authorization code received.",
            )

        # Exchange code for tokens
        client_id = os.environ.get("OURA_CLIENT_ID")
        client_secret = os.environ.get("OURA_CLIENT_SECRET")

        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.oura_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.oura_oauth_callback"
            )

        try:
            token_response = http_requests.post(
                "https://api.ouraring.com/oauth/token",
                data={
                    "code": code,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "redirect_uri": callback_url,
                    "grant_type": "authorization_code",
                },
                timeout=30,
            )
            token_response.raise_for_status()
            token_data = token_response.json()
        except Exception as e:
            logger.error(f"Failed to exchange Oura OAuth code: {e}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Failed to get tokens: {e}",
            )

        # Get user info using the personal endpoint
        try:
            userinfo_response = http_requests.get(
                "https://api.ouraring.com/v2/usercollection/personal_info",
                headers={"Authorization": f"Bearer {token_data['access_token']}"},
                timeout=30,
            )
            userinfo_response.raise_for_status()
            userinfo = userinfo_response.json()
            # Oura personal_info returns age, weight, height, biological_sex, email
            account_email = userinfo.get("email", "oura-user")
            # Use email as name since Oura doesn't return a display name
            account_name = (
                account_email.split("@")[0] if "@" in account_email else account_email
            )
        except Exception as e:
            logger.warning(f"Failed to get Oura user info: {e}")
            account_email = "oura-user"
            account_name = "Oura User"

        # Store the tokens
        from db.oauth_tokens import store_oauth_token

        scopes = oauth_data.get("scopes", [])

        # Add client credentials to token data for later refresh
        token_data["client_id"] = client_id
        token_data["client_secret"] = client_secret

        token_id = store_oauth_token(
            provider="oura",
            account_email=account_email,
            token_data=token_data,
            scopes=scopes,
            account_name=account_name,
        )

        redirect_url = oauth_data.get("redirect", "/live-data-sources")

        return render_template(
            "oauth_result.html",
            success=True,
            account_id=token_id,
            account_email=account_email,
            account_name=account_name,
            redirect_url=redirect_url,
            provider="Oura",
        )

    @admin.route("/api/oauth/oura/status", methods=["GET"])
    @require_auth_api
    def oura_oauth_status():
        """Check if Oura OAuth is configured and list connected accounts."""
        from db.oauth_tokens import list_oauth_tokens

        client_id = os.environ.get("OURA_CLIENT_ID")
        client_secret = os.environ.get("OURA_CLIENT_SECRET")

        configured = bool(client_id and client_secret)
        accounts = list_oauth_tokens(provider="oura") if configured else []

        return jsonify(
            {
                "configured": configured,
                "accounts": accounts,
            }
        )

    # =========================================================================
    # Withings OAuth
    # =========================================================================

    # Withings OAuth scopes
    # See: https://developer.withings.com/developer-guide/v3/integration-guide/public-health-data-api/get-access/oauth-authorization-url/
    WITHINGS_SCOPES = [
        "user.info",  # User profile info
        "user.metrics",  # Body measurements (weight, body composition)
        "user.activity",  # Activity and sleep data
    ]

    @admin.route("/api/oauth/withings/start", methods=["GET"])
    @require_auth_api
    def withings_oauth_start():
        """
        Start Withings OAuth flow.

        Query params:
            - redirect_uri: Where to redirect after auth (default: /live-data-sources)

        Returns redirect URL to Withings consent screen.
        """
        import secrets
        import urllib.parse

        client_id = os.environ.get("WITHINGS_CLIENT_ID")
        if not client_id:
            return jsonify(
                {
                    "error": "WITHINGS_CLIENT_ID not configured",
                    "setup_required": True,
                }
            ), 400

        # Generate state token for CSRF protection
        import json

        from db import set_setting

        state = secrets.token_urlsafe(32)
        oauth_data = {
            "redirect": request.args.get("redirect_uri", "/live-data-sources"),
            "scopes": WITHINGS_SCOPES,
        }
        set_setting(f"oauth_state_{state}", json.dumps(oauth_data))

        # Build the OAuth URL
        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.withings_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.withings_oauth_callback"
            )

        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": callback_url,
            "scope": ",".join(WITHINGS_SCOPES),
            "state": state,
        }

        auth_url = (
            "https://account.withings.com/oauth2_user/authorize2?"
            + urllib.parse.urlencode(params)
        )

        return jsonify({"auth_url": auth_url})

    @admin.route("/api/oauth/withings/callback", methods=["GET"])
    def withings_oauth_callback():
        """
        Handle Withings OAuth callback.

        Withings redirects here after user consents. We exchange the code for tokens.
        """
        import hashlib
        import time

        import requests as http_requests

        error = request.args.get("error")
        if error:
            logger.error(f"Withings OAuth error: {error}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Withings authentication failed: {error}",
            )

        # Verify state
        import json

        from db import delete_setting, get_setting

        state = request.args.get("state")
        oauth_data_json = get_setting(f"oauth_state_{state}") if state else None

        if not state or not oauth_data_json:
            logger.error("OAuth state mismatch or not found")
            return render_template(
                "oauth_result.html",
                success=False,
                error="Invalid OAuth state. Please try again.",
            )

        # Parse stored OAuth data and clean up
        oauth_data = json.loads(oauth_data_json)
        delete_setting(f"oauth_state_{state}")

        code = request.args.get("code")
        if not code:
            return render_template(
                "oauth_result.html",
                success=False,
                error="No authorization code received.",
            )

        # Exchange code for tokens
        client_id = os.environ.get("WITHINGS_CLIENT_ID")
        client_secret = os.environ.get("WITHINGS_CLIENT_SECRET")

        external_url = os.environ.get("EXTERNAL_URL")
        if external_url:
            callback_url = external_url.rstrip("/") + url_for(
                "admin.withings_oauth_callback"
            )
        else:
            callback_url = request.url_root.rstrip("/") + url_for(
                "admin.withings_oauth_callback"
            )

        try:
            # Withings OAuth token exchange - no signature needed for this step
            token_response = http_requests.post(
                "https://wbsapi.withings.net/v2/oauth2",
                data={
                    "action": "requesttoken",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": callback_url,
                },
                timeout=30,
            )
            token_response.raise_for_status()
            response_data = token_response.json()

            # Withings wraps the response in a "body" object
            if response_data.get("status") != 0:
                error_msg = response_data.get("error", "Unknown error")
                logger.error(f"Withings token error: {error_msg}")
                return render_template(
                    "oauth_result.html",
                    success=False,
                    error=f"Withings token error: {error_msg}",
                )

            token_data = response_data.get("body", {})
        except Exception as e:
            logger.error(f"Failed to exchange Withings OAuth code: {e}")
            return render_template(
                "oauth_result.html",
                success=False,
                error=f"Failed to get tokens: {e}",
            )

        # Extract user ID from response
        user_id = token_data.get("userid", "withings-user")
        account_email = f"user-{user_id}@withings"
        account_name = f"Withings User {user_id}"

        # Store the tokens
        from db.oauth_tokens import store_oauth_token

        scopes = oauth_data.get("scopes", [])

        # Add client credentials to token data for later refresh
        token_data["client_id"] = client_id
        token_data["client_secret"] = client_secret

        token_id = store_oauth_token(
            provider="withings",
            account_email=account_email,
            token_data=token_data,
            scopes=scopes,
            account_name=account_name,
        )

        redirect_url = oauth_data.get("redirect", "/live-data-sources")

        return render_template(
            "oauth_result.html",
            success=True,
            account_id=token_id,
            account_email=account_email,
            account_name=account_name,
            redirect_url=redirect_url,
            provider="Withings",
        )

    @admin.route("/api/oauth/withings/status", methods=["GET"])
    @require_auth_api
    def withings_oauth_status():
        """Check if Withings OAuth is configured and list connected accounts."""
        from db.oauth_tokens import list_oauth_tokens

        client_id = os.environ.get("WITHINGS_CLIENT_ID")
        client_secret = os.environ.get("WITHINGS_CLIENT_SECRET")

        configured = bool(client_id and client_secret)
        accounts = list_oauth_tokens(provider="withings") if configured else []

        return jsonify(
            {
                "configured": configured,
                "accounts": accounts,
            }
        )

    # =========================================================================
    # Smart Enrichers (unified RAG + Web)
    # =========================================================================

    # ========================================================================
    # Smart Aliases (unified routing + enrichment + caching)
    # ========================================================================

    @admin.route("/smart-aliases")
    @require_auth
    def smart_aliases_page():
        """Smart Aliases management page."""
        return render_template("smart_aliases.html")

    @admin.route("/api/smart-aliases/models", methods=["GET"])
    @require_auth_api
    def list_available_models_for_smart_alias():
        """List all available models that can be used as targets/candidates."""
        models = []

        # Only provider models - no aliases/routers to avoid circular routing
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

        # Sort by provider, then model_id
        models.sort(key=lambda m: (m["provider"], m["model_id"]))
        return jsonify(models)

    @admin.route("/api/smart-aliases/stores", methods=["GET"])
    @require_auth_api
    def list_document_stores_for_smart_alias():
        """List all document stores for RAG configuration."""
        from db import get_all_document_stores

        stores = get_all_document_stores()
        return jsonify([s.to_dict() for s in stores])

    @admin.route("/api/smart-aliases", methods=["GET"])
    @require_auth_api
    def list_smart_aliases():
        """List all smart aliases."""
        from db import get_all_smart_aliases

        aliases = get_all_smart_aliases()
        return jsonify([a.to_dict() for a in aliases])

    @admin.route("/api/smart-aliases/<int:alias_id>", methods=["GET"])
    @require_auth_api
    def get_smart_alias(alias_id: int):
        """Get a specific smart alias."""
        from db import get_smart_alias_by_id

        alias = get_smart_alias_by_id(alias_id)
        if not alias:
            return jsonify({"error": "Smart alias not found"}), 404
        return jsonify(alias.to_dict())

    @admin.route("/api/smart-aliases", methods=["POST"])
    @require_auth_api
    def create_smart_alias_endpoint():
        """Create a new smart alias."""
        from db import create_smart_alias

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        name = data.get("name", "").strip()
        if not name:
            return jsonify({"error": "Name is required"}), 400

        target_model = data.get("target_model", "").strip()
        if not target_model:
            return jsonify({"error": "Target model is required"}), 400

        # Feature toggles
        use_routing = data.get("use_routing", False)
        use_rag = data.get("use_rag", False)
        use_web = data.get("use_web", False)
        use_cache = data.get("use_cache", False)
        use_smart_source_selection = data.get("use_smart_source_selection", False)
        use_two_pass_retrieval = data.get("use_two_pass_retrieval", False)

        # Validate: can't have cache + web
        if use_cache and use_web:
            return jsonify(
                {"error": "Cannot enable caching with realtime web search"}
            ), 400

        # Validate routing requirements
        if use_routing:
            if not data.get("designator_model"):
                return jsonify(
                    {"error": "Designator model is required for routing"}
                ), 400
            if not data.get("candidates"):
                return jsonify(
                    {"error": "Candidate models are required for routing"}
                ), 400

        # Validate smart source selection requirements
        if use_smart_source_selection:
            if not data.get("designator_model"):
                return jsonify(
                    {"error": "Designator model is required for smart source selection"}
                ), 400

        # Validate RAG requirements
        store_ids = data.get("document_store_ids") or data.get("store_ids")
        if use_rag and not store_ids:
            return jsonify(
                {"error": "Document stores are required when RAG is enabled"}
            ), 400

        try:
            alias = create_smart_alias(
                name=name,
                target_model=target_model,
                # Feature toggles
                use_routing=use_routing,
                use_rag=use_rag,
                use_web=use_web,
                use_cache=use_cache,
                use_smart_source_selection=use_smart_source_selection,
                use_two_pass_retrieval=use_two_pass_retrieval,
                # Smart tag settings
                is_smart_tag=data.get("is_smart_tag", False),
                passthrough_model=data.get("passthrough_model", False),
                # Routing settings
                designator_model=data.get("designator_model"),
                purpose=data.get("purpose"),
                candidates=data.get("candidates"),
                fallback_model=data.get("fallback_model"),
                routing_strategy=data.get("routing_strategy", "per_request"),
                session_ttl=data.get("session_ttl", 3600),
                use_model_intelligence=data.get("use_model_intelligence", False),
                search_provider=data.get("search_provider"),
                intelligence_model=data.get("intelligence_model"),
                # RAG settings
                store_ids=store_ids,
                max_results=data.get("max_results", 5),
                similarity_threshold=data.get("similarity_threshold", 0.7),
                # Web settings
                max_search_results=data.get("max_search_results", 5),
                max_scrape_urls=data.get("max_scrape_urls", 3),
                # Live data settings
                use_live_data=data.get("use_live_data", False),
                live_data_source_ids=data.get("live_data_source_ids"),
                # Common enrichment
                max_context_tokens=data.get("max_context_tokens", 4000),
                rerank_provider=data.get("rerank_provider", "local"),
                rerank_model=data.get("rerank_model"),
                rerank_top_n=data.get("rerank_top_n", 20),
                context_priority=data.get("context_priority", "balanced"),
                show_sources=data.get("show_sources", False),
                # Cache settings
                cache_similarity_threshold=data.get("cache_similarity_threshold", 0.95),
                cache_match_system_prompt=data.get("cache_match_system_prompt", True),
                cache_match_last_message_only=data.get(
                    "cache_match_last_message_only", False
                ),
                cache_ttl_hours=data.get("cache_ttl_hours", 168),
                cache_min_tokens=data.get("cache_min_tokens", 50),
                cache_max_tokens=data.get("cache_max_tokens", 4000),
                cache_collection=data.get("cache_collection"),
                # Metadata
                tags=data.get("tags", []),
                description=data.get("description"),
                system_prompt=data.get("system_prompt"),
                enabled=data.get("enabled", True),
                # Memory
                use_memory=data.get("use_memory", False),
                memory_max_tokens=data.get("memory_max_tokens", 500),
                # Actions
                use_actions=data.get("use_actions", False),
                allowed_actions=data.get("allowed_actions", []),
                action_email_account_id=data.get("action_email_account_id"),
                action_calendar_account_id=data.get("action_calendar_account_id"),
                action_calendar_id=data.get("action_calendar_id"),
                action_tasks_account_id=data.get("action_tasks_account_id"),
                action_tasks_provider=data.get("action_tasks_provider"),
                action_tasks_list_id=data.get("action_tasks_list_id"),
                action_notification_urls=data.get("action_notification_urls"),
                # Scheduled Prompts
                scheduled_prompts_enabled=data.get("scheduled_prompts_enabled", False),
                scheduled_prompts_account_id=data.get("scheduled_prompts_account_id"),
                scheduled_prompts_calendar_id=data.get("scheduled_prompts_calendar_id"),
                scheduled_prompts_calendar_name=data.get(
                    "scheduled_prompts_calendar_name"
                ),
                scheduled_prompts_lookahead=data.get("scheduled_prompts_lookahead", 15),
                scheduled_prompts_store_response=data.get(
                    "scheduled_prompts_store_response", False
                ),
            )
            return jsonify(alias.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/smart-aliases/<int:alias_id>", methods=["PUT"])
    @require_auth_api
    def update_smart_alias_endpoint(alias_id: int):
        """Update a smart alias."""
        from db import update_smart_alias

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Document store IDs
        store_ids = data.get("document_store_ids") or data.get("store_ids")

        try:
            alias = update_smart_alias(
                alias_id=alias_id,
                name=data.get("name"),
                target_model=data.get("target_model"),
                # Feature toggles
                use_routing=data.get("use_routing"),
                use_rag=data.get("use_rag"),
                use_web=data.get("use_web"),
                use_cache=data.get("use_cache"),
                use_smart_source_selection=data.get("use_smart_source_selection"),
                use_two_pass_retrieval=data.get("use_two_pass_retrieval"),
                # Smart tag settings
                is_smart_tag=data.get("is_smart_tag"),
                passthrough_model=data.get("passthrough_model"),
                # Routing settings
                designator_model=data.get("designator_model"),
                purpose=data.get("purpose"),
                candidates=data.get("candidates"),
                fallback_model=data.get("fallback_model"),
                routing_strategy=data.get("routing_strategy"),
                session_ttl=data.get("session_ttl"),
                use_model_intelligence=data.get("use_model_intelligence"),
                search_provider=data.get("search_provider"),
                intelligence_model=data.get("intelligence_model"),
                # RAG settings
                store_ids=store_ids,
                max_results=data.get("max_results"),
                similarity_threshold=data.get("similarity_threshold"),
                # Web settings
                max_search_results=data.get("max_search_results"),
                max_scrape_urls=data.get("max_scrape_urls"),
                # Live data settings
                use_live_data=data.get("use_live_data"),
                live_data_source_ids=data.get("live_data_source_ids"),
                # Common enrichment
                max_context_tokens=data.get("max_context_tokens"),
                rerank_provider=data.get("rerank_provider"),
                rerank_model=data.get("rerank_model"),
                rerank_top_n=data.get("rerank_top_n"),
                context_priority=data.get("context_priority"),
                show_sources=data.get("show_sources"),
                # Cache settings
                cache_similarity_threshold=data.get("cache_similarity_threshold"),
                cache_match_system_prompt=data.get("cache_match_system_prompt"),
                cache_match_last_message_only=data.get("cache_match_last_message_only"),
                cache_ttl_hours=data.get("cache_ttl_hours"),
                cache_min_tokens=data.get("cache_min_tokens"),
                cache_max_tokens=data.get("cache_max_tokens"),
                cache_collection=data.get("cache_collection"),
                # Metadata
                tags=data.get("tags"),
                description=data.get("description"),
                system_prompt=data.get("system_prompt"),
                enabled=data.get("enabled"),
                # Memory
                use_memory=data.get("use_memory"),
                memory=data.get("memory"),
                memory_max_tokens=data.get("memory_max_tokens"),
                # Actions
                use_actions=data.get("use_actions"),
                allowed_actions=data.get("allowed_actions"),
                action_email_account_id=data.get("action_email_account_id"),
                action_calendar_account_id=data.get("action_calendar_account_id"),
                action_calendar_id=data.get("action_calendar_id"),
                action_tasks_account_id=data.get("action_tasks_account_id"),
                action_tasks_provider=data.get("action_tasks_provider"),
                action_tasks_list_id=data.get("action_tasks_list_id"),
                action_notification_urls=data.get("action_notification_urls"),
                # Scheduled Prompts
                scheduled_prompts_enabled=data.get("scheduled_prompts_enabled"),
                scheduled_prompts_account_id=data.get("scheduled_prompts_account_id"),
                scheduled_prompts_calendar_id=data.get("scheduled_prompts_calendar_id"),
                scheduled_prompts_calendar_name=data.get(
                    "scheduled_prompts_calendar_name"
                ),
                scheduled_prompts_lookahead=data.get("scheduled_prompts_lookahead"),
                scheduled_prompts_store_response=data.get(
                    "scheduled_prompts_store_response"
                ),
            )
            if not alias:
                return jsonify({"error": "Smart alias not found"}), 404
            return jsonify(alias.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @admin.route("/api/smart-aliases/<int:alias_id>", methods=["DELETE"])
    @require_auth_api
    def delete_smart_alias_endpoint(alias_id: int):
        """Delete a smart alias."""
        from db import delete_smart_alias

        if delete_smart_alias(alias_id):
            return jsonify({"success": True})
        return jsonify({"error": "Failed to delete smart alias"}), 500

    @admin.route("/api/smart-aliases/<int:alias_id>/reset-stats", methods=["POST"])
    @require_auth_api
    def reset_smart_alias_stats_endpoint(alias_id: int):
        """Reset statistics for a smart alias."""
        from db import reset_smart_alias_stats

        if reset_smart_alias_stats(alias_id):
            return jsonify({"success": True})
        return jsonify({"error": "Smart alias not found"}), 404

    @admin.route("/api/smart-aliases/stats", methods=["GET"])
    @require_auth_api
    def get_smart_alias_routing_stats():
        """Get per-candidate routing statistics from request logs."""
        from sqlalchemy import func

        from db import get_db_context
        from db.models import RequestLog

        stats = {}
        try:
            with get_db_context() as db:
                # Get routing stats grouped by router_name, provider_id, and model_id
                results = (
                    db.query(
                        RequestLog.router_name,
                        RequestLog.provider_id,
                        RequestLog.model_id,
                        func.count().label("count"),
                    )
                    .filter(RequestLog.router_name.isnot(None))
                    .filter(RequestLog.router_name != "")
                    .group_by(
                        RequestLog.router_name,
                        RequestLog.provider_id,
                        RequestLog.model_id,
                    )
                    .all()
                )

                for router_name, provider_id, model_id, count in results:
                    if router_name not in stats:
                        stats[router_name] = {"requests": 0, "candidates": {}}
                    # Use full path format: provider/model (e.g., "anthropic/claude-haiku-4-5")
                    full_model = f"{provider_id}/{model_id}"
                    stats[router_name]["candidates"][full_model] = count
                    stats[router_name]["requests"] += count

        except Exception as e:
            logger.error(f"Error getting routing stats: {e}")

        return jsonify(stats)

    # -------------------------------------------------------------------------
    # Scheduled Prompts Routes
    # -------------------------------------------------------------------------

    @admin.route("/api/scheduled-prompts", methods=["GET"])
    @require_auth_api
    def list_scheduled_prompts():
        """List scheduled prompt executions."""
        from db.scheduled_prompts import get_executions_for_alias, get_recent_executions

        alias_id = request.args.get("alias_id", type=int)
        limit = request.args.get("limit", default=50, type=int)

        if alias_id:
            executions = get_executions_for_alias(alias_id, limit=limit)
        else:
            executions = get_recent_executions(limit=limit)

        return jsonify([e.to_dict() for e in executions])

    @admin.route("/api/scheduled-prompts/stats", methods=["GET"])
    @require_auth_api
    def get_scheduled_prompt_stats():
        """Get scheduled prompt execution statistics."""
        from db.scheduled_prompts import get_execution_stats

        alias_id = request.args.get("alias_id", type=int)
        stats = get_execution_stats(smart_alias_id=alias_id)
        return jsonify(stats)

    @admin.route("/api/scheduled-prompts/<int:execution_id>", methods=["GET"])
    @require_auth_api
    def get_scheduled_prompt(execution_id: int):
        """Get a specific scheduled prompt execution."""
        from db.scheduled_prompts import get_execution_by_id

        execution = get_execution_by_id(execution_id)
        if not execution:
            return jsonify({"error": "Execution not found"}), 404
        return jsonify(execution.to_dict())

    @admin.route("/api/scheduled-prompts/<int:execution_id>/retry", methods=["POST"])
    @require_auth_api
    def retry_scheduled_prompt(execution_id: int):
        """Retry a failed or skipped scheduled prompt."""
        from db.scheduled_prompts import get_execution_by_id, reset_execution_to_pending

        execution = get_execution_by_id(execution_id)
        if not execution:
            return jsonify({"error": "Execution not found"}), 404

        if execution.status not in ("failed", "skipped"):
            return jsonify(
                {"error": "Can only retry failed or skipped executions"}
            ), 400

        reset_execution_to_pending(execution_id)
        return jsonify({"success": True})

    @admin.route("/api/scheduled-prompts/<int:execution_id>", methods=["DELETE"])
    @require_auth_api
    def delete_scheduled_prompt(execution_id: int):
        """Delete a scheduled prompt execution record."""
        from db.scheduled_prompts import delete_execution

        if delete_execution(execution_id):
            return jsonify({"success": True})
        return jsonify({"error": "Execution not found"}), 404

    @admin.route("/api/scheduled-prompts/poll", methods=["POST"])
    @require_auth_api
    def trigger_prompt_poll():
        """Manually trigger calendar polling for scheduled prompts."""
        from scheduling import get_prompt_scheduler

        alias_id = request.args.get("alias_id", type=int)

        try:
            scheduler = get_prompt_scheduler()
            scheduler.poll_now(alias_id=alias_id)
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error triggering prompt poll: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/scheduled-prompts/<int:execution_id>/execute", methods=["POST"])
    @require_auth_api
    def execute_scheduled_prompt_now(execution_id: int):
        """Manually execute a pending scheduled prompt."""
        from db.scheduled_prompts import get_execution_by_id
        from scheduling import get_prompt_scheduler

        execution = get_execution_by_id(execution_id)
        if not execution:
            return jsonify({"error": "Execution not found"}), 404

        if execution.status != "pending":
            return jsonify({"error": "Can only execute pending prompts"}), 400

        try:
            scheduler = get_prompt_scheduler()
            scheduler.execute_now(execution_id)
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error executing prompt: {e}")
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------------
    # Live Data Sources Routes
    # -------------------------------------------------------------------------

    @admin.route("/live-data-sources")
    @require_auth
    def live_data_sources_page():
        """Live data sources management page."""
        return render_template("live_data_sources.html")

    @admin.route("/smart-actions")
    @require_auth
    def smart_actions_page():
        """Smart actions (action handlers) management page."""
        return render_template("smart_actions.html")

    @admin.route("/api/smart-actions", methods=["GET"])
    @require_auth_api
    def list_smart_actions():
        """List all available action handlers with OAuth status."""
        from actions import load_action_handlers
        from actions.registry import _handlers as all_handlers
        from db.oauth_tokens import list_oauth_tokens

        # Ensure handlers are loaded
        if not all_handlers:
            load_action_handlers()

        # Get available OAuth accounts
        oauth_accounts = {}
        for provider in ["google", "microsoft"]:
            try:
                tokens = list_oauth_tokens(provider=provider)
                oauth_accounts[provider] = [
                    {
                        "id": t["id"],
                        "account_email": t["account_email"],
                        "account_name": t.get("account_name"),
                    }
                    for t in tokens
                ]
            except Exception:
                oauth_accounts[provider] = []

        handlers = []
        for action_type, handler in all_handlers.items():
            # Determine which OAuth providers this handler can use
            oauth_providers = []
            if handler.requires_oauth:
                if handler.oauth_provider:
                    oauth_providers = [handler.oauth_provider]
                else:
                    # Handler can use multiple providers (e.g., email works with google or microsoft)
                    if action_type in ["email", "calendar"]:
                        oauth_providers = ["google", "microsoft"]
                    elif action_type == "schedule":
                        oauth_providers = ["google"]  # Calendar-based scheduling
                    else:
                        oauth_providers = ["google"]

            # Check if any accounts are available
            available_accounts = []
            for provider in oauth_providers:
                available_accounts.extend(oauth_accounts.get(provider, []))

            handlers.append(
                {
                    "action_type": action_type,
                    "supported_actions": handler.supported_actions,
                    "requires_oauth": handler.requires_oauth,
                    "oauth_providers": oauth_providers,
                    "available_accounts": available_accounts,
                    "is_configured": len(available_accounts) > 0
                    or not handler.requires_oauth,
                    "description": handler.__doc__ or "",
                }
            )

        return jsonify(
            {
                "handlers": handlers,
                "oauth_accounts": oauth_accounts,
            }
        )

    @admin.route("/api/live-data-sources", methods=["GET"])
    @require_auth_api
    def list_live_data_sources():
        """List all live data sources.

        Excludes unified sources (which are shown in Document Stores instead).
        """
        from db import get_all_live_data_sources

        # Get live source types that are handled by unified sources
        # These should only appear in Document Stores, not here
        try:
            from plugin_base.loader import get_live_source_to_unified_map

            unified_live_types = set(get_live_source_to_unified_map().keys())
        except ImportError:
            unified_live_types = set()

        sources = get_all_live_data_sources()
        return jsonify(
            [s.to_dict() for s in sources if s.source_type not in unified_live_types]
        )

    @admin.route("/api/live-data-sources", methods=["POST"])
    @require_auth_api
    def create_live_data_source():
        """Create a new live data source."""
        from db import create_live_data_source as db_create_source

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ["name", "source_type"]
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}"}), 400

        try:
            # Helper to parse JSON strings
            def parse_json(val):
                if val is None:
                    return None
                if isinstance(val, dict):
                    return val
                try:
                    import json as json_mod

                    return json_mod.loads(val) if val else None
                except:
                    return None

            source = db_create_source(
                name=data["name"],
                source_type=data["source_type"],
                description=data.get("description"),
                endpoint_url=data.get("endpoint_url"),
                http_method=data.get("http_method", "GET"),
                headers=parse_json(data.get("headers_json")),
                auth_type=data.get("auth_type", "none"),
                auth_config=parse_json(data.get("auth_config_json")),
                request_template=parse_json(data.get("request_template_json")),
                query_params=parse_json(data.get("query_params_json")),
                response_path=data.get("response_path"),
                response_format_template=data.get("response_format_template"),
                cache_ttl_seconds=data.get("cache_ttl_seconds", 60),
                timeout_seconds=data.get("timeout_seconds", 10),
                retry_count=data.get("retry_count", 1),
                rate_limit_rpm=data.get("rate_limit_rpm", 60),
                data_type=data.get("data_type", "general"),
                best_for=data.get("best_for"),
                enabled=data.get("enabled", True),
                config=parse_json(data.get("config_json")),
            )
            return jsonify(source.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error creating live data source: {e}")
            return jsonify({"error": "Failed to create live data source"}), 500

    @admin.route("/api/live-data-sources/rapidapi-status", methods=["GET"])
    @require_auth_api
    def rapidapi_status():
        """Check if RapidAPI key is configured."""
        import os

        return jsonify({"configured": bool(os.environ.get("RAPIDAPI_KEY"))})

    @admin.route("/api/live-data-sources/env-var-status", methods=["GET"])
    @require_auth_api
    def env_var_status():
        """Return which plugin env vars are configured.

        Used by frontend to hide fields that have env var fallbacks when
        the env var is already set.
        """
        import os

        from plugin_base.loader import live_source_registry

        configured_vars = {}

        # Check all live source plugins for env_var fields
        for source_type, plugin_class in live_source_registry.get_all().items():
            try:
                fields = plugin_class.get_config_fields()
                for field in fields:
                    if field.env_var:
                        # Check if this env var is set
                        configured_vars[field.env_var] = bool(
                            os.environ.get(field.env_var)
                        )
            except Exception as e:
                logger.debug(f"Error getting fields for {source_type}: {e}")

        return jsonify(configured_vars)

    @admin.route("/api/live-data-sources/test-config", methods=["POST"])
    @require_auth_api
    def test_live_data_config():
        """Test a live data source configuration without saving."""
        from dataclasses import dataclass

        from live.sources import MCPProvider

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        logger.info(f"test-config received: {data}")
        source_type = data.get("source_type", "mcp_server")

        # Create a mock source object for testing
        @dataclass
        class MockSource:
            name: str = "test"
            source_type: str = "mcp_server"
            endpoint_url: str = ""
            http_method: str = "GET"
            timeout_seconds: int = 10
            cache_ttl_seconds: int = 0
            response_path: str = ""
            response_format_template: str = ""
            headers_json: str = ""
            auth_type: str = "none"
            auth_config_json: str = ""
            request_template_json: str = ""
            query_params_json: str = ""

        try:
            mock = MockSource(
                source_type=source_type,
                endpoint_url=data.get("endpoint_url", ""),
                http_method=data.get("http_method", "GET"),
                timeout_seconds=data.get("timeout_seconds", 10),
                response_path=data.get("response_path", ""),
                response_format_template=data.get("response_format_template", ""),
                headers_json=data.get("headers_json", ""),
                auth_type=data.get("auth_type", "none"),
                auth_config_json=data.get("auth_config_json", ""),
                request_template_json=data.get("request_template_json", ""),
                query_params_json=data.get("query_params_json", ""),
            )

            # Use MCPProvider for MCP sources
            if source_type == "mcp_server":
                provider = MCPProvider(mock)
                success, message = provider.test_connection()
                if success:
                    # Also report number of tools discovered
                    tools = provider.list_tools()
                    message = f"{message} ({len(tools)} tools discovered)"
                return jsonify({"success": success, "message": message})
            else:
                # For legacy REST API sources, just validate the config
                if not mock.endpoint_url:
                    return jsonify(
                        {"success": False, "message": "Endpoint URL required"}
                    )
                return jsonify(
                    {"success": True, "message": "Configuration valid (not tested)"}
                )
        except Exception as e:
            logger.error(f"Error testing config: {e}")
            return jsonify({"success": False, "message": str(e)})

    @admin.route("/api/live-data-sources/<int:source_id>", methods=["GET"])
    @require_auth_api
    def get_live_data_source(source_id: int):
        """Get a specific live data source."""
        from db import get_live_data_source_by_id

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404
        return jsonify(source.to_dict())

    @admin.route("/api/live-data-sources/<int:source_id>", methods=["PUT"])
    @require_auth_api
    def update_live_data_source(source_id: int):
        """Update a live data source."""
        from db import (
            get_live_data_source_by_id,
        )
        from db import (
            update_live_data_source as db_update_source,
        )

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Check if source exists and if it's a built-in (limited updates)
        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        try:
            # For built-in sources, only allow enabled toggle
            if source.source_type.startswith("builtin_"):
                if "enabled" not in data:
                    return jsonify(
                        {"error": "Only 'enabled' can be updated for built-in sources"}
                    ), 400
                updated = db_update_source(
                    source_id=source_id, enabled=data.get("enabled")
                )
            else:
                # For custom sources, allow full updates
                # Helper to parse JSON strings
                def parse_json(val):
                    if val is None:
                        return None
                    if isinstance(val, dict):
                        return val

                    try:
                        import json as json_mod

                        return json_mod.loads(val) if val else None
                    except:
                        return None

                updated = db_update_source(
                    source_id=source_id,
                    enabled=data.get("enabled", source.enabled),
                    description=data.get("description", source.description),
                    endpoint_url=data.get("endpoint_url", source.endpoint_url),
                    http_method=data.get("http_method", source.http_method),
                    headers=parse_json(data.get("headers_json")),
                    auth_type=data.get("auth_type", source.auth_type),
                    auth_config=parse_json(data.get("auth_config_json")),
                    request_template=parse_json(data.get("request_template_json")),
                    query_params=parse_json(data.get("query_params_json")),
                    response_path=data.get("response_path", source.response_path),
                    response_format_template=data.get(
                        "response_format_template", source.response_format_template
                    ),
                    cache_ttl_seconds=data.get(
                        "cache_ttl_seconds", source.cache_ttl_seconds
                    ),
                    timeout_seconds=data.get("timeout_seconds", source.timeout_seconds),
                    retry_count=data.get("retry_count", source.retry_count),
                    rate_limit_rpm=data.get("rate_limit_rpm", source.rate_limit_rpm),
                    data_type=data.get("data_type", source.data_type),
                    best_for=data.get("best_for", source.best_for),
                    config=parse_json(data.get("config_json")),
                )

            if not updated:
                return jsonify({"error": "Live data source not found"}), 404
            return jsonify(updated.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.error(f"Error updating live data source: {e}")
            return jsonify({"error": "Failed to update live data source"}), 500

    @admin.route("/api/live-data-sources/<int:source_id>", methods=["DELETE"])
    @require_auth_api
    def delete_live_data_source(source_id: int):
        """Delete a live data source."""
        from db import (
            delete_live_data_source as db_delete_source,
        )
        from db import (
            get_live_data_source_by_id,
        )

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        # Don't allow deleting built-in sources
        if source.source_type.startswith("builtin_"):
            return jsonify({"error": "Cannot delete built-in sources"}), 400

        try:
            success = db_delete_source(source_id)
            if not success:
                return jsonify(
                    {
                        "error": "Cannot delete: source is linked to one or more Smart Aliases"
                    }
                ), 400
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Error deleting live data source: {e}")
            return jsonify({"error": "Failed to delete live data source"}), 500

    @admin.route("/api/live-data-sources/<int:source_id>/test", methods=["POST"])
    @require_auth_api
    def test_live_data_source(source_id: int):
        """Test a live data source connection and update status."""
        from db import get_live_data_source_by_id, update_live_data_source_status
        from live import get_provider_for_source

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        try:
            provider = get_provider_for_source(source)
            success, message = provider.test_connection()

            # Get tool count for MCP sources
            tool_count = 0
            if success and source.source_type == "mcp_server":
                from live.sources import MCPProvider

                if isinstance(provider, MCPProvider):
                    tools = provider.list_tools()
                    tool_count = len(tools)

            # Update the source status in database
            if success:
                update_live_data_source_status(
                    source_id, success=True, tool_count=tool_count
                )
            else:
                update_live_data_source_status(source_id, success=False, error=message)

            return jsonify(
                {"success": success, "message": message, "tool_count": tool_count}
            )
        except Exception as e:
            logger.error(f"Error testing live data source: {e}")
            update_live_data_source_status(source_id, success=False, error=str(e))
            return jsonify({"success": False, "message": str(e)})

    @admin.route("/api/live-data-sources/<int:source_id>/tools", methods=["GET"])
    @require_auth_api
    def get_live_data_source_tools(source_id: int):
        """Get MCP tools for a live data source."""
        from db import get_live_data_source_by_id
        from live.sources import MCPProvider

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        if source.source_type != "mcp_server":
            return jsonify({"tools": [], "message": "Not an MCP source"})

        try:
            provider = MCPProvider(source)
            tools = provider.list_tools()
            # Simplify tool info for UI
            tool_list = []
            for tool in tools:
                schema = tool.get("inputSchema", {})
                props = schema.get("properties", {})
                # Get visible params only
                params = [
                    {"name": k, "required": k in schema.get("required", [])}
                    for k, v in props.items()
                    if not v.get("hidden")
                ]
                tool_list.append(
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "params": params,
                    }
                )
            return jsonify({"tools": tool_list, "count": len(tool_list)})
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return jsonify({"tools": [], "error": str(e)})

    @admin.route(
        "/api/live-data-sources/<int:source_id>/endpoint-stats", methods=["GET"]
    )
    @require_auth_api
    def get_live_data_source_endpoint_stats(source_id: int):
        """Get endpoint statistics for a live data source."""
        from db import get_live_data_source_by_id
        from db.live_data_sources import get_endpoint_stats

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        stats = get_endpoint_stats(source_id)
        return jsonify({"stats": stats})

    @admin.route(
        "/api/live-data-sources/<int:source_id>/reset-broken-tools", methods=["POST"]
    )
    @require_auth_api
    def reset_broken_tools(source_id: int):
        """Reset broken tools for a live data source so they can be tried again."""
        from db import get_live_data_source_by_id
        from db.live_data_sources import reset_all_broken_tools, reset_broken_tool

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        data = request.get_json() or {}
        tool_name = data.get("tool_name")

        if tool_name:
            # Reset specific tool
            success = reset_broken_tool(source_id, tool_name)
            if success:
                return jsonify({"success": True, "message": f"Reset tool: {tool_name}"})
            else:
                return jsonify({"error": f"Tool not found: {tool_name}"}), 404
        else:
            # Reset all broken tools
            count = reset_all_broken_tools(source_id)
            return jsonify(
                {"success": True, "message": f"Reset {count} broken tool(s)"}
            )

    @admin.route("/api/live-data-sources/<int:source_id>/fetch", methods=["POST"])
    @require_auth_api
    def fetch_live_data_source(source_id: int):
        """Fetch data from a live data source with a test query."""
        from db import get_live_data_source_by_id
        from live import get_provider_for_source

        source = get_live_data_source_by_id(source_id)
        if not source:
            return jsonify({"error": "Live data source not found"}), 404

        data = request.get_json() or {}
        query = data.get("query", "test")
        params = data.get("params")  # Context/params for the provider

        try:
            provider = get_provider_for_source(source)
            result = provider.fetch(query, params)
            return jsonify(
                {
                    "success": result.success,
                    "data": result.data,
                    "formatted": result.formatted,
                    "cache_hit": result.cache_hit,
                    "latency_ms": result.latency_ms,
                    "error": result.error,
                }
            )
        except Exception as e:
            logger.error(f"Error fetching from live data source: {e}")
            return jsonify({"success": False, "error": str(e)})

    @admin.route("/api/live-data-sources/sync-google", methods=["POST"])
    @require_auth_api
    def sync_google_live_sources():
        """Create live sources for all existing Google Calendar/Tasks/Gmail document stores."""
        from db.live_data_sources import sync_all_google_live_sources

        try:
            created = sync_all_google_live_sources()
            return jsonify(
                {
                    "success": True,
                    "created": created,
                    "message": f"Created {len(created)} live source(s)",
                }
            )
        except Exception as e:
            logger.error(f"Error syncing Google live sources: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/live-data-sources/for-smart-alias", methods=["GET"])
    @require_auth_api
    def list_live_data_sources_for_smart_alias():
        """List all live data sources for Smart Alias configuration.

        Excludes unified sources (which are shown in Document Stores instead).
        """
        from db import get_all_live_data_sources

        # Get live source types that are handled by unified sources
        # These should only appear in Document Stores, not here
        try:
            from plugin_base.loader import get_live_source_to_unified_map

            unified_live_types = set(get_live_source_to_unified_map().keys())
        except ImportError:
            unified_live_types = set()

        sources = get_all_live_data_sources()
        return jsonify(
            [
                {
                    "id": s.id,
                    "name": s.name,
                    "source_type": s.source_type,
                    "data_type": s.data_type,
                    "description": s.description,
                    "enabled": s.enabled,
                    "config": s.config,  # Include plugin config for display name
                }
                for s in sources
                if s.source_type not in unified_live_types
            ]
        )

    # -------------------------------------------------------------------------
    # Widget API - External Dashboard Integration (Homepage, etc.)
    # -------------------------------------------------------------------------

    @admin.route("/api/widget", methods=["GET"])
    @require_widget_api_key
    def get_widget_data():
        """
        Get statistics for external dashboard widgets.

        Authenticates via X-API-Key header (set WIDGET_API_KEY env var).

        Query parameters:
        - days: Number of days to include (default: 1 for today, options: 1, 7, 30, 90)
        - tag: Optional tag filter to show stats for specific tag only

        Returns compact JSON suitable for Homepage widgets:
        {
            "requests": 123,
            "tokens": 50000,
            "cost": 1.23,
            "cache_hits": 10,
            "cache_saved": 0.45,
            "errors": 2,
            "period": "today"
        }
        """
        from datetime import datetime, timedelta

        from sqlalchemy import func

        from db.models import DailyStats, RequestLog

        # Parse parameters
        days = request.args.get("days", "1")
        try:
            days = int(days)
            if days not in (1, 7, 30, 90):
                days = 1
        except ValueError:
            days = 1

        tag_filter = request.args.get("tag")

        # Calculate date range
        start_date = datetime.utcnow() - timedelta(days=days)

        # Period label for response
        period_labels = {1: "today", 7: "7d", 30: "30d", 90: "90d"}
        period = period_labels.get(days, f"{days}d")

        with get_db_context() as db:
            if tag_filter:
                # Query RequestLog for tag-filtered stats
                # Note: RequestLog.tag may contain comma-separated tags
                stats = (
                    db.query(
                        func.count(RequestLog.id),
                        func.sum(RequestLog.input_tokens),
                        func.sum(RequestLog.output_tokens),
                        func.coalesce(func.sum(RequestLog.cost), 0),
                        func.count(RequestLog.id).filter(
                            RequestLog.is_cache_hit == True
                        ),
                        func.coalesce(func.sum(RequestLog.cache_cost_saved), 0),
                        func.count(RequestLog.id).filter(RequestLog.status_code >= 400),
                        # Inbound requests (client requests received)
                        func.count(RequestLog.id).filter(
                            RequestLog.request_type == "inbound"
                        ),
                        # Outbound requests (main LLM calls made)
                        func.count(RequestLog.id).filter(
                            RequestLog.request_type == "main"
                        ),
                    )
                    .filter(
                        RequestLog.timestamp >= start_date,
                        RequestLog.tag.contains(tag_filter),
                    )
                    .first()
                )

                return jsonify(
                    {
                        "requests": stats[0] or 0,
                        "requests_in": stats[7] or 0,
                        "requests_out": stats[8] or 0,
                        "tokens": (stats[1] or 0) + (stats[2] or 0),
                        "cost": round(float(stats[3] or 0), 2),
                        "cache_hits": stats[4] or 0,
                        "cache_saved": round(float(stats[5] or 0), 2),
                        "errors": stats[6] or 0,
                        "period": period,
                        "tag": tag_filter,
                    }
                )
            else:
                # Use pre-aggregated DailyStats for better performance
                stats = (
                    db.query(
                        func.sum(DailyStats.request_count),
                        func.sum(DailyStats.input_tokens),
                        func.sum(DailyStats.output_tokens),
                        func.sum(DailyStats.estimated_cost),
                        func.sum(DailyStats.error_count),
                    )
                    .filter(
                        DailyStats.date >= start_date,
                        DailyStats.tag.is_(None),
                        DailyStats.provider_id.is_(None),
                        DailyStats.model_id.is_(None),
                    )
                    .first()
                )

                # Get cache stats and request type counts from RequestLog
                extra_stats = (
                    db.query(
                        func.count(RequestLog.id).filter(
                            RequestLog.is_cache_hit == True
                        ),
                        func.sum(RequestLog.cache_cost_saved),
                        # Inbound requests (client requests received)
                        func.count(RequestLog.id).filter(
                            RequestLog.request_type == "inbound"
                        ),
                        # Outbound requests (main LLM calls made)
                        func.count(RequestLog.id).filter(
                            RequestLog.request_type == "main"
                        ),
                    )
                    .filter(RequestLog.timestamp >= start_date)
                    .first()
                )

                return jsonify(
                    {
                        "requests": stats[0] or 0,
                        "requests_in": extra_stats[2] or 0,
                        "requests_out": extra_stats[3] or 0,
                        "tokens": (stats[1] or 0) + (stats[2] or 0),
                        "cost": round(stats[3] or 0, 2),
                        "cache_hits": extra_stats[0] or 0,
                        "cache_saved": round(extra_stats[1] or 0, 2),
                        "errors": stats[4] or 0,
                        "period": period,
                    }
                )

    # =========================================================================
    # PLUGIN MANAGEMENT ENDPOINTS
    # =========================================================================

    @admin.route("/api/plugins")
    @require_auth_api
    def get_all_plugins():
        """Get metadata for all discovered plugins."""
        try:
            from plugin_base.loader import get_all_plugin_metadata

            metadata = get_all_plugin_metadata()
            return jsonify(metadata)
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to get plugin metadata: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/actions")
    @require_auth_api
    def get_action_plugins():
        """Get metadata for all action plugins."""
        try:
            from plugin_base.loader import action_registry

            return jsonify(action_registry.get_all_metadata())
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to get action plugins: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/actions/<action_type>")
    @require_auth_api
    def get_action_plugin(action_type):
        """Get details for a specific action plugin."""
        try:
            from plugin_base.loader import action_registry

            plugin_class = action_registry.get(action_type)
            if not plugin_class:
                return jsonify({"error": f"Plugin not found: {action_type}"}), 404

            # Get detailed metadata
            metadata = {
                "action_type": action_type,
                "display_name": getattr(plugin_class, "display_name", action_type),
                "description": getattr(plugin_class, "description", ""),
                "category": getattr(plugin_class, "category", "other"),
                "icon": getattr(plugin_class, "icon", ""),
                "fields": [f.to_dict() for f in plugin_class.get_config_fields()],
                "actions": [a.to_dict() for a in plugin_class.get_actions()],
            }

            return jsonify(metadata)
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to get action plugin {action_type}: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/live-sources")
    @require_auth_api
    def get_live_source_plugins():
        """Get metadata for all live source plugins."""
        try:
            from plugin_base.loader import live_source_registry

            return jsonify(live_source_registry.get_all_metadata())
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to get live source plugins: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/live-sources/<source_type>")
    @require_auth_api
    def get_live_source_plugin(source_type):
        """Get details for a specific live source plugin."""
        try:
            from plugin_base.loader import live_source_registry

            plugin_class = live_source_registry.get(source_type)
            if not plugin_class:
                return jsonify({"error": f"Plugin not found: {source_type}"}), 404

            # Get detailed metadata
            metadata = {
                "source_type": source_type,
                "display_name": getattr(plugin_class, "display_name", source_type),
                "description": getattr(plugin_class, "description", ""),
                "data_type": getattr(plugin_class, "data_type", ""),
                "best_for": getattr(plugin_class, "best_for", ""),
                "icon": getattr(plugin_class, "icon", ""),
                "is_builtin": live_source_registry._builtin.get(source_type, False),
                "fields": [f.to_dict() for f in plugin_class.get_config_fields()],
                "params": [p.to_dict() for p in plugin_class.get_param_definitions()],
                "designator_hint": plugin_class.get_designator_hint(),
            }

            return jsonify(metadata)
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to get live source plugin {source_type}: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/configs", methods=["GET"])
    @require_auth_api
    def list_plugin_configs_endpoint():
        """List all saved plugin configurations."""
        try:
            from db.plugin_configs import (
                get_all_plugin_configs,
                get_plugin_configs_by_type,
            )

            plugin_type = request.args.get("type")  # Optional filter
            if plugin_type:
                configs = get_plugin_configs_by_type(plugin_type)
            else:
                configs = get_all_plugin_configs()

            # Convert to dicts for JSON serialization
            result = []
            for c in configs:
                import json

                result.append(
                    {
                        "id": c.id,
                        "plugin_type": c.plugin_type,
                        "source_type": c.source_type,
                        "name": c.name,
                        "config": json.loads(c.config_json) if c.config_json else {},
                        "enabled": c.enabled,
                        "created_at": c.created_at.isoformat()
                        if c.created_at
                        else None,
                        "updated_at": c.updated_at.isoformat()
                        if c.updated_at
                        else None,
                    }
                )
            return jsonify(result)
        except Exception as e:
            logger.error(f"Failed to list plugin configs: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/configs", methods=["POST"])
    @require_auth_api
    def create_plugin_config_endpoint():
        """Create a new plugin configuration."""
        try:
            from db.plugin_configs import create_plugin_config as db_create_config
            from plugin_base.loader import action_registry

            data = request.get_json()
            plugin_type = data.get("plugin_type")
            source_type = data.get("source_type")
            name = data.get("name")
            config = data.get("config", {})

            if not plugin_type or not source_type or not name:
                return jsonify(
                    {"error": "plugin_type, source_type, and name are required"}
                ), 400

            # Validate config against plugin schema
            if plugin_type == "action":
                plugin_class = action_registry.get(source_type)
                if not plugin_class:
                    return jsonify(
                        {"error": f"Unknown action plugin: {source_type}"}
                    ), 400

                validation = plugin_class.validate_config(config)
                if not validation.valid:
                    errors = [
                        {"field": e.field, "message": e.message}
                        for e in validation.errors
                    ]
                    return jsonify(
                        {"error": "Validation failed", "errors": errors}
                    ), 400

            new_config = db_create_config(
                plugin_type=plugin_type,
                source_type=source_type,
                name=name,
                config=config,
                enabled=data.get("enabled", True),
            )

            if new_config:
                return jsonify(
                    {"id": new_config.id, "message": "Plugin configuration created"}
                )
            else:
                return jsonify(
                    {"error": "Configuration with this name already exists"}
                ), 400
        except Exception as e:
            logger.error(f"Failed to create plugin config: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/configs/<int:config_id>", methods=["GET"])
    @require_auth_api
    def get_plugin_config_endpoint(config_id):
        """Get a specific plugin configuration."""
        try:
            import json as json_module

            from db.plugin_configs import get_plugin_config as db_get_config

            config = db_get_config(config_id)
            if not config:
                return jsonify({"error": "Configuration not found"}), 404

            result = {
                "id": config.id,
                "plugin_type": config.plugin_type,
                "source_type": config.source_type,
                "config": json_module.loads(config.config_json)
                if config.config_json
                else {},
                "enabled": config.enabled,
                "created_at": config.created_at.isoformat()
                if config.created_at
                else None,
                "updated_at": config.updated_at.isoformat()
                if config.updated_at
                else None,
                "updated_at": config.updated_at.isoformat()
                if config.updated_at
                else None,
            }
            return jsonify(result)
        except Exception as e:
            logger.error(f"Failed to get plugin config {config_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/configs/<int:config_id>", methods=["PUT"])
    @require_auth_api
    def update_plugin_config_endpoint(config_id):
        """Update a plugin configuration."""
        try:
            from db.plugin_configs import get_plugin_config as db_get_config
            from db.plugin_configs import update_plugin_config as db_update_config
            from plugin_base.loader import action_registry

            existing = db_get_config(config_id)
            if not existing:
                return jsonify({"error": "Configuration not found"}), 404

            data = request.get_json()
            config = data.get("config")

            # Validate config if provided
            if config is not None:
                plugin_type = existing.get("plugin_type")
                source_type = existing.get("source_type")

                if plugin_type == "action":
                    plugin_class = action_registry.get(source_type)
                    if plugin_class:
                        validation = plugin_class.validate_config(config)
                        if not validation.valid:
                            errors = [
                                {"field": e.field, "message": e.message}
                                for e in validation.errors
                            ]
                            return jsonify(
                                {"error": "Validation failed", "errors": errors}
                            ), 400

            success = db_update_config(
                config_id,
                name=data.get("name"),
                config=config,
                enabled=data.get("enabled"),
            )

            if success:
                return jsonify({"message": "Plugin configuration updated"})
            else:
                return jsonify({"error": "Update failed"}), 500
        except Exception as e:
            logger.error(f"Failed to update plugin config {config_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/configs/<int:config_id>", methods=["DELETE"])
    @require_auth_api
    def delete_plugin_config_endpoint(config_id):
        """Delete a plugin configuration."""
        try:
            from db.plugin_configs import delete_plugin_config as db_delete_config

            success = db_delete_config(config_id)
            if success:
                return jsonify({"message": "Plugin configuration deleted"})
            else:
                return jsonify({"error": "Configuration not found"}), 404
        except Exception as e:
            logger.error(f"Failed to delete plugin config {config_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/plugins/configs/<int:config_id>/test", methods=["POST"])
    @require_auth_api
    def test_plugin_config_endpoint(config_id):
        """Test a plugin configuration."""
        try:
            from db.plugin_configs import get_plugin_config as db_get_config
            from plugin_base.loader import action_registry

            config_data = db_get_config(config_id)
            if not config_data:
                return jsonify({"error": "Configuration not found"}), 404

            plugin_type = config_data.get("plugin_type")
            source_type = config_data.get("source_type")
            config = config_data.get("config", {})

            if plugin_type == "action":
                plugin_class = action_registry.get(source_type)
                if not plugin_class:
                    return jsonify({"error": f"Plugin not found: {source_type}"}), 404

                try:
                    instance = plugin_class(config)
                    success, message = instance.test_connection()
                    return jsonify({"success": success, "message": message})
                except Exception as e:
                    return jsonify({"success": False, "message": str(e)})
            else:
                return jsonify({"error": f"Test not supported for {plugin_type}"}), 400
        except Exception as e:
            logger.error(f"Failed to test plugin config {config_id}: {e}")
            return jsonify({"error": str(e)}), 500

    # =========================================================================
    # Unified Sources API
    # =========================================================================

    @admin.route("/unified-sources")
    @require_auth
    def unified_sources_page():
        """Unified sources management page."""
        return render_template("unified_sources.html")

    @admin.route("/api/unified-sources/plugins", methods=["GET"])
    @require_auth_api
    def list_unified_source_plugins():
        """List all registered unified source plugins with their metadata."""
        try:
            from plugin_base.loader import unified_source_registry

            plugins = []
            for source_type, plugin_class in unified_source_registry.get_all().items():
                plugins.append(
                    {
                        "source_type": source_type,
                        "display_name": getattr(
                            plugin_class, "display_name", source_type
                        ),
                        "description": getattr(plugin_class, "description", ""),
                        "category": getattr(plugin_class, "category", "other"),
                        "icon": getattr(plugin_class, "icon", "📦"),
                        "supports_rag": getattr(plugin_class, "supports_rag", True),
                        "supports_live": getattr(plugin_class, "supports_live", True),
                        "supports_actions": getattr(
                            plugin_class, "supports_actions", False
                        ),
                        "config_fields": [
                            f.to_dict() for f in plugin_class.get_config_fields()
                        ],
                        "live_params": [
                            p.to_dict() for p in plugin_class.get_live_params()
                        ],
                        "designator_hint": plugin_class.get_designator_hint(),
                    }
                )

            return jsonify({"plugins": plugins})
        except ImportError:
            return jsonify({"plugins": []})
        except Exception as e:
            logger.error(f"Failed to list unified source plugins: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/unified-sources/plugins/<source_type>", methods=["GET"])
    @require_auth_api
    def get_unified_source_plugin(source_type: str):
        """Get detailed information about a specific unified source plugin."""
        try:
            from plugin_base.loader import unified_source_registry

            plugin_class = unified_source_registry.get(source_type)
            if not plugin_class:
                return jsonify({"error": f"Plugin not found: {source_type}"}), 404

            return jsonify(
                {
                    "source_type": source_type,
                    "display_name": getattr(plugin_class, "display_name", source_type),
                    "description": getattr(plugin_class, "description", ""),
                    "category": getattr(plugin_class, "category", "other"),
                    "icon": getattr(plugin_class, "icon", "📦"),
                    "supports_rag": getattr(plugin_class, "supports_rag", True),
                    "supports_live": getattr(plugin_class, "supports_live", True),
                    "supports_actions": getattr(
                        plugin_class, "supports_actions", False
                    ),
                    "supports_incremental": getattr(
                        plugin_class, "supports_incremental", True
                    ),
                    "default_cache_ttl": getattr(
                        plugin_class, "default_cache_ttl", 300
                    ),
                    "default_index_days": getattr(
                        plugin_class, "default_index_days", 90
                    ),
                    "config_fields": [
                        f.to_dict() for f in plugin_class.get_config_fields()
                    ],
                    "live_params": [
                        p.to_dict() for p in plugin_class.get_live_params()
                    ],
                    "designator_hint": plugin_class.get_designator_hint(),
                }
            )
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to get unified source plugin {source_type}: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route(
        "/api/unified-sources/plugins/<source_type>/validate", methods=["POST"]
    )
    @require_auth_api
    def validate_unified_source_config(source_type: str):
        """Validate configuration for a unified source plugin."""
        try:
            from plugin_base.loader import unified_source_registry

            plugin_class = unified_source_registry.get(source_type)
            if not plugin_class:
                return jsonify({"error": f"Plugin not found: {source_type}"}), 404

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            config = data.get("config", {})
            result = plugin_class.validate_config(config)

            return jsonify(result.to_dict())
        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to validate unified source config: {e}")
            return jsonify({"error": str(e)}), 500

    @admin.route("/api/unified-sources/plugins/<source_type>/test", methods=["POST"])
    @require_auth_api
    def test_unified_source_plugin(source_type: str):
        """Test a unified source plugin configuration."""
        try:
            from plugin_base.loader import unified_source_registry

            plugin_class = unified_source_registry.get(source_type)
            if not plugin_class:
                return jsonify({"error": f"Plugin not found: {source_type}"}), 404

            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            config = data.get("config", {})

            # Validate first
            validation = plugin_class.validate_config(config)
            if not validation.valid:
                return jsonify(
                    {
                        "success": False,
                        "message": f"Invalid configuration: {validation.error_message}",
                    }
                )

            # Create instance and test
            try:
                instance = plugin_class(config)
                success, message = instance.test_connection()
                return jsonify({"success": success, "message": message})
            except Exception as e:
                return jsonify({"success": False, "message": str(e)})

        except ImportError:
            return jsonify({"error": "Plugin system not available"}), 500
        except Exception as e:
            logger.error(f"Failed to test unified source plugin: {e}")
            return jsonify({"error": str(e)}), 500

    return admin
