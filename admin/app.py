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
            Setting.KEY_WEB_PDF_PARSER,
            Setting.KEY_RAG_PDF_PARSER,
            Setting.KEY_EMBEDDING_PROVIDER,
            Setting.KEY_EMBEDDING_MODEL,
            Setting.KEY_EMBEDDING_OLLAMA_URL,
            Setting.KEY_WEB_RERANK_PROVIDER,
            Setting.KEY_VISION_PROVIDER,
            Setting.KEY_VISION_MODEL,
            Setting.KEY_VISION_OLLAMA_URL,
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
        scraper_provider = data.get("scraper_provider", "builtin")
        if scraper_provider not in ("builtin", "jina"):
            return jsonify({"error": "Invalid scraper provider"}), 400

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
        settings_to_save = {
            Setting.KEY_WEB_SEARCH_PROVIDER: search_provider,
            Setting.KEY_WEB_SEARCH_URL: data.get("search_url", ""),
            Setting.KEY_WEB_SCRAPER_PROVIDER: scraper_provider,
            Setting.KEY_WEB_PDF_PARSER: pdf_parser,
            Setting.KEY_RAG_PDF_PARSER: rag_pdf_parser,
            Setting.KEY_EMBEDDING_PROVIDER: embedding_provider,
            Setting.KEY_EMBEDDING_MODEL: embedding_model,
            Setting.KEY_EMBEDDING_OLLAMA_URL: embedding_ollama_url,
            Setting.KEY_WEB_RERANK_PROVIDER: rerank_provider,
            Setting.KEY_VISION_PROVIDER: vision_provider,
            Setting.KEY_VISION_MODEL: vision_model,
            Setting.KEY_VISION_OLLAMA_URL: vision_ollama_url,
        }

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
                    model_conditions = build_model_filter_conditions(
                        filters["models"], RequestLog
                    )
                    if model_conditions:
                        query = query.filter(or_(*model_conditions))

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

        return jsonify(
            {
                "google": {
                    "available": has_google,
                    "reason": None
                    if has_google
                    else "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET required",
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
        """List all document stores with Google account info."""
        from db import get_all_document_stores
        from db.oauth_tokens import get_oauth_token_info

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
        elif source_type in ("mcp:gdrive", "mcp:gmail", "mcp:gcalendar"):
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
                paperless_url=data.get("paperless_url"),
                paperless_token=data.get("paperless_token"),
                github_repo=data.get("github_repo"),
                github_branch=data.get("github_branch"),
                github_path=data.get("github_path"),
                notion_database_id=data.get("notion_database_id"),
                notion_page_id=data.get("notion_page_id"),
                embedding_provider=data.get("embedding_provider", "local"),
                embedding_model=data.get("embedding_model"),
                ollama_url=data.get("ollama_url"),
                vision_provider=data.get("vision_provider", "local"),
                vision_model=data.get("vision_model"),
                vision_ollama_url=data.get("vision_ollama_url"),
                index_schedule=data.get("index_schedule"),
                chunk_size=data.get("chunk_size", 512),
                chunk_overlap=data.get("chunk_overlap", 50),
                description=data.get("description"),
                enabled=data.get("enabled", True),
            )
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
                paperless_url=data.get("paperless_url"),
                paperless_token=data.get("paperless_token"),
                github_repo=data.get("github_repo"),
                github_branch=data.get("github_branch"),
                github_path=data.get("github_path"),
                notion_database_id=data.get("notion_database_id"),
                notion_page_id=data.get("notion_page_id"),
                embedding_provider=data.get("embedding_provider"),
                embedding_model=data.get("embedding_model"),
                ollama_url=data.get("ollama_url"),
                vision_provider=data.get("vision_provider"),
                vision_model=data.get("vision_model"),
                vision_ollama_url=data.get("vision_ollama_url"),
                index_schedule=data.get("index_schedule"),
                chunk_size=data.get("chunk_size"),
                chunk_overlap=data.get("chunk_overlap"),
                description=data.get("description"),
                enabled=data.get("enabled"),
            )
            if not store:
                return jsonify({"error": "Document store not found"}), 404
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

        if store.smart_enrichers:
            enricher_names = [e.name for e in store.smart_enrichers]
            return jsonify(
                {
                    "error": f"Cannot delete store - used by enrichers: {', '.join(enricher_names)}"
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

    # -------------------------------------------------------------------------
    # OAuth Routes (Google, etc.)
    # -------------------------------------------------------------------------

    # Google OAuth scopes for different services
    # All include userinfo.email to identify the account
    GOOGLE_SCOPES = {
        "drive": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/drive.readonly",
        ],
        "gmail": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/gmail.readonly",
        ],
        "calendar": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/calendar.readonly",
        ],
        "workspace": [
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
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
                # Common enrichment
                max_context_tokens=data.get("max_context_tokens", 4000),
                rerank_provider=data.get("rerank_provider", "local"),
                rerank_model=data.get("rerank_model"),
                rerank_top_n=data.get("rerank_top_n", 20),
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
                # Common enrichment
                max_context_tokens=data.get("max_context_tokens"),
                rerank_provider=data.get("rerank_provider"),
                rerank_model=data.get("rerank_model"),
                rerank_top_n=data.get("rerank_top_n"),
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

    return admin
