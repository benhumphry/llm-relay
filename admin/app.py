"""
Admin Flask blueprint for ollama-llm-proxy.

Provides:
- Web UI for managing providers, models, and settings
- REST API for CRUD operations
- Authentication via session cookies
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from db import (
    Alias,
    AliasOverride,
    CustomAlias,
    CustomModel,
    DailyStats,
    Model,
    ModelOverride,
    Provider,
    RequestLog,
    Setting,
    get_db_context,
)
from db.importer import import_all_from_yaml
from providers import registry
from providers.hybrid_loader import (
    get_all_aliases_with_metadata,
    get_all_models_with_metadata,
)
from providers.loader import get_all_provider_names

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
    def inject_auth_status():
        """Make auth status available in all templates."""
        return {"auth_enabled": is_auth_enabled()}

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
        """Providers management page."""
        return render_template("providers.html")

    @admin.route("/models")
    @require_auth
    def models_page():
        """Models management page."""
        return render_template("models.html")

    @admin.route("/aliases")
    @require_auth
    def aliases_page():
        """Aliases management page."""
        return render_template("aliases.html")

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
        from db import OllamaInstance
        from providers import registry
        from providers.loader import get_provider_config
        from providers.ollama_provider import OllamaProvider

        providers_list = []
        seen_ids = set()

        for provider in registry.get_all_providers():
            config = get_provider_config(provider.name)
            api_key_env = config.get("api_key_env")
            has_api_key = bool(api_key_env and os.environ.get(api_key_env))

            # Determine provider type
            if isinstance(provider, OllamaProvider):
                provider_type = "ollama"
                has_api_key = True  # Ollama doesn't need API key
                base_url = provider.base_url
                # Check if it's a system (YAML) or user-created (DB) provider
                is_system = bool(config.get("type") == "ollama")
            else:
                provider_type = config.get("type", provider.name)
                base_url = config.get("base_url")
                is_system = True  # Non-Ollama providers are always from YAML

            providers_list.append(
                {
                    "id": provider.name,
                    "type": provider_type,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "enabled": True,  # All registered providers are enabled
                    "has_api_key": has_api_key,
                    "model_count": len(provider.get_models()),
                    "alias_count": len(provider.get_aliases()),
                    "is_system": is_system,
                }
            )
            seen_ids.add(provider.name)

        # Also include DB Ollama instances not yet in registry (e.g., pending restart)
        with get_db_context() as db:
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
                            "alias_count": 0,
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
                        "alias_count": 0,
                    }
                ), 201
        else:
            # Other provider types (openai-compatible, anthropic) must be configured in YAML
            return jsonify(
                {
                    "error": f"Provider type '{provider_type}' must be configured in config/providers.yml. Only Ollama providers can be added via UI."
                }
            ), 400

    @admin.route("/api/providers/<provider_id>", methods=["PUT"])
    @require_auth_api
    def update_provider(provider_id: str):
        """Update a provider (only Ollama providers can be updated via UI)."""
        from db import OllamaInstance
        from providers import registry
        from providers.loader import get_provider_config

        data = request.get_json() or {}

        # Check if this is a YAML-configured (system) provider
        yaml_config = get_provider_config(provider_id)
        if yaml_config:
            return jsonify(
                {
                    "error": "System providers cannot be modified via UI. Edit config/providers.yml instead."
                }
            ), 400

        # Check if it's an Ollama instance in DB
        with get_db_context() as db:
            instance = (
                db.query(OllamaInstance)
                .filter(OllamaInstance.name == provider_id)
                .first()
            )
            if not instance:
                return jsonify({"error": f"Provider '{provider_id}' not found"}), 404

            # Update fields
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
                    "alias_count": 0,
                }
            )

    @admin.route("/api/providers/<provider_id>", methods=["DELETE"])
    @require_auth_api
    def delete_provider(provider_id: str):
        """Delete a provider (only Ollama providers can be deleted via UI)."""
        from db import OllamaInstance
        from providers import registry
        from providers.loader import get_provider_config

        # Check if this is a YAML-configured (system) provider
        yaml_config = get_provider_config(provider_id)
        if yaml_config:
            return jsonify(
                {
                    "error": "System providers cannot be deleted via UI. Edit config/providers.yml instead."
                }
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

        # Get models for all providers (YAML + registry)
        all_models = []
        seen_providers = set()

        # First, get YAML providers
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
            else:
                override = ModelOverride(
                    provider_id=provider_id,
                    model_id=model_id,
                    disabled=disabled,
                    input_cost=data.get("input_cost"),
                    output_cost=data.get("output_cost"),
                    capabilities=data.get("capabilities"),
                    context_length=data.get("context_length"),
                    description=data.get("description"),
                )
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
    # Alias API (Hybrid: System + Custom)
    # -------------------------------------------------------------------------

    @admin.route("/api/aliases", methods=["GET"])
    @require_auth_api
    def list_aliases():
        """List all aliases (system + custom) with metadata."""
        provider_id = request.args.get("provider")

        if provider_id:
            # Get aliases for specific provider
            aliases = get_all_aliases_with_metadata(provider_id)
            return jsonify(aliases)

        # Get aliases for all providers (YAML + registry)
        all_aliases = []
        seen_providers = set()

        # First, get YAML providers
        for prov_name in get_all_provider_names():
            all_aliases.extend(get_all_aliases_with_metadata(prov_name))
            seen_providers.add(prov_name)

        # Then, get any additional providers from registry (custom DB providers)
        for provider in registry.get_all_providers():
            if provider.name not in seen_providers:
                all_aliases.extend(get_all_aliases_with_metadata(provider.name))

        return jsonify(all_aliases)

    @admin.route("/api/aliases/override", methods=["POST"])
    @require_auth_api
    def set_alias_override():
        """Create or update an override for a system alias."""
        data = request.get_json() or {}

        required = ["provider_id", "alias"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        provider_id = data["provider_id"]
        alias = data["alias"]
        disabled = data.get("disabled", True)

        with get_db_context() as db:
            # Find existing override or create new
            override = (
                db.query(AliasOverride)
                .filter(
                    AliasOverride.provider_id == provider_id,
                    AliasOverride.alias == alias,
                )
                .first()
            )

            if override:
                override.disabled = disabled
            else:
                override = AliasOverride(
                    provider_id=provider_id,
                    alias=alias,
                    disabled=disabled,
                )
                db.add(override)

            db.commit()
            return jsonify(override.to_dict())

    @admin.route("/api/aliases/override/<provider_id>/<alias>", methods=["DELETE"])
    @require_auth_api
    def delete_alias_override(provider_id: str, alias: str):
        """Remove an override for a system alias (re-enables it)."""
        with get_db_context() as db:
            override = (
                db.query(AliasOverride)
                .filter(
                    AliasOverride.provider_id == provider_id,
                    AliasOverride.alias == alias,
                )
                .first()
            )

            if override:
                db.delete(override)
                db.commit()

            return jsonify({"success": True})

    @admin.route("/api/aliases/custom", methods=["POST"])
    @require_auth_api
    def create_custom_alias():
        """Create a new custom alias."""
        data = request.get_json() or {}

        required = ["provider_id", "alias", "model_id"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        with get_db_context() as db:
            # Check if custom alias already exists
            existing = (
                db.query(CustomAlias)
                .filter(
                    CustomAlias.provider_id == data["provider_id"],
                    CustomAlias.alias == data["alias"],
                )
                .first()
            )
            if existing:
                return jsonify({"error": "Custom alias already exists"}), 409

            alias_obj = CustomAlias(
                provider_id=data["provider_id"],
                alias=data["alias"],
                model_id=data["model_id"],
            )
            db.add(alias_obj)
            db.commit()

            return jsonify(alias_obj.to_dict()), 201

    @admin.route("/api/aliases/custom/<int:db_id>", methods=["PUT"])
    @require_auth_api
    def update_custom_alias(db_id: int):
        """Update a custom alias."""
        data = request.get_json() or {}

        with get_db_context() as db:
            alias_obj = db.query(CustomAlias).filter(CustomAlias.id == db_id).first()
            if not alias_obj:
                return jsonify({"error": "Custom alias not found"}), 404

            if "alias" in data:
                alias_obj.alias = data["alias"]
            if "model_id" in data:
                alias_obj.model_id = data["model_id"]

            db.commit()
            return jsonify(alias_obj.to_dict())

    @admin.route("/api/aliases/custom/<int:db_id>", methods=["DELETE"])
    @require_auth_api
    def delete_custom_alias(db_id: int):
        """Delete a custom alias."""
        with get_db_context() as db:
            alias_obj = db.query(CustomAlias).filter(CustomAlias.id == db_id).first()
            if not alias_obj:
                return jsonify({"error": "Custom alias not found"}), 404

            db.delete(alias_obj)
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
            AliasOverride,
            CustomAlias,
            CustomModel,
            CustomProvider,
            ModelOverride,
            OllamaInstance,
            Setting,
        )

        with get_db_context() as db:
            export_data = {
                "version": "2.2",
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
                        "capabilities": o.capabilities,
                        "context_length": o.context_length,
                        "description": o.description,
                    }
                    for o in db.query(ModelOverride).all()
                ],
                "alias_overrides": [
                    {
                        "provider_id": o.provider_id,
                        "alias": o.alias,
                        "disabled": o.disabled,
                    }
                    for o in db.query(AliasOverride).all()
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
                    }
                    for m in db.query(CustomModel).all()
                ],
                "custom_aliases": [
                    {
                        "provider_id": a.provider_id,
                        "alias": a.alias,
                        "model_id": a.model_id,
                    }
                    for a in db.query(CustomAlias).all()
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
            }

        response = jsonify(export_data)
        response.headers["Content-Disposition"] = (
            f"attachment; filename=llm-proxy-config-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.json"
        )
        return response

    @admin.route("/api/config/import", methods=["POST"])
    @require_auth_api
    def import_config():
        """Import database configuration from JSON backup."""
        from db.models import (
            AliasOverride,
            CustomAlias,
            CustomModel,
            CustomProvider,
            ModelOverride,
            OllamaInstance,
            Setting,
        )

        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Validate version
        version = data.get("version", "")
        if not version.startswith("2."):
            return jsonify(
                {"success": False, "error": f"Unsupported export version: {version}"}
            ), 400

        stats = {
            "settings": 0,
            "model_overrides": 0,
            "alias_overrides": 0,
            "custom_models": 0,
            "custom_aliases": 0,
            "ollama_instances": 0,
            "custom_providers": 0,
        }

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
                        context_length=o.get("context_length"),
                        description=o.get("description"),
                    )
                    override.capabilities = o.get("capabilities")
                    db.add(override)
                stats["model_overrides"] += 1

            # Import alias overrides
            for o in data.get("alias_overrides", []):
                existing = (
                    db.query(AliasOverride)
                    .filter(
                        AliasOverride.provider_id == o["provider_id"],
                        AliasOverride.alias == o["alias"],
                    )
                    .first()
                )
                if existing:
                    existing.disabled = o.get("disabled", False)
                else:
                    db.add(
                        AliasOverride(
                            provider_id=o["provider_id"],
                            alias=o["alias"],
                            disabled=o.get("disabled", False),
                        )
                    )
                stats["alias_overrides"] += 1

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
                else:
                    model = CustomModel(
                        provider_id=m["provider_id"],
                        model_id=m["model_id"],
                        family=m.get("family"),
                        description=m.get("description"),
                        context_length=m.get("context_length", 128000),
                        supports_system_prompt=m.get("supports_system_prompt", True),
                        use_max_completion_tokens=m.get(
                            "use_max_completion_tokens", False
                        ),
                        enabled=m.get("enabled", True),
                        input_cost=m.get("input_cost"),
                        output_cost=m.get("output_cost"),
                    )
                    model.capabilities = m.get("capabilities", [])
                    model.unsupported_params = m.get("unsupported_params", [])
                    db.add(model)
                stats["custom_models"] += 1

            # Import custom aliases
            for a in data.get("custom_aliases", []):
                existing = (
                    db.query(CustomAlias)
                    .filter(
                        CustomAlias.provider_id == a["provider_id"],
                        CustomAlias.alias == a["alias"],
                    )
                    .first()
                )
                if existing:
                    existing.model_id = a["model_id"]
                else:
                    db.add(
                        CustomAlias(
                            provider_id=a["provider_id"],
                            alias=a["alias"],
                            model_id=a["model_id"],
                        )
                    )
                stats["custom_aliases"] += 1

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

        return jsonify({"success": True, "stats": stats})

    @admin.route("/api/config/reload", methods=["POST"])
    @require_auth_api
    def reload_config():
        """Reload provider configuration."""
        # This will be implemented when we update the provider loading
        # For now, return success
        return jsonify({"success": True, "message": "Configuration reload requested"})

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

        # Count models and aliases from hybrid system
        total_models = 0
        enabled_models = 0
        system_models = 0
        custom_models = 0
        total_aliases = 0
        system_aliases = 0
        custom_aliases = 0

        for prov_name in get_all_provider_names():
            models = get_all_models_with_metadata(prov_name)
            aliases = get_all_aliases_with_metadata(prov_name)

            total_models += len(models)
            enabled_models += sum(1 for m in models if m.get("enabled", True))
            system_models += sum(1 for m in models if m.get("is_system", True))
            custom_models += sum(1 for m in models if not m.get("is_system", True))

            total_aliases += len(aliases)
            system_aliases += sum(1 for a in aliases if a.get("is_system", True))
            custom_aliases += sum(1 for a in aliases if not a.get("is_system", True))

        # Count providers from registry
        all_providers = registry.get_all_providers()
        provider_count = len(all_providers)

        # Count configured providers (have API key or don't need one)
        configured = 0
        for provider in all_providers:
            # Check if this is an Ollama-type provider (doesn't need API key)
            if isinstance(provider, OllamaProvider):
                # Ollama providers are configured if they're running/reachable
                if provider.is_configured():
                    configured += 1
            else:
                # Other providers need an API key
                config = get_provider_config(provider.name)
                api_key_env = config.get("api_key_env")
                if api_key_env and os.environ.get(api_key_env):
                    configured += 1

        return jsonify(
            {
                "providers": {
                    "total": provider_count,
                    "enabled": provider_count,  # All registered providers are enabled
                    "configured": configured,
                },
                "models": {
                    "total": total_models,
                    "enabled": enabled_models,
                    "system": system_models,
                    "custom": custom_models,
                },
                "aliases": {
                    "total": total_aliases,
                    "system": system_aliases,
                    "custom": custom_aliases,
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
        """List all configured Ollama instances (YAML + DB)."""
        from db import OllamaInstance
        from providers.loader import get_provider_config

        instances = []

        # Get instances from registry (includes both YAML and dynamically added)
        for provider in _get_ollama_providers():
            running = provider.is_configured()
            # Check if this is a YAML-configured provider
            yaml_config = get_provider_config(provider.name)
            is_system = yaml_config.get("type") == "ollama"

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
        from db import OllamaInstance
        from providers import registry
        from providers.loader import get_provider_config

        # Check if this is a YAML-configured (system) instance
        yaml_config = get_provider_config(instance_name)
        if yaml_config.get("type") == "ollama":
            return jsonify(
                {
                    "error": "Cannot modify YAML-configured instances. Edit config/providers.yml instead."
                }
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
        from db import OllamaInstance
        from providers import registry
        from providers.loader import get_provider_config

        # Check if this is a YAML-configured (system) instance
        yaml_config = get_provider_config(instance_name)
        if yaml_config.get("type") == "ollama":
            return jsonify(
                {
                    "error": "Cannot delete YAML-configured instances. Edit config/providers.yml instead."
                }
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
        days = request.args.get("days")
        start = request.args.get("start")
        end = request.args.get("end")

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

        return {
            "start_date": start_date,
            "end_date": end_date,
            "tags": tags,
            "providers": providers,
            "models": models,
            "clients": clients,
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

            return jsonify(
                {
                    "tags": tags,
                    "providers": providers,
                    "models": models,
                    "clients": clients,
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
            if filters["tags"] or filters["providers"] or filters["models"]:
                # Query from RequestLog for filtered results
                query = db.query(
                    func.count(RequestLog.id),
                    func.sum(RequestLog.input_tokens),
                    func.sum(RequestLog.output_tokens),
                    func.count(RequestLog.id).filter(RequestLog.status_code < 400),
                    func.count(RequestLog.id).filter(RequestLog.status_code >= 400),
                ).filter(RequestLog.timestamp >= filters["start_date"])

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["tags"]:
                    # Support multi-tag filtering (tag column may contain comma-separated tags)
                    from sqlalchemy import or_

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

                stats = query.first()

                # Calculate cost from RequestLog (need to join with cost data or estimate)
                # For now, use a simpler cost query
                cost_query = db.query(
                    func.sum(RequestLog.input_tokens),
                    func.sum(RequestLog.output_tokens),
                ).filter(RequestLog.timestamp >= filters["start_date"])

                if filters["end_date"]:
                    cost_query = cost_query.filter(
                        RequestLog.timestamp <= filters["end_date"]
                    )
                if filters["tags"]:
                    from sqlalchemy import or_

                    tag_conditions = []
                    for tag in filters["tags"]:
                        tag_conditions.append(RequestLog.tag == tag)
                        tag_conditions.append(RequestLog.tag.like(f"{tag},%"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag}"))
                        tag_conditions.append(RequestLog.tag.like(f"%,{tag},%"))
                    cost_query = cost_query.filter(or_(*tag_conditions))
                if filters["providers"]:
                    cost_query = cost_query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )
                if filters["models"]:
                    cost_query = cost_query.filter(
                        RequestLog.model_id.in_(filters["models"])
                    )

                # Estimate cost (simplified - ideally would join with model costs)
                estimated_cost = 0.0

                return jsonify(
                    {
                        "period_days": request.args.get("days", 30),
                        "total_requests": stats[0] or 0,
                        "total_input_tokens": stats[1] or 0,
                        "total_output_tokens": stats[2] or 0,
                        "total_tokens": (stats[1] or 0) + (stats[2] or 0),
                        "estimated_cost": estimated_cost,
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
            if filters["providers"] or filters["models"]:
                # Query from RequestLog when filtering by provider/model
                query = db.query(
                    RequestLog.tag,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                ).filter(
                    RequestLog.timestamp >= filters["start_date"],
                    RequestLog.tag.isnot(None),
                )

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

                if filters["providers"]:
                    query = query.filter(
                        RequestLog.provider_id.in_(filters["providers"])
                    )

                if filters["models"]:
                    query = query.filter(RequestLog.model_id.in_(filters["models"]))

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
                            "cost": 0,  # Cost requires model info, simplified for now
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
            if filters["tags"] or filters["models"]:
                # Query from RequestLog when filtering by tag/model
                query = db.query(
                    RequestLog.provider_id,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
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
                            "cost": 0,  # Cost requires model info, simplified for now
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
            if filters["tags"]:
                # Query from RequestLog when filtering by tag
                query = db.query(
                    RequestLog.provider_id,
                    RequestLog.model_id,
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
                ).filter(
                    RequestLog.timestamp >= filters["start_date"],
                    RequestLog.provider_id.isnot(None),
                    RequestLog.model_id.isnot(None),
                )

                if filters["end_date"]:
                    query = query.filter(RequestLog.timestamp <= filters["end_date"])

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
                            "cost": 0,  # Cost requires model info, simplified for now
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
                query = query.filter(client_col.in_(filters["clients"]))

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
                        "cost": 0,  # Cost requires model info, simplified for now
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
            if filters["tags"] or filters["providers"] or filters["models"]:
                # Query from RequestLog when filtering
                # Use func.date() for SQLite compatibility (works with datetime columns)
                date_expr = func.date(RequestLog.timestamp)
                query = db.query(
                    date_expr.label("date"),
                    func.count(RequestLog.id).label("requests"),
                    func.sum(RequestLog.input_tokens).label("input_tokens"),
                    func.sum(RequestLog.output_tokens).label("output_tokens"),
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
                            "cost": 0,  # Cost requires model info, simplified for now
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
            default_tag = (
                db.query(Setting).filter(Setting.key == Setting.KEY_DEFAULT_TAG).first()
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
                    "default_tag": default_tag.value if default_tag else "default",
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
                "default_tag": (Setting.KEY_DEFAULT_TAG, str),
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

    return admin
