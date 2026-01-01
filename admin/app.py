"""
Admin Flask blueprint for ollama-llm-proxy.

Provides:
- Web UI for managing providers, models, and settings
- REST API for CRUD operations
- Authentication via session cookies
"""

import logging
import os
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

from db import Alias, Model, Provider, Setting, get_db_context
from db.importer import import_all_from_yaml

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

    # -------------------------------------------------------------------------
    # Provider API
    # -------------------------------------------------------------------------

    @admin.route("/api/providers", methods=["GET"])
    @require_auth_api
    def list_providers():
        """List all providers."""
        with get_db_context() as db:
            providers = db.query(Provider).order_by(Provider.display_order).all()
            return jsonify([p.to_dict() for p in providers])

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
        """Create a new provider."""
        data = request.get_json() or {}

        required = ["id", "type"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        with get_db_context() as db:
            # Check if already exists
            existing = db.query(Provider).filter(Provider.id == data["id"]).first()
            if existing:
                return jsonify({"error": "Provider already exists"}), 409

            # Get next display order
            max_order = (
                db.query(Provider.display_order)
                .order_by(Provider.display_order.desc())
                .first()
            )
            next_order = (max_order[0] + 1) if max_order else 0

            provider = Provider(
                id=data["id"],
                type=data["type"],
                base_url=data.get("base_url"),
                api_key_env=data.get("api_key_env"),
                enabled=data.get("enabled", True),
                display_order=next_order,
            )
            db.add(provider)
            db.commit()

            return jsonify(provider.to_dict()), 201

    @admin.route("/api/providers/<provider_id>", methods=["PUT"])
    @require_auth_api
    def update_provider(provider_id: str):
        """Update a provider."""
        data = request.get_json() or {}

        with get_db_context() as db:
            provider = db.query(Provider).filter(Provider.id == provider_id).first()
            if not provider:
                return jsonify({"error": "Provider not found"}), 404

            # Update allowed fields
            if "type" in data:
                provider.type = data["type"]
            if "base_url" in data:
                provider.base_url = data["base_url"]
            if "api_key_env" in data:
                provider.api_key_env = data["api_key_env"]
            if "enabled" in data:
                provider.enabled = data["enabled"]
            if "display_order" in data:
                provider.display_order = data["display_order"]

            db.commit()
            return jsonify(provider.to_dict())

    @admin.route("/api/providers/<provider_id>", methods=["DELETE"])
    @require_auth_api
    def delete_provider(provider_id: str):
        """Delete a provider and all its models/aliases."""
        with get_db_context() as db:
            provider = db.query(Provider).filter(Provider.id == provider_id).first()
            if not provider:
                return jsonify({"error": "Provider not found"}), 404

            db.delete(provider)
            db.commit()
            return jsonify({"success": True})

    @admin.route("/api/providers/<provider_id>/test", methods=["POST"])
    @require_auth_api
    def test_provider(provider_id: str):
        """Test a provider's API key."""
        with get_db_context() as db:
            provider = db.query(Provider).filter(Provider.id == provider_id).first()
            if not provider:
                return jsonify({"error": "Provider not found"}), 404

            # Check if API key is configured
            api_key = None
            if provider.api_key_env:
                api_key = os.environ.get(provider.api_key_env)

            if not api_key:
                return jsonify(
                    {
                        "success": False,
                        "error": f"API key not found. Set {provider.api_key_env} environment variable.",
                    }
                )

            # TODO: Actually test the API connection
            # For now, just verify the key exists
            return jsonify(
                {
                    "success": True,
                    "message": f"API key found ({len(api_key)} characters)",
                }
            )

    # -------------------------------------------------------------------------
    # Model API
    # -------------------------------------------------------------------------

    @admin.route("/api/models", methods=["GET"])
    @require_auth_api
    def list_models():
        """List all models, optionally filtered by provider."""
        provider_id = request.args.get("provider")

        with get_db_context() as db:
            query = db.query(Model)
            if provider_id:
                query = query.filter(Model.provider_id == provider_id)

            models = query.order_by(Model.provider_id, Model.id).all()
            return jsonify([m.to_dict() for m in models])

    @admin.route("/api/models/<provider_id>/<model_id>", methods=["GET"])
    @require_auth_api
    def get_model(provider_id: str, model_id: str):
        """Get a single model."""
        with get_db_context() as db:
            model = (
                db.query(Model)
                .filter(Model.id == model_id, Model.provider_id == provider_id)
                .first()
            )
            if not model:
                return jsonify({"error": "Model not found"}), 404
            return jsonify(model.to_dict())

    @admin.route("/api/models", methods=["POST"])
    @require_auth_api
    def create_model():
        """Create a new model."""
        data = request.get_json() or {}

        required = ["id", "provider_id", "family"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        with get_db_context() as db:
            # Check provider exists
            provider = (
                db.query(Provider).filter(Provider.id == data["provider_id"]).first()
            )
            if not provider:
                return jsonify({"error": "Provider not found"}), 404

            # Check if model already exists
            existing = (
                db.query(Model)
                .filter(
                    Model.id == data["id"], Model.provider_id == data["provider_id"]
                )
                .first()
            )
            if existing:
                return jsonify({"error": "Model already exists"}), 409

            model = Model(
                id=data["id"],
                provider_id=data["provider_id"],
                family=data["family"],
                description=data.get("description"),
                context_length=data.get("context_length", 128000),
                supports_system_prompt=data.get("supports_system_prompt", True),
                use_max_completion_tokens=data.get("use_max_completion_tokens", False),
                enabled=data.get("enabled", True),
            )
            model.capabilities = data.get("capabilities", [])
            model.unsupported_params = data.get("unsupported_params", [])

            db.add(model)
            db.commit()

            return jsonify(model.to_dict()), 201

    @admin.route("/api/models/<provider_id>/<model_id>", methods=["PUT"])
    @require_auth_api
    def update_model(provider_id: str, model_id: str):
        """Update a model."""
        data = request.get_json() or {}

        with get_db_context() as db:
            model = (
                db.query(Model)
                .filter(Model.id == model_id, Model.provider_id == provider_id)
                .first()
            )
            if not model:
                return jsonify({"error": "Model not found"}), 404

            # Update allowed fields
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

            db.commit()
            return jsonify(model.to_dict())

    @admin.route("/api/models/<provider_id>/<model_id>", methods=["DELETE"])
    @require_auth_api
    def delete_model(provider_id: str, model_id: str):
        """Delete a model."""
        with get_db_context() as db:
            model = (
                db.query(Model)
                .filter(Model.id == model_id, Model.provider_id == provider_id)
                .first()
            )
            if not model:
                return jsonify({"error": "Model not found"}), 404

            db.delete(model)
            db.commit()
            return jsonify({"success": True})

    # -------------------------------------------------------------------------
    # Alias API
    # -------------------------------------------------------------------------

    @admin.route("/api/aliases", methods=["GET"])
    @require_auth_api
    def list_aliases():
        """List all aliases, optionally filtered by provider."""
        provider_id = request.args.get("provider")

        with get_db_context() as db:
            query = db.query(Alias)
            if provider_id:
                query = query.filter(Alias.provider_id == provider_id)

            aliases = query.order_by(Alias.provider_id, Alias.alias).all()
            return jsonify([a.to_dict() for a in aliases])

    @admin.route("/api/aliases", methods=["POST"])
    @require_auth_api
    def create_alias():
        """Create a new alias."""
        data = request.get_json() or {}

        required = ["alias", "model_id", "provider_id"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        with get_db_context() as db:
            # Check if alias already exists
            existing = db.query(Alias).filter(Alias.alias == data["alias"]).first()
            if existing:
                return jsonify({"error": "Alias already exists"}), 409

            alias = Alias(
                alias=data["alias"],
                model_id=data["model_id"],
                provider_id=data["provider_id"],
            )
            db.add(alias)
            db.commit()

            return jsonify(alias.to_dict()), 201

    @admin.route("/api/aliases/<alias>", methods=["PUT"])
    @require_auth_api
    def update_alias(alias: str):
        """Update an alias."""
        data = request.get_json() or {}

        with get_db_context() as db:
            alias_obj = db.query(Alias).filter(Alias.alias == alias).first()
            if not alias_obj:
                return jsonify({"error": "Alias not found"}), 404

            if "model_id" in data:
                alias_obj.model_id = data["model_id"]
            if "provider_id" in data:
                alias_obj.provider_id = data["provider_id"]

            db.commit()
            return jsonify(alias_obj.to_dict())

    @admin.route("/api/aliases/<alias>", methods=["DELETE"])
    @require_auth_api
    def delete_alias(alias: str):
        """Delete an alias."""
        with get_db_context() as db:
            alias_obj = db.query(Alias).filter(Alias.alias == alias).first()
            if not alias_obj:
                return jsonify({"error": "Alias not found"}), 404

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

    @admin.route("/api/config/import", methods=["POST"])
    @require_auth_api
    def import_config():
        """Import configuration from YAML files."""
        data = request.get_json() or {}
        overwrite = data.get("overwrite", False)

        with get_db_context() as db:
            stats = import_all_from_yaml(db, overwrite=overwrite)

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
        with get_db_context() as db:
            provider_count = db.query(Provider).count()
            enabled_providers = (
                db.query(Provider).filter(Provider.enabled == True).count()
            )
            model_count = db.query(Model).count()
            enabled_models = db.query(Model).filter(Model.enabled == True).count()
            alias_count = db.query(Alias).count()

            # Get configured providers (have API key)
            configured = 0
            providers = db.query(Provider).filter(Provider.enabled == True).all()
            for p in providers:
                if p.api_key_env and os.environ.get(p.api_key_env):
                    configured += 1

            return jsonify(
                {
                    "providers": {
                        "total": provider_count,
                        "enabled": enabled_providers,
                        "configured": configured,
                    },
                    "models": {
                        "total": model_count,
                        "enabled": enabled_models,
                    },
                    "aliases": {
                        "total": alias_count,
                    },
                }
            )

    return admin
