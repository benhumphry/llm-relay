"""
Unit tests for plugin_base/common.py

Tests the core dataclasses and validation utilities used by all plugin types.
"""

import pytest

from plugin_base.common import (
    FieldDefinition,
    FieldType,
    SelectOption,
    ValidationError,
    ValidationResult,
    validate_config,
    validate_field_value,
)


class TestSelectOption:
    """Tests for SelectOption dataclass."""

    def test_to_dict(self):
        opt = SelectOption(value="foo", label="Foo Label")
        assert opt.to_dict() == {"value": "foo", "label": "Foo Label"}


class TestFieldDefinition:
    """Tests for FieldDefinition dataclass."""

    def test_minimal_definition(self):
        field = FieldDefinition(
            name="api_key",
            label="API Key",
            field_type=FieldType.PASSWORD,
        )
        assert field.name == "api_key"
        assert field.required is False
        assert field.default is None

    def test_to_dict_minimal(self):
        field = FieldDefinition(
            name="api_key",
            label="API Key",
            field_type=FieldType.PASSWORD,
        )
        result = field.to_dict()
        assert result["name"] == "api_key"
        assert result["label"] == "API Key"
        assert result["field_type"] == "password"
        assert result["required"] is False
        assert "default" not in result  # Not included when None

    def test_to_dict_full(self):
        field = FieldDefinition(
            name="count",
            label="Count",
            field_type=FieldType.INTEGER,
            required=True,
            default=10,
            help_text="Number of items",
            placeholder="Enter count",
            min_value=1,
            max_value=100,
        )
        result = field.to_dict()
        assert result["name"] == "count"
        assert result["field_type"] == "integer"
        assert result["required"] is True
        assert result["default"] == 10
        assert result["help_text"] == "Number of items"
        assert result["min_value"] == 1
        assert result["max_value"] == 100

    def test_to_dict_with_options(self):
        field = FieldDefinition(
            name="provider",
            label="Provider",
            field_type=FieldType.SELECT,
            options=[
                SelectOption(value="google", label="Google"),
                {"value": "slack", "label": "Slack"},  # Dict form also supported
            ],
        )
        result = field.to_dict()
        assert len(result["options"]) == 2
        assert result["options"][0] == {"value": "google", "label": "Google"}
        assert result["options"][1] == {"value": "slack", "label": "Slack"}

    def test_to_dict_with_depends_on(self):
        field = FieldDefinition(
            name="folder_id",
            label="Folder",
            field_type=FieldType.FOLDER_PICKER,
            depends_on={"source_type": "gdrive"},
        )
        result = field.to_dict()
        assert result["depends_on"] == {"source_type": "gdrive"}

    def test_to_dict_with_picker_options(self):
        field = FieldDefinition(
            name="oauth_account",
            label="Google Account",
            field_type=FieldType.OAUTH_ACCOUNT,
            picker_options={"provider": "google", "scopes": ["gmail", "drive"]},
        )
        result = field.to_dict()
        assert result["picker_options"]["provider"] == "google"
        assert "gmail" in result["picker_options"]["scopes"]

    def test_string_field_type(self):
        """Field type can be string for flexibility."""
        field = FieldDefinition(
            name="custom",
            label="Custom",
            field_type="custom_type",
        )
        result = field.to_dict()
        assert result["field_type"] == "custom_type"


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_to_dict(self):
        error = ValidationError(field="api_key", message="This field is required")
        assert error.to_dict() == {
            "field": "api_key",
            "message": "This field is required",
        }


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert len(result.errors) == 0
        assert result.error_message == ""

    def test_invalid_result(self):
        result = ValidationResult(
            valid=False,
            errors=[
                ValidationError("field1", "Error 1"),
                ValidationError("field2", "Error 2"),
            ],
        )
        assert result.valid is False
        assert len(result.errors) == 2
        assert "field1: Error 1" in result.error_message
        assert "field2: Error 2" in result.error_message

    def test_to_dict(self):
        result = ValidationResult(
            valid=False,
            errors=[ValidationError("api_key", "Required")],
        )
        d = result.to_dict()
        assert d["valid"] is False
        assert len(d["errors"]) == 1
        assert d["errors"][0]["field"] == "api_key"


class TestValidateFieldValue:
    """Tests for validate_field_value function."""

    def test_required_field_missing(self):
        field = FieldDefinition(
            name="api_key",
            label="API Key",
            field_type=FieldType.TEXT,
            required=True,
        )
        errors = validate_field_value(field, None)
        assert len(errors) == 1
        assert errors[0].message == "This field is required"

    def test_required_field_empty_string(self):
        field = FieldDefinition(
            name="api_key",
            label="API Key",
            field_type=FieldType.TEXT,
            required=True,
        )
        errors = validate_field_value(field, "")
        assert len(errors) == 1
        assert errors[0].message == "This field is required"

    def test_optional_field_missing(self):
        field = FieldDefinition(
            name="description",
            label="Description",
            field_type=FieldType.TEXT,
            required=False,
        )
        errors = validate_field_value(field, None)
        assert len(errors) == 0

    def test_integer_valid(self):
        field = FieldDefinition(
            name="count",
            label="Count",
            field_type=FieldType.INTEGER,
            min_value=1,
            max_value=100,
        )
        errors = validate_field_value(field, 50)
        assert len(errors) == 0

    def test_integer_below_min(self):
        field = FieldDefinition(
            name="count",
            label="Count",
            field_type=FieldType.INTEGER,
            min_value=1,
        )
        errors = validate_field_value(field, 0)
        assert len(errors) == 1
        assert "Minimum value" in errors[0].message

    def test_integer_above_max(self):
        field = FieldDefinition(
            name="count",
            label="Count",
            field_type=FieldType.INTEGER,
            max_value=100,
        )
        errors = validate_field_value(field, 101)
        assert len(errors) == 1
        assert "Maximum value" in errors[0].message

    def test_integer_invalid_type(self):
        field = FieldDefinition(
            name="count",
            label="Count",
            field_type=FieldType.INTEGER,
        )
        errors = validate_field_value(field, "not a number")
        assert len(errors) == 1
        assert "Must be an integer" in errors[0].message

    def test_number_valid(self):
        field = FieldDefinition(
            name="rate",
            label="Rate",
            field_type=FieldType.NUMBER,
            min_value=0.0,
            max_value=1.0,
        )
        errors = validate_field_value(field, 0.5)
        assert len(errors) == 0

    def test_number_invalid(self):
        field = FieldDefinition(
            name="rate",
            label="Rate",
            field_type=FieldType.NUMBER,
        )
        errors = validate_field_value(field, "not a number")
        assert len(errors) == 1
        assert "Must be a number" in errors[0].message

    def test_text_length_valid(self):
        field = FieldDefinition(
            name="name",
            label="Name",
            field_type=FieldType.TEXT,
            min_length=3,
            max_length=50,
        )
        errors = validate_field_value(field, "Valid Name")
        assert len(errors) == 0

    def test_text_too_short(self):
        field = FieldDefinition(
            name="name",
            label="Name",
            field_type=FieldType.TEXT,
            min_length=3,
        )
        errors = validate_field_value(field, "AB")
        assert len(errors) == 1
        assert "Minimum length" in errors[0].message

    def test_text_too_long(self):
        field = FieldDefinition(
            name="name",
            label="Name",
            field_type=FieldType.TEXT,
            max_length=5,
        )
        errors = validate_field_value(field, "Too Long Name")
        assert len(errors) == 1
        assert "Maximum length" in errors[0].message

    def test_text_pattern_valid(self):
        field = FieldDefinition(
            name="email",
            label="Email",
            field_type=FieldType.TEXT,
            pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        )
        errors = validate_field_value(field, "test@example.com")
        assert len(errors) == 0

    def test_text_pattern_invalid(self):
        field = FieldDefinition(
            name="email",
            label="Email",
            field_type=FieldType.TEXT,
            pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
        )
        errors = validate_field_value(field, "not-an-email")
        assert len(errors) == 1
        assert "Must match pattern" in errors[0].message

    def test_select_valid_option(self):
        field = FieldDefinition(
            name="provider",
            label="Provider",
            field_type=FieldType.SELECT,
            options=[
                SelectOption("google", "Google"),
                SelectOption("slack", "Slack"),
            ],
        )
        errors = validate_field_value(field, "google")
        assert len(errors) == 0

    def test_select_invalid_option(self):
        field = FieldDefinition(
            name="provider",
            label="Provider",
            field_type=FieldType.SELECT,
            options=[
                SelectOption("google", "Google"),
                SelectOption("slack", "Slack"),
            ],
        )
        errors = validate_field_value(field, "invalid")
        assert len(errors) == 1
        assert "Invalid option" in errors[0].message

    def test_multiselect_valid(self):
        field = FieldDefinition(
            name="scopes",
            label="Scopes",
            field_type=FieldType.MULTISELECT,
            options=[
                SelectOption("read", "Read"),
                SelectOption("write", "Write"),
                SelectOption("delete", "Delete"),
            ],
        )
        errors = validate_field_value(field, ["read", "write"])
        assert len(errors) == 0

    def test_multiselect_not_list(self):
        field = FieldDefinition(
            name="scopes",
            label="Scopes",
            field_type=FieldType.MULTISELECT,
            options=[SelectOption("read", "Read")],
        )
        errors = validate_field_value(field, "read")
        assert len(errors) == 1
        assert "Must be a list" in errors[0].message

    def test_multiselect_invalid_option(self):
        field = FieldDefinition(
            name="scopes",
            label="Scopes",
            field_type=FieldType.MULTISELECT,
            options=[SelectOption("read", "Read")],
        )
        errors = validate_field_value(field, ["read", "invalid"])
        assert len(errors) == 1
        assert "Invalid option" in errors[0].message

    def test_boolean_valid_true(self):
        field = FieldDefinition(
            name="enabled",
            label="Enabled",
            field_type=FieldType.BOOLEAN,
        )
        errors = validate_field_value(field, True)
        assert len(errors) == 0

    def test_boolean_valid_string(self):
        field = FieldDefinition(
            name="enabled",
            label="Enabled",
            field_type=FieldType.BOOLEAN,
        )
        errors = validate_field_value(field, "true")
        assert len(errors) == 0

    def test_boolean_invalid(self):
        field = FieldDefinition(
            name="enabled",
            label="Enabled",
            field_type=FieldType.BOOLEAN,
        )
        errors = validate_field_value(field, "maybe")
        assert len(errors) == 1
        assert "Must be a boolean" in errors[0].message


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config(self):
        fields = [
            FieldDefinition("api_key", "API Key", FieldType.PASSWORD, required=True),
            FieldDefinition("timeout", "Timeout", FieldType.INTEGER, default=30),
        ]
        config = {"api_key": "secret123", "timeout": 60}
        result = validate_config(fields, config)
        assert result.valid is True

    def test_missing_required_field(self):
        fields = [
            FieldDefinition("api_key", "API Key", FieldType.PASSWORD, required=True),
        ]
        config = {}
        result = validate_config(fields, config)
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].field == "api_key"

    def test_multiple_errors(self):
        fields = [
            FieldDefinition("api_key", "API Key", FieldType.PASSWORD, required=True),
            FieldDefinition("count", "Count", FieldType.INTEGER, min_value=1),
        ]
        config = {"count": 0}  # Missing api_key and count below min
        result = validate_config(fields, config)
        assert result.valid is False
        assert len(result.errors) == 2

    def test_depends_on_visible(self):
        """Field with depends_on is validated when condition is met."""
        fields = [
            FieldDefinition("source_type", "Source", FieldType.SELECT),
            FieldDefinition(
                "folder_id",
                "Folder",
                FieldType.TEXT,
                required=True,
                depends_on={"source_type": "gdrive"},
            ),
        ]
        config = {"source_type": "gdrive"}  # folder_id required but missing
        result = validate_config(fields, config)
        assert result.valid is False
        assert any(e.field == "folder_id" for e in result.errors)

    def test_depends_on_hidden(self):
        """Field with depends_on is skipped when condition is not met."""
        fields = [
            FieldDefinition("source_type", "Source", FieldType.SELECT),
            FieldDefinition(
                "folder_id",
                "Folder",
                FieldType.TEXT,
                required=True,
                depends_on={"source_type": "gdrive"},
            ),
        ]
        config = {"source_type": "notion"}  # folder_id not required
        result = validate_config(fields, config)
        assert result.valid is True

    def test_depends_on_multiple_conditions(self):
        """Field with multiple depends_on conditions."""
        fields = [
            FieldDefinition("source_type", "Source", FieldType.SELECT),
            FieldDefinition("use_oauth", "Use OAuth", FieldType.BOOLEAN),
            FieldDefinition(
                "oauth_account",
                "Account",
                FieldType.TEXT,
                required=True,
                depends_on={"source_type": "gdrive", "use_oauth": True},
            ),
        ]
        # Both conditions met, field required
        config = {"source_type": "gdrive", "use_oauth": True}
        result = validate_config(fields, config)
        assert result.valid is False
        assert any(e.field == "oauth_account" for e in result.errors)

        # One condition not met, field not required
        config = {"source_type": "gdrive", "use_oauth": False}
        result = validate_config(fields, config)
        assert result.valid is True
