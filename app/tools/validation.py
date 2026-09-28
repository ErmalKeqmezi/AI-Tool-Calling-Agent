"""A small, dependency-free validator for the JSON Schema subset our tools use.

Supported: type (string, number, integer, boolean, object, array, null), properties,
required, additionalProperties (false), enum, minimum/maximum, minLength/maxLength, items.
Deliberately small so you can read the whole thing.
"""

from __future__ import annotations

from typing import Any

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    # bool is a subclass of int in Python - exclude it explicitly
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "null": lambda v: v is None,
}


def validate(value: Any, schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    errors: list[str] = []

    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS[t](value) for t in types):
            return [f"{path}: expected {' or '.join(types)}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}, got {value!r}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: must be at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: must be at most {schema['maxLength']} characters")

    if _TYPE_CHECKS["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required argument '{name}'")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}: unexpected argument '{name}'")
        for name, sub_value in value.items():
            if name in properties:
                errors.extend(validate(sub_value, properties[name], f"{path}.{name}"))

    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate(item, schema["items"], f"{path}[{i}]"))

    return errors
