"""Safe JSON parsing utilities for storage operations."""

import json
from typing import Any, Dict, Optional, Union

from ..utils.logging_config import get_logger

logger = get_logger("json_utils")


def safe_json_loads(data: Union[str, bytes, None], default: Any = None) -> Any:
    """
    Safely parse JSON data with error handling.

    Args:
        data: JSON string to parse
        default: Default value to return on parse error (defaults to None)

    Returns:
        Parsed JSON data or default value if parsing fails
    """
    if data is None:
        return default

    if isinstance(data, (dict, list)):
        # Already parsed
        return data

    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(
            f"Failed to parse JSON: {e}",
            extra={
                "data_preview": str(data)[:100] if data else None,
                "error_type": type(e).__name__,
            },
        )
        return default


def safe_json_dumps(obj: Any, default: Optional[str] = None) -> str:
    """
    Safely serialize object to JSON with error handling.

    Args:
        obj: Object to serialize
        default: Default value to return on serialization error

    Returns:
        JSON string or default value if serialization fails
    """
    if obj is None:
        return default or "null"

    try:
        return json.dumps(obj)
    except (TypeError, ValueError) as e:
        logger.warning(
            f"Failed to serialize to JSON: {e}",
            extra={"object_type": type(obj).__name__, "error_type": type(e).__name__},
        )
        return default or "{}"


def parse_json_field(row: Dict[str, Any], field: str, default: Any = None) -> None:
    """
    Parse a JSON field in a database row in-place.

    Args:
        row: Dictionary representing a database row
        field: Field name to parse
        default: Default value if parsing fails
    """
    if field in row:
        if row[field] is None:
            row[field] = default
        else:
            row[field] = safe_json_loads(row[field], default)


def parse_json_fields(
    row: Dict[str, Any], fields: list[str], defaults: Optional[Dict[str, Any]] = None
) -> None:
    """
    Parse multiple JSON fields in a database row in-place.

    Args:
        row: Dictionary representing a database row
        fields: List of field names to parse
        defaults: Dictionary of default values for each field
    """
    defaults = defaults or {}
    for field in fields:
        parse_json_field(row, field, defaults.get(field))
