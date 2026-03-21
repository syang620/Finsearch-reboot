from __future__ import annotations

import json
from enum import Enum
from typing import Any

from langchain_core.messages import BaseMessage, message_to_dict
from pydantic import BaseModel


def jsonable_snapshot(value: Any) -> Any:
    """Normalize graph snapshot values to JSON-safe structures for checkpoint tests.

    Allowed leaf/value types are:
    - primitives: None, str, int, float, bool
    - pydantic models: serialized with `model_dump(mode="json")`
    - enums: serialized to `.value`
    - LangChain messages: serialized with `message_to_dict`
    - dict/list/tuple containers containing only the above
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if isinstance(value, dict):
        return {str(key): jsonable_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_snapshot(item) for item in value]
    raise TypeError(f"Unsupported snapshot value type: {type(value)!r}")


def assert_snapshot_jsonable(value: Any) -> Any:
    normalized = jsonable_snapshot(value)
    json.dumps(normalized)
    return normalized


def assert_graph_snapshot_jsonable(snapshot: Any) -> Any:
    values = dict(getattr(snapshot, "values", {}) or {})
    return assert_snapshot_jsonable(values)
