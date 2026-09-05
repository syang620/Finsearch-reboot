"""Resolution is an identity decision, not execution permission."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict


class StructuredFactResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["resolved", "ambiguous", "unresolved", "missing_inputs"]
    metric_id: str | None
    ticker: str | None
    fiscal_year: int | None
    reason: str | None
    # Opaque copy of the existing selection, including identity and form metadata.
    # Selection belongs to resolution; interpretation of form_type does not.
    selected_target: dict[str, Any] | None
