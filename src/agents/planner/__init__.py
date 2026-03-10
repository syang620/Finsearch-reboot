"""Planner agent package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "PlannerAgent",
    "InteractivePlannerAgent",
    "DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE",
    "build_target_resolution_payload",
    "render_target_resolution_prompt",
    "run_target_resolution_prompt",
    "run_interactive_target_resolution",
]

if TYPE_CHECKING:
    from .interactive_target_resolution import (
        DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE,
        InteractivePlannerAgent,
        InteractivePlannerAgent as PlannerAgent,
        build_target_resolution_payload,
        render_target_resolution_prompt,
        run_interactive_target_resolution,
        run_target_resolution_prompt,
    )


def __getattr__(name: str) -> Any:
    if name == "PlannerAgent":
        module = import_module(".interactive_target_resolution", __name__)
        return getattr(module, "InteractivePlannerAgent")

    if name in {
        "InteractivePlannerAgent",
        "DEFAULT_TARGET_RESOLUTION_PROMPT_TEMPLATE",
        "build_target_resolution_payload",
        "render_target_resolution_prompt",
        "run_target_resolution_prompt",
        "run_interactive_target_resolution",
    }:
        module = import_module(".interactive_target_resolution", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
