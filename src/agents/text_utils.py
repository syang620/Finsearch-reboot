from __future__ import annotations

from typing import Any, Optional


def normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text or None
