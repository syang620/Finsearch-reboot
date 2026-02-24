from __future__ import annotations

import ast
import json
import re
from typing import Any


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", t)
        t = re.sub(r"\n```$", "", t)
    return t.strip()


def parse_llm_list_output(text: str) -> list[str]:
    """
    Best-effort parse of an LLM "list" response into list[str].
    """
    if not isinstance(text, str):
        return []
    t = _strip_code_fences(text)
    if not t or t.startswith("Error:") or t.startswith("Unexpected error:"):
        return []

    try:
        obj = json.loads(t)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    try:
        obj = ast.literal_eval(t)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass

    lines = []
    for line in t.splitlines():
        s = line.strip()
        s = re.sub(r"^[-*•]\s+", "", s)
        s = re.sub(r"^\d+[\.)]\s+", "", s)
        s = s.strip().strip('"').strip("'")
        if s:
            lines.append(s)
    if len(lines) >= 2:
        return lines

    t2 = t.strip().strip("[]")
    parts = [p.strip().strip('"').strip("'") for p in t2.split(",")]
    return [p for p in parts if p]


def chat_with_ollama(
    prompt: str,
    model: str = "llama2",
    *,
    as_list: bool = True,
    options: dict[str, Any] | None = None,
):
    """
    Call a local Ollama server via the `ollama` Python package.
    """
    try:
        import ollama
    except Exception as exc:
        raise ImportError(
            "chat_with_ollama requires the `ollama` Python package and a running Ollama server."
        ) from exc

    if options is None:
        options = {
            "temperature": 0.1,
            "num_predict": 512,
        }

    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            options=options,
        )
        out = response["response"]
        return parse_llm_list_output(out) if as_list else out
    except Exception as e:
        return [] if as_list else f"Error: {e}"


__all__ = [
    "chat_with_ollama",
    "parse_llm_list_output",
]
