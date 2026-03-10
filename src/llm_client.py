from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional, Sequence

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import default_tool_parser
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_DASHSCOPE_TIMEOUT_S = 120.0
DEFAULT_DASHSCOPE_TEMPERATURE = 0.0
DEFAULT_DASHSCOPE_ENABLE_THINKING = False
DEFAULT_LITELLM_TIMEOUT_S = 120.0

_LITELLM_SHORTCUT_MODELS = {
    "gpt": ("LITELLM_GPT_MODEL", "OPENAI_MODEL", "OPENAI_CHAT_MODEL"),
    "claude": ("LITELLM_CLAUDE_MODEL", "ANTHROPIC_MODEL", "ANTHROPIC_CHAT_MODEL"),
    "gemini": ("LITELLM_GEMINI_MODEL", "GEMINI_MODEL", "GOOGLE_MODEL"),
}

_QWEN_MODEL_ALIASES = {
    "qwen2.5:7b": "qwen2.5-7b-instruct",
    "qwen2.5:7b-instruct": "qwen2.5-7b-instruct",
    "qwen2.5:7b-instruct-1m": "qwen2.5-7b-instruct-1m",
    "qwen2.5:14b-instruct": "qwen2.5-14b-instruct",
    "qwen2.5:14b-instruct-1m": "qwen2.5-14b-instruct-1m",
    "qwen2.5-14b-1m": "qwen2.5-14b-instruct-1m",
    "qwen3:4b": "qwen3-4b",
    "qwen3:4b-instruct": "qwen3-4b",
    "qwen3:14b": "qwen3-14b",
    "qwen3:14b-instruct": "qwen3-14b",
}


def is_gpt_chat_model(model_name: str) -> bool:
    raw = str(model_name or "").strip().lower()
    return raw.startswith("gpt")


def is_claude_chat_model(model_name: str) -> bool:
    raw = str(model_name or "").strip().lower()
    return raw.startswith("claude")


def is_qwen_chat_model(model_name: str) -> bool:
    raw = str(model_name or "").strip().lower()
    if not raw.startswith("qwen"):
        return False
    return "embed" not in raw and "rerank" not in raw


def is_gemini_chat_model(model_name: str) -> bool:
    raw = str(model_name or "").strip().lower()
    return raw.startswith("gemini/") or raw.startswith("gemini-")


def resolve_chat_model_backend(model_name: str) -> tuple[str, str]:
    resolved_model = _normalize_litellm_model_alias(model_name)

    if is_qwen_chat_model(resolved_model):
        return "dashscope", resolved_model

    if is_gemini_chat_model(resolved_model):
        return "google_genai", normalize_gemini_chat_model_name(resolved_model)

    if is_claude_chat_model(resolved_model):
        return "litellm", resolved_model

    if is_gpt_chat_model(resolved_model):
        return "litellm", resolved_model

    return "litellm", resolved_model


def normalize_gemini_chat_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if raw.lower().startswith("gemini/"):
        return raw.split("/", 1)[1]
    return raw


def normalize_qwen_chat_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    return _QWEN_MODEL_ALIASES.get(raw.lower(), raw)


def _normalize_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return raw

    if raw.lower().startswith("gemini-") and not raw.lower().startswith("gemini/"):
        return f"gemini/{raw}"

    return raw


def _normalize_litellm_model_alias(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return raw

    key = raw.lower()

    if key in _LITELLM_SHORTCUT_MODELS:
        for env_name in _LITELLM_SHORTCUT_MODELS[key]:
            value = str(os.getenv(env_name, "")).strip()
            if value:
                return _normalize_model_name(value)
        raise RuntimeError(
            f"Model alias {raw!r} requires an env model mapping: "
            f"set one of {', '.join(_LITELLM_SHORTCUT_MODELS[key])}."
        )

    key = raw.lower()
    return _normalize_model_name(raw)


def _normalize_gemini_api_key() -> None:
    if os.getenv("GEMINI_API_KEY"):
        return
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY_1")
    if google_key:
        os.environ["GEMINI_API_KEY"] = google_key


def dashscope_base_url() -> str:
    return str(os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL)).rstrip("/")


def dashscope_api_key() -> str:
    api_key = str(os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set.")
    return api_key


def _normalize_options(
    options: Optional[Dict[str, Any]],
    *,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(options, dict):
        out.update({k: v for k, v in options.items() if v is not None})
    if temperature is not None:
        out["temperature"] = temperature
    if num_predict is not None:
        out["num_predict"] = num_predict
    return out


def _options_to_chat_payload(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if not isinstance(options, dict):
        return payload

    for key, value in options.items():
        if value is None:
            continue
        if key == "num_predict":
            payload["max_tokens"] = int(value)
        else:
            payload[key] = value
    return payload


def _normalize_tool_choice(tool_choice: Any) -> Any:
    if tool_choice == "any":
        return "required"
    return tool_choice


def _response_message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return "" if content is None else str(content)


def dashscope_chat_completion(
    messages: Sequence[Dict[str, Any]],
    *,
    model: str,
    enable_thinking: bool | None = None,
    options: Optional[Dict[str, Any]] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    timeout: float = DEFAULT_DASHSCOPE_TIMEOUT_S,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_url = dashscope_base_url()
    url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    payload: Dict[str, Any] = {
        "model": normalize_qwen_chat_model_name(model),
        "messages": list(messages),
        "stream": False,
    }
    payload.update(_options_to_chat_payload(options))
    if tools:
        payload["tools"] = list(tools)
    normalized_tool_choice = _normalize_tool_choice(tool_choice)
    if normalized_tool_choice is not None:
        payload["tool_choice"] = normalized_tool_choice
    if extra_payload:
        payload.update({k: v for k, v in extra_payload.items() if v is not None})

    # Keep Qwen completions deterministic and non-thinking by default.
    # These flags are intentionally fixed for this backend.
    payload["enable_thinking"] = DEFAULT_DASHSCOPE_ENABLE_THINKING
    _ = enable_thinking
    payload["temperature"] = DEFAULT_DASHSCOPE_TEMPERATURE

    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {dashscope_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def chat_with_dashscope(
    prompt: str,
    *,
    model: str,
    as_list: bool,
    enable_thinking: bool = DEFAULT_DASHSCOPE_ENABLE_THINKING,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_DASHSCOPE_TIMEOUT_S,
) -> str:
    data = dashscope_chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        enable_thinking=enable_thinking,
        options=options,
        timeout=timeout,
    )
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return _response_message_text(message)


def _litellm_chat_completion(
    messages: Sequence[Dict[str, Any]],
    *,
    model: str,
    options: Optional[Dict[str, Any]] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    tool_choice: Any = None,
    timeout: float = DEFAULT_LITELLM_TIMEOUT_S,
    base_url: Optional[str] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    try:
        from litellm import completion
    except Exception as exc:
        raise ImportError(
            "Litellm is required for OpenAI/Anthropic/Gemini-compatible models. "
            "Install it with `pip install -U litellm`."
        ) from exc

    payload: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "stream": False,
    }
    payload.update(_options_to_chat_payload(options))
    if base_url:
        payload["api_base"] = base_url
    if tools:
        payload["tools"] = list(tools)
    normalized_tool_choice = _normalize_tool_choice(tool_choice)
    if normalized_tool_choice is not None:
        payload["tool_choice"] = normalized_tool_choice
    if extra_payload:
        payload.update({k: v for k, v in extra_payload.items() if v is not None})

    resp = completion(**payload)

    if isinstance(resp, dict):
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "dict"):
        return resp.dict()
    return {}


def chat_with_litellm(
    prompt: str,
    *,
    model: str,
    as_list: bool,
    options: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_LITELLM_TIMEOUT_S,
    base_url: Optional[str] = None,
) -> str:
    _ = as_list
    data = _litellm_chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        options=options,
        timeout=timeout,
        base_url=base_url,
    )
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    return _response_message_text(message)


class LiteLLMChatModel(BaseChatModel):
    model_name: str = Field()
    temperature: Optional[float] = Field(default=None)
    num_predict: Optional[int] = Field(default=None)
    timeout: float = Field(default=DEFAULT_LITELLM_TIMEOUT_S)
    base_url: Optional[str] = Field(default=None, exclude=True)
    bound_tools: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    bound_tool_choice: Any = Field(default=None, exclude=True)
    bound_tool_kwargs: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "litellm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "base_url": self.base_url,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "LiteLLMChatModel":
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(
            update={
                "bound_tools": openai_tools,
                "bound_tool_choice": tool_choice,
                "bound_tool_kwargs": dict(kwargs),
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = convert_to_openai_messages(messages)
        options = _normalize_options(
            kwargs.get("options"),
            temperature=self.temperature,
            num_predict=self.num_predict,
        )
        if stop:
            options["stop"] = stop

        data = _litellm_chat_completion(
            openai_messages,
            model=self.model_name,
            options=options,
            tools=self.bound_tools,
            tool_choice=self.bound_tool_choice,
            timeout=float(kwargs.get("timeout", self.timeout)),
            base_url=self.base_url,
            extra_payload=self.bound_tool_kwargs,
        )
        choice = (data.get("choices") or [{}])[0]
        raw_message = choice.get("message") or {}
        raw_tool_calls = raw_message.get("tool_calls") or []
        tool_calls, invalid_tool_calls = default_tool_parser(raw_tool_calls)
        message = AIMessage(
            content=_response_message_text(raw_message),
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            additional_kwargs={"tool_calls": raw_tool_calls} if raw_tool_calls else {},
            response_metadata={
                "model_name": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "token_usage": data.get("usage"),
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"usage": data.get("usage"), "model": data.get("model")},
        )


class DashScopeChatModel(BaseChatModel):
    model_name: str = Field()
    temperature: Optional[float] = Field(default=None)
    num_predict: Optional[int] = Field(default=None)
    timeout: float = Field(default=DEFAULT_DASHSCOPE_TIMEOUT_S)
    bound_tools: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    bound_tool_choice: Any = Field(default=None, exclude=True)
    bound_tool_kwargs: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "dashscope-openai-compatible"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": normalize_qwen_chat_model_name(self.model_name),
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "DashScopeChatModel":
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(
            update={
                "bound_tools": openai_tools,
                "bound_tool_choice": tool_choice,
                "bound_tool_kwargs": dict(kwargs),
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = convert_to_openai_messages(messages)
        options = _normalize_options(
            kwargs.get("options"),
            temperature=self.temperature,
            num_predict=self.num_predict,
        )
        if stop:
            options["stop"] = stop

        data = dashscope_chat_completion(
            openai_messages,
            model=self.model_name,
            options=options,
            tools=self.bound_tools,
            tool_choice=self.bound_tool_choice,
            timeout=float(kwargs.get("timeout", self.timeout)),
            extra_payload=self.bound_tool_kwargs,
        )
        choice = (data.get("choices") or [{}])[0]
        raw_message = choice.get("message") or {}
        raw_tool_calls = raw_message.get("tool_calls") or []
        tool_calls, invalid_tool_calls = default_tool_parser(raw_tool_calls)
        message = AIMessage(
            content=_response_message_text(raw_message),
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            additional_kwargs={"tool_calls": raw_tool_calls} if raw_tool_calls else {},
            response_metadata={
                "model_name": data.get("model"),
                "finish_reason": choice.get("finish_reason"),
                "token_usage": data.get("usage"),
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"usage": data.get("usage"), "model": data.get("model")},
        )


class GoogleGenAIChatModel(BaseChatModel):
    model_name: str = Field()
    temperature: Optional[float] = Field(default=None)
    num_predict: Optional[int] = Field(default=None)
    timeout: float = Field(default=DEFAULT_LITELLM_TIMEOUT_S)
    bound_tools: list[dict[str, Any]] = Field(default_factory=list, exclude=True)
    bound_tool_choice: Any = Field(default=None, exclude=True)
    bound_tool_kwargs: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "google-genai"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "GoogleGenAIChatModel":
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.model_copy(
            update={
                "bound_tools": openai_tools,
                "bound_tool_choice": tool_choice,
                "bound_tool_kwargs": dict(kwargs),
            }
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        from google import genai

        openai_messages = convert_to_openai_messages(messages)
        prompt_parts: list[str] = []
        for msg in openai_messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = _response_message_text(msg)
            if not content:
                continue
            if role == "assistant":
                label = "assistant"
            elif role == "system":
                label = "system"
            else:
                label = "user"
            prompt_parts.append(f"{label.title()}: {content}")
        prompt = "\n".join(prompt_parts)
        if stop:
            prompt += f"\n\nStop sequence(s): {', '.join(stop)}"

        client = genai.Client()
        config: Dict[str, Any] = {}
        if self.temperature is not None:
            config["temperature"] = float(self.temperature)
        if self.num_predict is not None:
            config["max_output_tokens"] = int(self.num_predict)

        if config:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        else:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

        response_text = ""
        if hasattr(response, "text"):
            response_text = str(response.text or "").strip()
        elif isinstance(response, dict):
            response_text = str((response.get("text") or "")).strip()

        usage = None
        if hasattr(response, "usage_metadata"):
            usage = getattr(response, "usage_metadata")
            if hasattr(usage, "dict"):
                usage = usage.dict()
        elif isinstance(response, dict):
            usage = response.get("usage")

        message = AIMessage(content=response_text)
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={"usage": usage, "model": self.model_name},
        )


def build_chat_model(
    *,
    model: str,
    temperature: Optional[float] = None,
    num_predict: Optional[int] = None,
    timeout: float = DEFAULT_DASHSCOPE_TIMEOUT_S,
    base_url: Optional[str] = None,
) -> Any:
    backend, resolved_model = resolve_chat_model_backend(model)

    if backend == "dashscope":
        return DashScopeChatModel(
            model_name=resolved_model,
            temperature=DEFAULT_DASHSCOPE_TEMPERATURE,
            num_predict=num_predict,
            timeout=timeout,
        )

    if backend == "google_genai":
        _normalize_gemini_api_key()
        return GoogleGenAIChatModel(
            model_name=resolved_model,
            temperature=temperature,
            num_predict=num_predict,
            timeout=timeout,
        )

    return LiteLLMChatModel(
        model_name=resolved_model,
        temperature=temperature,
        num_predict=num_predict,
        timeout=timeout,
        base_url=base_url,
    )


__all__ = [
    "LiteLLMChatModel",
    "chat_with_litellm",
    "DashScopeChatModel",
    "build_chat_model",
    "chat_with_dashscope",
    "dashscope_api_key",
    "dashscope_base_url",
    "dashscope_chat_completion",
    "is_qwen_chat_model",
    "normalize_qwen_chat_model_name",
    "resolve_chat_model_backend",
    "is_claude_chat_model",
    "is_gpt_chat_model",
]
