import sys
import types as py_types

from langchain_core.messages import HumanMessage

from llm_client import GoogleGenAIChatModel, build_chat_model


def test_build_chat_model_defaults_gemini_temperature_to_zero() -> None:
    model = build_chat_model(model="gemini-3.1-pro-preview")

    assert isinstance(model, GoogleGenAIChatModel)
    assert model.temperature == 0.0


def test_google_genai_chat_model_uses_generate_content_config(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            captured["config_kwargs"] = dict(kwargs)
            self.kwargs = dict(kwargs)

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return py_types.SimpleNamespace(text="stub response", usage_metadata=None)

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    fake_genai_module = py_types.ModuleType("google.genai")
    fake_genai_module.Client = FakeClient
    fake_genai_module.types = py_types.SimpleNamespace(GenerateContentConfig=FakeGenerateContentConfig)

    fake_google_module = py_types.ModuleType("google")
    fake_google_module.genai = fake_genai_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)

    model = GoogleGenAIChatModel(model_name="gemini-3.1-pro-preview", temperature=None, num_predict=123)
    result = model._generate([HumanMessage(content="Hello from Gemini")])

    assert result.generations[0].message.content == "stub response"
    assert captured["model"] == "gemini-3.1-pro-preview"
    assert captured["config_kwargs"] == {"temperature": 0.0, "max_output_tokens": 123}
    assert isinstance(captured["config"], FakeGenerateContentConfig)
    assert captured["contents"] == "User: Hello from Gemini"
