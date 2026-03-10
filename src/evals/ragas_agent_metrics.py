from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_client import build_chat_model, is_qwen_chat_model


@dataclass(frozen=True)
class RagasAgentConfig:
    ollama_base_url: str = "http://localhost:11434"
    judge_model: str = "llama3.1:8b-instruct"
    embed_model: str = "nomic-embed-text"
    timeout_s: int = 120
    max_workers: int = 1
    max_retries: int = 2
    max_wait_s: int = 30
    batch_size: int = 1
    show_progress: bool = False
    allow_nest_asyncio: bool = True
    preflight_check_models: bool = True

    enable_answer_relevancy: bool = True
    enable_faithfulness: bool = True
    enable_context_precision: bool = False
    enable_context_recall: bool = False


def _metric_alias(column_name: str) -> str | None:
    name = str(column_name).lower()
    if "answer" in name and ("relev" in name or "similar" in name):
        return "answer_relevancy"
    if "faith" in name:
        return "faithfulness"
    if "context" in name and "precision" in name:
        return "context_precision"
    if "context" in name and "recall" in name:
        return "context_recall"
    return None


def _resolve_ragas_agent_metrics(
    *,
    enable_answer_relevancy: bool,
    enable_faithfulness: bool,
    enable_context_precision: bool,
    enable_context_recall: bool,
) -> List[Tuple[Any, str]]:
    import ragas.metrics as rm

    resolved: List[Tuple[Any, str]] = []

    def _add_metric(candidate: Any, alias: str) -> None:
        metric = candidate
        if isinstance(candidate, type):
            try:
                metric = candidate()
            except Exception:
                return
        resolved.append((metric, alias))

    for attr, alias, enabled in (
        ("answer_relevancy", "answer_relevancy", enable_answer_relevancy),
        ("faithfulness", "faithfulness", enable_faithfulness),
        ("context_precision", "context_precision", enable_context_precision),
        ("context_recall", "context_recall", enable_context_recall),
    ):
        if not enabled:
            continue
        obj = getattr(rm, attr, None)
        if obj is not None:
            _add_metric(obj, alias)

    if not resolved:
        for cls_name, alias, enabled in (
            ("AnswerRelevancy", "answer_relevancy", enable_answer_relevancy),
            ("ResponseRelevancy", "answer_relevancy", enable_answer_relevancy),
            ("Faithfulness", "faithfulness", enable_faithfulness),
            ("LLMContextPrecisionWithReference", "context_precision", enable_context_precision),
            ("ContextPrecision", "context_precision", enable_context_precision),
            ("LLMContextRecall", "context_recall", enable_context_recall),
            ("ContextRecall", "context_recall", enable_context_recall),
        ):
            if not enabled:
                continue
            cls_obj = getattr(rm, cls_name, None)
            if cls_obj is not None:
                _add_metric(cls_obj, alias)

    return resolved


def _build_dataset(samples: Sequence[Dict[str, Any]]) -> Any:
    from datasets import Dataset

    questions: List[str] = []
    responses: List[str] = []
    contexts: List[List[str]] = []
    references: List[str] = []

    for s in samples:
        q = str(s.get("question") or "").strip()
        answer = str(s.get("answer") or "").strip()
        ref = str(s.get("reference") or "").strip()
        ctx = [str(x) for x in (s.get("contexts") or []) if str(x).strip()]

        questions.append(q)
        responses.append(answer)
        references.append(ref)
        contexts.append(ctx)

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": responses,
            "contexts": contexts,
            "ground_truth": references,
            "user_input": questions,
            "response": responses,
            "retrieved_contexts": contexts,
            "reference": references,
        }
    )


def _wrap_langchain_for_ragas(llm: Any, embeddings: Any) -> Tuple[Any, Any]:
    wrapped_llm = llm
    wrapped_embeddings = embeddings

    try:
        from ragas.llms import LangchainLLMWrapper

        wrapped_llm = LangchainLLMWrapper(llm)
    except Exception:
        pass

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        pass

    return wrapped_llm, wrapped_embeddings


def _pick_metric_column(df: Any, alias: str) -> Optional[str]:
    for col in df.columns:
        if _metric_alias(str(col)) == alias:
            return str(col)
    if alias in df.columns:
        return alias
    return None


def _normalize_model_name(name: str) -> str:
    return str(name or "").strip()


def _ollama_has_model(*, base_url: str, model_name: str) -> Tuple[bool, List[str], Optional[str]]:
    try:
        import requests

        url = f"{str(base_url).rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return False, [], f"Failed to query Ollama tags: {exc}"

    models = data.get("models") or []
    names: List[str] = []
    for m in models:
        if isinstance(m, dict):
            n = m.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())

    wanted = _normalize_model_name(model_name)
    if wanted in names:
        return True, names, None

    wanted_base = wanted.split(":", 1)[0]
    for n in names:
        if n.split(":", 1)[0] == wanted_base:
            return True, names, None

    return False, names, None


def _evaluate_single_metric(
    *,
    evaluate_fn: Any,
    dataset: Any,
    metric: Any,
    alias: str,
    llm: Any,
    embeddings: Any,
    timeout_s: int,
    max_workers: int,
    max_retries: int,
    max_wait_s: int,
    batch_size: int,
    show_progress: bool,
    allow_nest_asyncio: bool,
) -> Tuple[Dict[int, float], Optional[float], Optional[str]]:
    eval_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": [metric],
        "llm": llm,
        "embeddings": embeddings,
        "show_progress": show_progress,
        "batch_size": batch_size,
        "allow_nest_asyncio": allow_nest_asyncio,
    }

    try:
        from ragas.run_config import RunConfig

        eval_kwargs["run_config"] = RunConfig(
            timeout=timeout_s,
            max_retries=max_retries,
            max_wait=max_wait_s,
            max_workers=max_workers,
        )
    except Exception:
        pass

    try:
        result = evaluate_fn(**eval_kwargs, raise_exceptions=False)
    except TypeError:
        result = evaluate_fn(**eval_kwargs)
    except Exception as exc:
        return {}, None, str(exc)

    try:
        df = result.to_pandas() if hasattr(result, "to_pandas") else None
    except Exception as exc:
        return {}, None, f"Failed to parse ragas output: {exc}"

    if df is None:
        return {}, None, "Ragas result did not expose to_pandas()."

    metric_col = _pick_metric_column(df, alias)
    if metric_col is None:
        return {}, None, f"No output column found for metric alias '{alias}'."

    per_row: Dict[int, float] = {}
    vals: List[float] = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        value = row.get(metric_col)
        if value is None:
            continue
        try:
            fv = float(value)
        except Exception:
            continue
        per_row[idx] = fv
        vals.append(fv)

    metric_mean = (sum(vals) / float(len(vals))) if vals else None
    return per_row, metric_mean, None


def evaluate_ragas_agents(
    samples: Sequence[Dict[str, Any]],
    *,
    config: RagasAgentConfig,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], List[Dict[str, Any]]]:
    if not samples:
        return {}, {}, []

    errors: List[Dict[str, Any]] = []

    try:
        from langchain_ollama import OllamaEmbeddings
        from ragas import evaluate
    except Exception as exc:
        errors.append({"stage": "ragas_init", "error": f"Failed to import ragas stack: {exc}"})
        return {}, {}, errors

    if config.preflight_check_models:
        if not is_qwen_chat_model(config.judge_model):
            has_judge, available, judge_err = _ollama_has_model(
                base_url=config.ollama_base_url,
                model_name=config.judge_model,
            )
            if judge_err:
                errors.append({"stage": "ragas_init", "error": judge_err})
                return {}, {}, errors
            if not has_judge:
                avail = ", ".join(available[:20]) if available else "<none>"
                errors.append(
                    {
                        "stage": "ragas_init",
                        "error": (
                            f"Ollama judge model '{config.judge_model}' not found. "
                            f"Available models: {avail}"
                        ),
                    }
                )
                return {}, {}, errors

        has_embed, available2, embed_err = _ollama_has_model(
            base_url=config.ollama_base_url,
            model_name=config.embed_model,
        )
        if embed_err:
            errors.append({"stage": "ragas_init", "error": embed_err})
            return {}, {}, errors
        if not has_embed:
            avail = ", ".join(available2[:20]) if available2 else "<none>"
            errors.append(
                {
                    "stage": "ragas_init",
                    "error": (
                        f"Ollama embed model '{config.embed_model}' not found. "
                        f"Available models: {avail}"
                    ),
                }
            )
            return {}, {}, errors

    metrics_with_alias = _resolve_ragas_agent_metrics(
        enable_answer_relevancy=config.enable_answer_relevancy,
        enable_faithfulness=config.enable_faithfulness,
        enable_context_precision=config.enable_context_precision,
        enable_context_recall=config.enable_context_recall,
    )
    if not metrics_with_alias:
        errors.append(
            {
                "stage": "ragas_init",
                "error": "No compatible ragas metrics found for agent evaluation.",
            }
        )
        return {}, {}, errors

    llm = build_chat_model(
        model=config.judge_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )
    embeddings = OllamaEmbeddings(model=config.embed_model, base_url=config.ollama_base_url)
    ragas_llm, ragas_embeddings = _wrap_langchain_for_ragas(llm, embeddings)

    dataset = _build_dataset(samples)

    per_sample: Dict[str, Dict[str, float]] = {}
    for sample in samples:
        sid = str(sample.get("id"))
        per_sample[sid] = {}

    summary: Dict[str, float] = {}

    for metric, alias in metrics_with_alias:
        per_row, metric_mean, metric_error = _evaluate_single_metric(
            evaluate_fn=evaluate,
            dataset=dataset,
            metric=metric,
            alias=alias,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            timeout_s=config.timeout_s,
            max_workers=max(1, int(config.max_workers)),
            max_retries=max(0, int(config.max_retries)),
            max_wait_s=max(1, int(config.max_wait_s)),
            batch_size=max(1, int(config.batch_size)),
            show_progress=bool(config.show_progress),
            allow_nest_asyncio=bool(config.allow_nest_asyncio),
        )
        if metric_error:
            errors.append({"stage": f"ragas_eval:{alias}", "error": metric_error})
            continue

        if metric_mean is not None:
            summary[alias] = metric_mean
        for idx, value in per_row.items():
            if 0 <= idx < len(samples):
                sid = str(samples[idx].get("id"))
                per_sample.setdefault(sid, {})[alias] = value

    return per_sample, summary, errors
