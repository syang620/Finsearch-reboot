"""Evaluation utilities for retrieval and agentic RAG."""


def run_agent_eval(*args, **kwargs):
    from .agent_eval_runner import run_agent_eval as _run_agent_eval

    return _run_agent_eval(*args, **kwargs)


def run_retrieval_eval(*args, **kwargs):
    from .retrieval_eval_runner import run_retrieval_eval as _run_retrieval_eval

    return _run_retrieval_eval(*args, **kwargs)


def run_llm_judge_eval(*args, **kwargs):
    from .llm_judge_runner import run_llm_judge_eval as _run_llm_judge_eval

    return _run_llm_judge_eval(*args, **kwargs)


__all__ = ["run_retrieval_eval", "run_agent_eval", "run_llm_judge_eval"]
