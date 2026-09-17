from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from agents.orchestrator import (
    OrchestrationResumeValidationError,
    OrchestrationRunNotFoundError,
)
from finsearch_api.app import ApiConfig, create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _run_result(status: str = "completed") -> Dict[str, Any]:
    return {
        "run_id": "run-12345678",
        "route": "kb",
        "status": status,
        "ok": status in {"completed", "degraded"},
        "failure_stage": "none" if status != "failed" else "analyst",
        "lanes": {
            "kb": {
                "requested": True,
                "attempted": True,
                "status": "ok",
                "issues": [],
            },
            "structured_fact": {
                "requested": False,
                "attempted": False,
                "status": "not_requested",
                "issues": [],
            },
        },
        "degradation": {
            "active": status == "degraded",
            "affected_lanes": ["structured_fact"] if status == "degraded" else [],
            "notice": "Structured evidence is unavailable." if status == "degraded" else "",
        },
        "open_issues": [],
        "planner": {"status": "completed"},
        "planner_turn": {"status": "completed"},
        "retrieval": {"type": "retrieval", "ok": True},
        "structured_fact_results": [],
        "analyst": {"ok": status != "failed", "answer": "Answer"},
        "interrupt": (
            [{"value": {"kind": "planner_clarification"}}]
            if status == "interrupted"
            else []
        ),
        "orchestrator_trace": {"total_ms": 12},
    }


async def _asgi_request(
    app,
    method: str,
    path: str,
    *,
    payload: Any = None,
    headers: Optional[List[Tuple[bytes, bytes]]] = None,
):
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = list(headers or [])
    if payload is not None:
        request_headers.append((b"content-type", b"application/json"))
    request_headers.append((b"content-length", str(len(body)).encode("ascii")))

    messages = [{"type": "http.request", "body": body, "more_body": False}]
    sent = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": request_headers,
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )

    start = next(message for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    response_headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in start["headers"]
    }
    return start["status"], response_headers, json.loads(response_body or b"null")


@pytest.fixture
def api_config(tmp_path: Path) -> ApiConfig:
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()
    checkpoint_path = tmp_path / "checkpoints" / "orchestrator.sqlite"
    checkpoint_path.parent.mkdir()
    return ApiConfig(
        analyst_model="test-model",
        tables_dir=str(tables_dir),
        checkpoint_path=checkpoint_path,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["completed", "degraded", "interrupted", "failed"])
async def test_start_run_preserves_domain_outcome_and_server_controls(
    api_config: ApiConfig, outcome: str
) -> None:
    calls = []

    async def run_orchestration(**kwargs):
        calls.append(kwargs)
        return _run_result(outcome)

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        run_orchestration=run_orchestration,
        close_runtime=close_runtime,
    )
    status, headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs",
        payload={"query": "  What was Apple's revenue?  "},
        headers=[(b"x-request-id", b"request-123")],
    )

    assert status == 200
    assert headers["x-request-id"] == "request-123"
    assert body["request_id"] == "request-123"
    assert body["status"] == outcome
    assert calls == [
        {
            "user_query": "What was Apple's revenue?",
            "analyst_model": "test-model",
            "tables_dir": api_config.tables_dir,
            "debug": False,
            "include_evidence_trace": False,
        }
    ]


@pytest.mark.anyio
async def test_request_validation_uses_sanitized_error(api_config: ApiConfig) -> None:
    async def unused_run(**_kwargs):
        raise AssertionError("orchestration must not run")

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        run_orchestration=unused_run,
        close_runtime=close_runtime,
    )
    status, headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs",
        payload={"query": "   ", "unexpected": "secret"},
        headers=[(b"x-request-id", b"invalid request id")],
    )

    assert status == 422
    assert headers["x-request-id"].startswith("req-")
    assert body == {
        "request_id": headers["x-request-id"],
        "error": {
            "code": "invalid_request",
            "message": "Request validation failed.",
        },
    }
    assert "secret" not in json.dumps(body)


@pytest.mark.anyio
async def test_unexpected_error_is_sanitized(
    api_config: ApiConfig, caplog
) -> None:
    async def run_orchestration(**_kwargs):
        raise RuntimeError("provider secret")

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        run_orchestration=run_orchestration,
        close_runtime=close_runtime,
    )
    status, _headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs",
        payload={"query": "Question"},
    )

    assert status == 500
    assert body["error"] == {
        "code": "internal_error",
        "message": "The request could not be completed.",
    }
    assert "provider secret" not in json.dumps(body)
    assert "Unhandled FinSearch API error" in caplog.text


@pytest.mark.anyio
async def test_evaluation_trace_is_not_exposed(api_config: ApiConfig) -> None:
    async def run_orchestration(**_kwargs):
        return {**_run_result(), "evaluation_trace": {"private": "evidence"}}

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        run_orchestration=run_orchestration,
        close_runtime=close_runtime,
    )
    status, _headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs",
        payload={"query": "Question"},
    )

    assert status == 500
    assert body["error"]["code"] == "internal_error"
    assert "evidence" not in json.dumps(body)


@pytest.mark.anyio
async def test_resume_returns_typed_run_result(api_config: ApiConfig) -> None:
    calls = []

    async def resume_orchestration(run_id, answers):
        calls.append((run_id, answers))
        return _run_result()

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        resume_orchestration=resume_orchestration,
        close_runtime=close_runtime,
    )
    status, _headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs/run-12345678/resume",
        payload={"answers": ["  Apple, fiscal year 2024  "]},
    )

    assert status == 200
    assert body["run_id"] == "run-12345678"
    assert calls == [("run-12345678", ["Apple, fiscal year 2024"])]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (OrchestrationRunNotFoundError("private"), 404, "run_not_found"),
        (
            OrchestrationResumeValidationError("private"),
            400,
            "invalid_resume_answers",
        ),
    ],
)
async def test_resume_maps_expected_errors(
    api_config: ApiConfig,
    exception: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    async def resume_orchestration(_run_id, _answers):
        raise exception

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        resume_orchestration=resume_orchestration,
        close_runtime=close_runtime,
    )
    status, _headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs/run-12345678/resume",
        payload={"answers": ["Apple, fiscal year 2024"]},
    )

    assert status == expected_status
    assert body["error"]["code"] == expected_code
    assert "private" not in json.dumps(body)


@pytest.mark.anyio
async def test_invalid_run_id_returns_not_found_without_resume_call(
    api_config: ApiConfig,
) -> None:
    async def resume_orchestration(_run_id, _answers):
        raise AssertionError("resume must not run")

    async def close_runtime():
        return None

    app = create_app(
        config=api_config,
        resume_orchestration=resume_orchestration,
        close_runtime=close_runtime,
    )
    status, _headers, body = await _asgi_request(
        app,
        "POST",
        "/v1/runs/not%20valid/resume",
        payload={"answers": ["answer"]},
    )

    assert status == 404
    assert body["error"]["code"] == "run_not_found"


@pytest.mark.anyio
async def test_health_readiness_and_lifespan_cleanup(api_config: ApiConfig) -> None:
    closed = []

    async def close_runtime():
        closed.append(True)

    app = create_app(config=api_config, close_runtime=close_runtime)

    health_status, _headers, health = await _asgi_request(app, "GET", "/healthz")
    ready_status, _headers, ready = await _asgi_request(app, "GET", "/readyz")
    async with app.router.lifespan_context(app):
        pass

    assert health_status == 200
    assert health == {"status": "ok"}
    assert ready_status == 200
    assert ready == {"status": "ready"}
    assert closed == [True]


@pytest.mark.anyio
async def test_readiness_reports_missing_tables_dir(tmp_path: Path) -> None:
    async def close_runtime():
        return None

    app = create_app(
        config=ApiConfig(
            analyst_model="test-model",
            tables_dir=str(tmp_path / "missing"),
            checkpoint_path=tmp_path / "orchestrator.sqlite",
        ),
        close_runtime=close_runtime,
    )

    status, _headers, body = await _asgi_request(app, "GET", "/readyz")

    assert status == 503
    assert body["error"]["code"] == "tables_dir_unavailable"


def test_openapi_exposes_only_the_intended_routes(api_config: ApiConfig) -> None:
    async def close_runtime():
        return None

    app = create_app(config=api_config, close_runtime=close_runtime)
    paths = set(app.openapi()["paths"])

    assert paths == {
        "/healthz",
        "/readyz",
        "/v1/runs",
        "/v1/runs/{run_id}/resume",
    }
