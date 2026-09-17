"""FastAPI application for the private single-instance FinSearch service."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, Dict, Optional
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from agents.orchestrator import (
    OrchestrationResumeValidationError,
    OrchestrationRunNotFoundError,
    aclose_orchestrator_runtime,
    resume_multi_agent_orchestration,
    run_multi_agent_orchestration,
)

from .schemas import ErrorResponse, HealthResponse, ResumeRequest, RunRequest, RunResponse


logger = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

RunCallable = Callable[..., Awaitable[Dict[str, Any]]]
ResumeCallable = Callable[..., Awaitable[Dict[str, Any]]]
CloseCallable = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class ApiConfig:
    analyst_model: str
    tables_dir: str
    checkpoint_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_config() -> ApiConfig:
    repo_root = _repo_root()
    checkpoint_path = Path(
        os.getenv(
            "FINSEARCH_ORCHESTRATOR_CHECKPOINTER_PATH",
            str(repo_root / "artifacts" / "orchestrator_checkpointer.sqlite"),
        )
    )
    return ApiConfig(
        analyst_model=(
            os.getenv("FINSEARCH_API_ANALYST_MODEL", "qwen2.5-14b-instruct-1m").strip()
            or "qwen2.5-14b-instruct-1m"
        ),
        tables_dir=os.getenv("TABLES_DIR", "data/chunked").strip() or "data/chunked",
        checkpoint_path=checkpoint_path,
    )


def _resolve_tables_dir(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _repo_root() / candidate


def _writable_parent(path: Path) -> bool:
    candidate = path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", "")
    return value if _REQUEST_ID_RE.fullmatch(value) else "request-id-unavailable"


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    payload = ErrorResponse(
        request_id=_request_id(request),
        error={"code": code, "message": message},
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


def create_app(
    *,
    config: Optional[ApiConfig] = None,
    run_orchestration: RunCallable = run_multi_agent_orchestration,
    resume_orchestration: ResumeCallable = resume_multi_agent_orchestration,
    close_runtime: CloseCallable = aclose_orchestrator_runtime,
) -> FastAPI:
    resolved_config = config or _load_config()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await close_runtime()

    api = FastAPI(
        title="FinSearch API",
        version="1.0.0",
        description="Private single-instance HTTP adapter for FinSearch.",
        lifespan=lifespan,
    )

    @api.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "").strip()
        request.state.request_id = (
            supplied
            if _REQUEST_ID_RE.fullmatch(supplied)
            else f"req-{uuid.uuid4().hex}"
        )
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.exception(
                "Unhandled FinSearch API error request_id=%s",
                _request_id(request),
                exc_info=exc,
            )
            response = _error_response(
                request,
                status_code=500,
                code="internal_error",
                message="The request could not be completed.",
            )
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            request,
            status_code=422,
            code="invalid_request",
            message="Request validation failed.",
        )

    @api.get("/healthz", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get(
        "/readyz",
        response_model=HealthResponse,
        responses={503: {"model": ErrorResponse}},
    )
    async def readiness(request: Request):
        if not _resolve_tables_dir(resolved_config.tables_dir).is_dir():
            return _error_response(
                request,
                status_code=503,
                code="tables_dir_unavailable",
                message="The configured tables directory is unavailable.",
            )
        if not _writable_parent(resolved_config.checkpoint_path):
            return _error_response(
                request,
                status_code=503,
                code="checkpoint_path_unavailable",
                message="The checkpoint path is unavailable.",
            )
        return HealthResponse(status="ready")

    @api.post(
        "/v1/runs",
        response_model=RunResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def start_run(request: Request, payload: RunRequest) -> RunResponse:
        result = await run_orchestration(
            user_query=payload.query,
            analyst_model=resolved_config.analyst_model,
            tables_dir=resolved_config.tables_dir,
            debug=False,
            include_evidence_trace=False,
        )
        return RunResponse.model_validate(
            {"request_id": _request_id(request), **result}
        )

    @api.post(
        "/v1/runs/{run_id}/resume",
        response_model=RunResponse,
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
        },
    )
    async def resume_run(
        request: Request, run_id: str, payload: ResumeRequest
    ):
        if not _RUN_ID_RE.fullmatch(run_id):
            return _error_response(
                request,
                status_code=404,
                code="run_not_found",
                message="Run not found or expired.",
            )
        try:
            result = await resume_orchestration(run_id, payload.answers)
        except OrchestrationRunNotFoundError:
            return _error_response(
                request,
                status_code=404,
                code="run_not_found",
                message="Run not found or expired.",
            )
        except OrchestrationResumeValidationError:
            return _error_response(
                request,
                status_code=400,
                code="invalid_resume_answers",
                message="Answers must match the pending clarification questions.",
            )
        return RunResponse.model_validate(
            {"request_id": _request_id(request), **result}
        )

    return api


app = create_app()
