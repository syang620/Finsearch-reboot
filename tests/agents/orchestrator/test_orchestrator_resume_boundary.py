from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agents.orchestrator import OrchestrationRunNotFoundError
from agents.orchestrator import agent_orchestrator


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_resume_rejects_unknown_run_before_graph_invocation() -> None:
    with patch.object(
        agent_orchestrator,
        "_orchestrator_run_exists",
        new=AsyncMock(return_value=False),
    ), patch.object(
        agent_orchestrator,
        "_invoke_orchestrator",
        new=AsyncMock(),
    ) as invoke:
        with pytest.raises(
            OrchestrationRunNotFoundError, match="Run not found or expired"
        ):
            await agent_orchestrator.resume_multi_agent_orchestration(
                "run-missing", ["answer"]
            )

    invoke.assert_not_awaited()


@pytest.mark.anyio
async def test_resume_rejects_invalid_run_id_without_checkpoint_lookup() -> None:
    with patch.object(
        agent_orchestrator,
        "_orchestrator_run_exists",
        new=AsyncMock(),
    ) as exists:
        with pytest.raises(OrchestrationRunNotFoundError):
            await agent_orchestrator.resume_multi_agent_orchestration(
                "invalid run id", ["answer"]
            )

    exists.assert_not_awaited()
