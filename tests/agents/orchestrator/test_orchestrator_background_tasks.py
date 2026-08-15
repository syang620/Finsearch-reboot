from __future__ import annotations

import asyncio
import gc
import unittest
import weakref
from unittest import mock

from agents.orchestrator import agent_orchestrator as orchestrator


class BackgroundTaskObservationTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _raise_checkpoint_error() -> None:
        raise RuntimeError("checkpoint prune failed")

    @staticmethod
    def _observe_with_signal(task: asyncio.Task[object]) -> asyncio.Event:
        callback_finished = asyncio.Event()
        orchestrator._BACKGROUND_TASKS.add(task)
        task.add_done_callback(orchestrator._observe_background_task)
        task.add_done_callback(lambda _: callback_finished.set())
        return callback_finished

    async def test_success_removes_task_without_logging(self) -> None:
        task = asyncio.create_task(asyncio.sleep(0, result="done"))
        callback_finished = self._observe_with_signal(task)

        with mock.patch.object(orchestrator.logger, "exception") as log_exception:
            self.assertEqual(await task, "done")
            await asyncio.wait_for(callback_finished.wait(), timeout=1)

        self.assertNotIn(task, orchestrator._BACKGROUND_TASKS)
        log_exception.assert_not_called()

    async def test_cancellation_is_consumed_without_logging(self) -> None:
        blocker = asyncio.Event()
        task = asyncio.create_task(blocker.wait())
        callback_finished = self._observe_with_signal(task)

        with mock.patch.object(orchestrator.logger, "exception") as log_exception:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(callback_finished.wait(), timeout=1)

        self.assertNotIn(task, orchestrator._BACKGROUND_TASKS)
        log_exception.assert_not_called()

    async def test_exception_is_logged_and_fully_consumed(self) -> None:
        loop = asyncio.get_running_loop()
        previous_exception_handler = loop.get_exception_handler()
        unhandled_contexts: list[dict[str, object]] = []
        loop.set_exception_handler(
            lambda _loop, context: unhandled_contexts.append(dict(context))
        )

        try:
            task = asyncio.create_task(self._raise_checkpoint_error())
            callback_finished = self._observe_with_signal(task)

            with mock.patch.object(orchestrator.logger, "exception") as log_exception:
                await asyncio.wait_for(callback_finished.wait(), timeout=1)

            self.assertNotIn(task, orchestrator._BACKGROUND_TASKS)
            log_exception.assert_called_once()

            task_reference = weakref.ref(task)
            del task
            for _ in range(3):
                await asyncio.sleep(0)
                gc.collect()
                if task_reference() is None:
                    break

            self.assertIsNone(task_reference())
            self.assertFalse(
                any(
                    context.get("message") == "Task exception was never retrieved"
                    for context in unhandled_contexts
                )
            )
        finally:
            loop.set_exception_handler(previous_exception_handler)


if __name__ == "__main__":
    unittest.main()
