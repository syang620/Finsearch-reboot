from __future__ import annotations

import tempfile
import unittest

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class OrchestratorCheckpointPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_checkpointer_persists_between_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/orchestrator_checkpoint.sqlite"

            config_base = {
                "configurable": {
                    "thread_id": "run-123",
                    "checkpoint_ns": "",
                }
            }
            checkpoint1 = {
                "id": "cp-1",
                "ts": "2026-03-10T10:00:00.000000Z",
                "channel_values": {"foo": 1},
                "channel_versions": {},
            }
            checkpoint2 = {
                "id": "cp-2",
                "ts": "2026-03-10T10:00:01.000000Z",
                "channel_values": {"foo": 2},
                "channel_versions": {},
            }

            conn = aiosqlite.connect(db_path)
            saver = AsyncSqliteSaver(conn)
            try:
                await saver.aput(
                    {**config_base, "configurable": {**config_base["configurable"], "checkpoint_id": None}},
                    checkpoint1,
                    {"attempt": 1},
                    {},
                )
                await saver.aput(
                    {
                        **config_base,
                        "configurable": {
                            **config_base["configurable"],
                            "checkpoint_id": "cp-1",
                        },
                    },
                    checkpoint2,
                    {"attempt": 2},
                    {},
                )

                entries = [
                    entry async for entry in saver.alist({"configurable": {"thread_id": "run-123", "checkpoint_ns": ""}})
                ]
                self.assertEqual(len(entries), 2)
                self.assertEqual(entries[0].checkpoint["id"], "cp-2")

            finally:
                await conn.close()

            conn = aiosqlite.connect(db_path)
            saver = AsyncSqliteSaver(conn)
            try:
                loaded = await saver.aget_tuple(
                    {
                        "configurable": {
                            "thread_id": "run-123",
                            "checkpoint_ns": "",
                            "checkpoint_id": "cp-2",
                        }
                    }
                )
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded.checkpoint["id"], "cp-2")
                self.assertEqual(loaded.metadata.get("attempt"), 2)

                filtered = [
                    entry async for entry in saver.alist(
                        {"configurable": {"thread_id": "run-123", "checkpoint_ns": ""}},
                        filter={"attempt": 2},
                    )
                ]
                self.assertEqual(len(filtered), 1)
                self.assertEqual(filtered[0].checkpoint["id"], "cp-2")

                await saver.adelete_thread("run-123")
                after_delete = [
                    entry async for entry in saver.alist({"configurable": {"thread_id": "run-123", "checkpoint_ns": ""}})
                ]
                self.assertEqual(after_delete, [])
            finally:
                await conn.close()


if __name__ == "__main__":
    unittest.main()
