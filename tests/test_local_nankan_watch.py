from __future__ import annotations

import unittest

from racing_ml.data.local_nankan_watch import (
    build_readiness_watcher_manifest,
    should_trigger_handoff,
)


class LocalNankanWatchTest(unittest.TestCase):
    def test_should_trigger_handoff_only_when_ready(self) -> None:
        self.assertTrue(should_trigger_handoff({"status": "ready"}))
        self.assertFalse(should_trigger_handoff({"status": "not_ready"}))
        self.assertFalse(should_trigger_handoff(None))

    def test_build_readiness_watcher_manifest_keeps_probe_and_handoff(self) -> None:
        manifest = build_readiness_watcher_manifest(
            status="completed",
            current_phase="handoff_completed",
            recommended_action="review_handoff_outputs",
            attempts=2,
            waited_seconds=60,
            timed_out=False,
            probe_summary={"status": "ready"},
            handoff_manifest={"status": "completed"},
        )

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["attempts"], 2)
        self.assertEqual(manifest["probe_summary"]["status"], "ready")
        self.assertEqual(manifest["handoff_manifest"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
