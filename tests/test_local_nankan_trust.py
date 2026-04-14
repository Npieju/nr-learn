from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from racing_ml.common.local_nankan_trust import require_local_nankan_trust_ready, resolve_local_nankan_trust_block


class LocalNankanTrustGuardTest(unittest.TestCase):
    def test_historical_local_nankan_blocks_when_strict_trust_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "artifacts/reports/local_nankan_provenance_audit.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "readiness": {
                            "strict_trust_ready": False,
                            "recommended_action": "rerun_local_backfill_with_pre_race_market_capture",
                        },
                        "provenance_summary": {
                            "bucket_counts": {"pre_race": 0, "post_race": 728850, "unknown": 257}
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                require_local_nankan_trust_ready(
                    workspace_root=root,
                    data_config={"dataset": {"source_dataset": "local_nankan"}},
                    data_config_path="configs/data_local_nankan.yaml",
                    allow_diagnostic_override=False,
                    command_name="evaluate",
                    profile_name="local_nankan_recommended",
                )

    def test_future_only_pre_race_ready_is_not_blocked(self) -> None:
        require_local_nankan_trust_ready(
            workspace_root=Path.cwd(),
            data_config={"dataset": {"source_dataset": "local_nankan_pre_race_ready"}},
            data_config_path="configs/data_local_nankan_pre_race_ready.yaml",
            allow_diagnostic_override=False,
            command_name="evaluate",
        )

    def test_override_allows_historical_local_nankan(self) -> None:
        require_local_nankan_trust_ready(
            workspace_root=Path.cwd(),
            data_config={"dataset": {"source_dataset": "local_nankan"}},
            data_config_path="configs/data_local_nankan.yaml",
            allow_diagnostic_override=True,
            command_name="evaluate",
        )

    def test_resolve_block_returns_structured_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "artifacts/reports/local_nankan_provenance_audit.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "readiness": {
                            "strict_trust_ready": False,
                            "recommended_action": "rerun_local_backfill_with_pre_race_market_capture",
                        },
                        "provenance_summary": {
                            "bucket_counts": {"pre_race": 0, "post_race": 728850, "unknown": 257}
                        },
                    }
                ),
                encoding="utf-8",
            )

            block = resolve_local_nankan_trust_block(
                workspace_root=root,
                data_config={"dataset": {"source_dataset": "local_nankan"}},
                data_config_path="configs/data_local_nankan.yaml",
                allow_diagnostic_override=False,
                command_name="evaluate",
                profile_name="local_nankan_recommended",
            )

        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["error_code"], "historical_local_nankan_trust_not_ready")
        self.assertEqual(block["pre_race"], 0)
        self.assertEqual(block["post_race"], 728850)
        self.assertEqual(block["unknown"], 257)
        self.assertEqual(block["recommended_action"], "rerun_local_backfill_with_pre_race_market_capture")

    def test_legacy_issue_snapshot_fallback_is_used_when_current_alias_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "artifacts/reports/local_nankan_provenance_audit_issue120_repaired.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "readiness": {
                            "strict_trust_ready": False,
                            "recommended_action": "rerun_local_backfill_with_pre_race_market_capture",
                        },
                        "provenance_summary": {
                            "bucket_counts": {"pre_race": 0, "post_race": 111, "unknown": 2}
                        },
                    }
                ),
                encoding="utf-8",
            )

            block = resolve_local_nankan_trust_block(
                workspace_root=root,
                data_config={"dataset": {"source_dataset": "local_nankan"}},
                data_config_path="configs/data_local_nankan.yaml",
                allow_diagnostic_override=False,
                command_name="evaluate",
            )

        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["post_race"], 111)
        self.assertTrue(str(block["provenance_manifest_path"]).endswith("local_nankan_provenance_audit_issue120_repaired.json"))

    def test_source_timing_context_blocks_even_when_provenance_is_strict_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reports_dir = root / "artifacts/reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "local_nankan_provenance_audit.json").write_text(
                json.dumps(
                    {
                        "readiness": {
                            "strict_trust_ready": True,
                            "recommended_action": "inspect_local_nankan_provenance_audit",
                        },
                        "provenance_summary": {
                            "bucket_counts": {"pre_race": 10, "post_race": 0, "unknown": 0}
                        },
                    }
                ),
                encoding="utf-8",
            )
            (reports_dir / "local_nankan_source_timing_audit.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "current_phase": "future_only_pre_race_capture_available",
                        "recommended_action": "downgrade_historical_benchmark_to_diagnostic_only",
                        "historical_pre_race_recoverability": {
                            "status": "future_only_pre_race_capture_available",
                            "result_ready_pre_race_rows": 0,
                            "future_only_pre_race_rows": 562,
                        },
                    }
                ),
                encoding="utf-8",
            )

            block = resolve_local_nankan_trust_block(
                workspace_root=root,
                data_config={"dataset": {"source_dataset": "local_nankan"}},
                data_config_path="configs/data_local_nankan.yaml",
                allow_diagnostic_override=False,
                command_name="evaluate",
            )

        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["error_code"], "historical_local_nankan_source_timing_not_ready")
        self.assertEqual(block["historical_source_timing_status"], "future_only_pre_race_capture_available")
        self.assertEqual(block["result_ready_pre_race_rows"], 0)
        self.assertEqual(block["future_only_pre_race_rows"], 562)


if __name__ == "__main__":
    unittest.main()