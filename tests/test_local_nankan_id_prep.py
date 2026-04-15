from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.data.local_nankan_id_prep import prepare_local_nankan_ids_from_config


class LocalNankanIdPrepTest(unittest.TestCase):
    def test_prepare_ids_uses_configured_race_list_default(self) -> None:
        crawl_config = {
            "crawl": {
                "race_id_source_default": "race_list",
                "targets": {
                    "race_card": {
                        "id_file": "data/external/local_nankan/ids/race_ids.csv",
                        "output_file": "data/external/local_nankan/racecard/local_racecard.csv",
                    },
                },
            }
        }
        discovered = pd.DataFrame(
            [
                {"date": "2026-04-14", "meeting_id": "20260414000001", "race_id": "2026041400000101", "race_no": 1},
                {"date": "2026-04-14", "meeting_id": "20260414000001", "race_id": "2026041400000102", "race_no": 2},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "racing_ml.data.local_nankan_id_prep.discover_local_nankan_race_ids_from_calendar",
            return_value=(discovered, {"source": "race_list", "row_count": 2}),
        ) as discover_mock:
            summary = prepare_local_nankan_ids_from_config(
                crawl_config,
                base_dir=Path(tmp_dir),
                target_filter="race_card",
                start_date="2026-04-14",
                end_date="2026-04-14",
            )

        self.assertEqual(summary["race_id_source"], "race_list")
        self.assertEqual(summary["race_id_source_default"], "race_list")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["current_phase"], "ids_prepared")
        self.assertEqual(summary["recommended_action"], "run_local_collect")
        self.assertEqual(summary["read_order"][0:7], [
            "status",
            "current_phase",
            "recommended_action",
            "race_id_source",
            "upcoming_only",
            "as_of",
            "reports[0].row_count",
        ])
        discover_mock.assert_called_once()
        self.assertEqual(summary["reports"][0]["row_count"], 2)

    def test_prepare_ids_surfaces_effective_as_of_from_race_list_report(self) -> None:
        crawl_config = {
            "crawl": {
                "race_id_source_default": "race_list",
                "targets": {
                    "race_card": {
                        "id_file": "data/external/local_nankan/ids/race_ids.csv",
                        "output_file": "data/external/local_nankan/racecard/local_racecard.csv",
                    },
                },
            }
        }
        discovered = pd.DataFrame(
            [
                {"date": "2026-04-14", "meeting_id": "20260414000001", "race_id": "2026041400000102", "race_no": 2},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir, patch(
            "racing_ml.data.local_nankan_id_prep.discover_local_nankan_race_ids_from_calendar",
            return_value=(
                discovered,
                {
                    "source": "race_list",
                    "row_count": 1,
                    "as_of": "2026-04-14T15:00:00+09:00",
                    "upcoming_only": True,
                    "pre_filter_row_count": 2,
                    "filtered_out_count": 1,
                },
            ),
        ):
            summary = prepare_local_nankan_ids_from_config(
                crawl_config,
                base_dir=Path(tmp_dir),
                target_filter="race_card",
                start_date="2026-04-14",
                end_date="2026-04-14",
                upcoming_only=True,
            )

        self.assertTrue(summary["upcoming_only"])
        self.assertEqual(summary["as_of"], "2026-04-14T15:00:00+09:00")
        self.assertEqual(summary["read_order"][-2:], [
            "race_id_source_report.pre_filter_row_count",
            "race_id_source_report.filtered_out_count",
        ])
        self.assertEqual(summary["race_id_source_report"]["pre_filter_row_count"], 2)
        self.assertEqual(summary["race_id_source_report"]["filtered_out_count"], 1)

    def test_prepare_ids_rejects_invalid_configured_default(self) -> None:
        crawl_config = {
            "crawl": {
                "race_id_source_default": "broken_source",
                "targets": {
                    "race_card": {
                        "id_file": "data/external/local_nankan/ids/race_ids.csv",
                        "output_file": "data/external/local_nankan/racecard/local_racecard.csv",
                    },
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "Unsupported race_id_source"):
                prepare_local_nankan_ids_from_config(
                    crawl_config,
                    base_dir=Path(tmp_dir),
                    target_filter="race_card",
                    start_date="2026-04-14",
                    end_date="2026-04-14",
                )


if __name__ == "__main__":
    unittest.main()
