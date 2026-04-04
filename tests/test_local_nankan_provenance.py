from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.data.local_nankan_provenance import (
    MARKET_TIMING_BUCKET_COLUMN,
    RELATION_POST_RACE,
    RELATION_PRE_RACE,
    RELATION_UNKNOWN,
    annotate_market_timing_bucket,
    build_pre_race_only_materialization_summary,
    build_provenance_summary,
    filter_pre_race_only,
)


class LocalNankanProvenanceTest(unittest.TestCase):
    def test_annotate_market_timing_bucket_is_conservative(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "unknown"},
                {"race_id": "r3", "card_snapshot_relation": "unknown", "odds_snapshot_relation": "post_race"},
            ]
        )

        annotated = annotate_market_timing_bucket(frame)

        self.assertEqual(annotated.loc[0, MARKET_TIMING_BUCKET_COLUMN], RELATION_PRE_RACE)
        self.assertEqual(annotated.loc[1, MARKET_TIMING_BUCKET_COLUMN], RELATION_UNKNOWN)
        self.assertEqual(annotated.loc[2, MARKET_TIMING_BUCKET_COLUMN], RELATION_POST_RACE)

    def test_filter_pre_race_only_requires_both_card_and_odds(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "horse_id": "r1:1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "horse_id": "r2:1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "unknown"},
            ]
        )

        filtered = filter_pre_race_only(frame)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(str(filtered.loc[0, "horse_id"]), "r1:1")

    def test_build_provenance_summary_counts_buckets(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "card_snapshot_relation": "unknown", "odds_snapshot_relation": "unknown"},
                {"race_id": "r3", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "post_race"},
            ]
        )

        summary = build_provenance_summary(frame)

        self.assertEqual(summary["row_count"], 4)
        self.assertEqual(summary["race_count"], 3)
        self.assertEqual(summary["bucket_counts"][RELATION_PRE_RACE], 2)
        self.assertEqual(summary["bucket_counts"][RELATION_UNKNOWN], 1)
        self.assertEqual(summary["bucket_counts"][RELATION_POST_RACE], 1)

    def test_build_pre_race_only_materialization_summary_reports_label_readiness(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "date": "2026-04-07", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r3", "date": "2026-04-07", "card_snapshot_relation": "unknown", "odds_snapshot_relation": "unknown"},
            ]
        )
        result_frame = pd.DataFrame([{"race_id": "r1"}])

        summary = build_pre_race_only_materialization_summary(frame, result_frame=result_frame)

        self.assertEqual(summary["pre_race_only_rows"], 3)
        self.assertEqual(summary["pre_race_only_races"], 2)
        self.assertEqual(summary["pre_race_only_dates"], ["2026-04-06", "2026-04-07"])
        self.assertEqual(summary["result_ready_races"], 1)
        self.assertEqual(summary["pending_result_races"], 1)
        self.assertEqual(summary["result_ready_rows"], 2)
        self.assertEqual(summary["pending_result_rows"], 1)
        self.assertFalse(summary["ready_for_benchmark_rerun"])


if __name__ == "__main__":
    unittest.main()
