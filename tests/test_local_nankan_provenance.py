from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.data.local_nankan_provenance import (
    MARKET_TIMING_BUCKET_COLUMN,
    RELATION_POST_RACE,
    RELATION_PRE_RACE,
    RELATION_UNKNOWN,
    annotate_market_timing_bucket,
    build_pre_race_capture_coverage_summary,
    build_pre_race_capture_date_coverage,
    build_pre_race_readiness_probe_summary,
    build_pre_race_only_materialization_summary,
    build_provenance_summary,
    filter_pre_race_only,
    filter_result_ready_pre_race_only,
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

    def test_filter_result_ready_pre_race_only_keeps_only_labeled_races(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "horse_id": "r1:1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "horse_id": "r2:1", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r3", "horse_id": "r3:1", "card_snapshot_relation": "unknown", "odds_snapshot_relation": "unknown"},
            ]
        )
        result_frame = pd.DataFrame([{"race_id": "r2"}])

        filtered = filter_result_ready_pre_race_only(frame, result_frame)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(str(filtered.loc[0, "race_id"]), "r2")

    def test_build_pre_race_capture_date_coverage_groups_by_date(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "date": "2026-04-07", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r3", "date": "2026-04-07", "card_snapshot_relation": "unknown", "odds_snapshot_relation": "unknown"},
            ]
        )
        result_frame = pd.DataFrame([{"race_id": "r1"}])

        coverage = build_pre_race_capture_date_coverage(frame, result_frame=result_frame)

        self.assertEqual(coverage["date"].tolist(), ["2026-04-06", "2026-04-07"])
        self.assertEqual(coverage["pre_race_rows"].tolist(), [2, 1])
        self.assertEqual(coverage["result_ready_races"].tolist(), [1, 0])
        self.assertEqual(coverage["pending_result_races"].tolist(), [0, 1])

    def test_build_pre_race_capture_coverage_summary_sets_capturing_phase_and_delta(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "date": "2026-04-07", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
            ]
        )
        result_frame = pd.DataFrame([{"race_id": "r1"}])
        previous_summary = {
            "materialization_summary": {
                "pre_race_only_rows": 2,
                "pre_race_only_races": 1,
                "pre_race_only_dates": ["2026-04-06"],
                "result_ready_races": 1,
                "pending_result_races": 0,
            }
        }

        summary = build_pre_race_capture_coverage_summary(
            frame,
            result_frame=result_frame,
            previous_summary=previous_summary,
        )

        self.assertEqual(summary["status"], "capturing")
        self.assertEqual(summary["current_phase"], "capturing_pre_race_pool")
        self.assertEqual(summary["recommended_action"], "continue_recrawl_cadence_and_wait_for_results")
        self.assertEqual(summary["pre_race_only_rows"], 3)
        self.assertEqual(summary["pre_race_only_races"], 2)
        self.assertEqual(summary["baseline_comparison"]["delta_pre_race_only_rows"], 1)
        self.assertEqual(summary["baseline_comparison"]["delta_pre_race_only_races"], 1)
        self.assertEqual(summary["baseline_comparison"]["added_dates"], ["2026-04-07"])
        self.assertEqual(summary["date_coverage"][0]["date"], "2026-04-06")

    def test_build_pre_race_readiness_probe_summary_marks_ready_when_any_ready_race_exists(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "date": "2026-04-06", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
                {"race_id": "r2", "date": "2026-04-07", "card_snapshot_relation": "pre_race", "odds_snapshot_relation": "pre_race"},
            ]
        )
        result_frame = pd.DataFrame([{"race_id": "r2"}])

        summary = build_pre_race_readiness_probe_summary(frame, result_frame=result_frame)

        self.assertEqual(summary["status"], "ready")
        self.assertEqual(summary["current_phase"], "ready_for_benchmark_handoff")
        self.assertEqual(summary["recommended_action"], "run_pre_race_benchmark_handoff")
        self.assertEqual(summary["materialization_summary"]["result_ready_races"], 1)


if __name__ == "__main__":
    unittest.main()
