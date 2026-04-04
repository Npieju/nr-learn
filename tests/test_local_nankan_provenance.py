from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.data.local_nankan_provenance import (
    MARKET_TIMING_BUCKET_COLUMN,
    RELATION_POST_RACE,
    RELATION_PRE_RACE,
    RELATION_UNKNOWN,
    annotate_market_timing_bucket,
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


if __name__ == "__main__":
    unittest.main()
