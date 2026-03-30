from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from racing_ml.data.dataset_loader import (
    _normalize_decimal_series,
    _normalize_digit_series,
    _normalize_columns,
    _append_external_tables,
    _load_matching_table,
    _resolve_exact_candidate_usecols,
    _read_csv_tail,
    _restrict_table_to_join_keys,
    _select_table_columns,
    _sort_and_tail,
    load_training_table,
    materialize_supplemental_table,
)


class DatasetLoaderTailReadTest(unittest.TestCase):
    def test_normalize_columns_reuses_frame_when_already_normalized(self) -> None:
        frame = pd.DataFrame({"date": ["2025-01-01"], "race_id": [1]})

        normalized = _normalize_columns(frame)

        self.assertIs(normalized, frame)

    def test_sort_and_tail_reuses_frame_when_full_frame_already_range_indexed(self) -> None:
        frame = pd.DataFrame({"race_id": [1, 2], "value": [10, 20]})

        sorted_frame = _sort_and_tail(frame, None)

        self.assertIs(sorted_frame, frame)

    def test_read_csv_tail_returns_requested_tail_and_total_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=12, freq="D"),
                    "race_id": [f"r{i}" for i in range(12)],
                    "value": list(range(12)),
                }
            )
            frame.to_csv(csv_path, index=False)

            tail_frame, total_rows = _read_csv_tail(csv_path, tail_rows=5)

            self.assertEqual(total_rows, 12)
            self.assertEqual(len(tail_frame), 5)
            self.assertEqual(tail_frame["value"].tolist(), [7, 8, 9, 10, 11])

    def test_read_csv_tail_preserves_current_multi_overflow_behavior(self) -> None:
        chunks = [
            pd.DataFrame({"value": [0, 1, 2]}),
            pd.DataFrame({"value": [3, 4, 5, 6]}),
            pd.DataFrame({"value": [7, 8, 9]}),
        ]

        with mock.patch("racing_ml.data.dataset_loader.pd.read_csv", return_value=iter(chunks)):
            tail_frame, total_rows = _read_csv_tail(Path("dummy.csv"), tail_rows=5)

        self.assertEqual(total_rows, 10)
        self.assertEqual(tail_frame["value"].tolist(), [5, 6, 7, 8, 9])

    def test_materialize_supplemental_table_writes_corner_passing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            corner_path = raw_dir / "sample_corner_passing_order.csv"
            pd.DataFrame(
                {
                    "レースID": [101],
                    "1コーナー": ["(3,1)2"],
                    "2コーナー": ["1,2,3"],
                    "3コーナー": [None],
                    "4コーナー": ["3-2-1"],
                }
            ).to_csv(corner_path, index=False)

            dataset_config = {
                "supplemental_tables": [
                    {
                        "name": "corner_passing_order",
                        "pattern": "**/*corner_passing_order*.csv",
                        "materialized_file": "processed/corner_passing_order.csv",
                        "join_on": ["race_id", "gate_no"],
                        "required_columns": ["race_id", "gate_no"],
                        "keep_columns": ["race_id", "gate_no", "corner_1_position", "corner_2_position", "corner_4_position"],
                        "dedupe_on": ["race_id", "gate_no"],
                        "table_loader": "corner_passing_order",
                    }
                ]
            }

            summary = materialize_supplemental_table(
                raw_dir,
                table_name="corner_passing_order",
                dataset_config=dataset_config,
                base_dir=root,
            )

            self.assertEqual(summary["status"], "completed")
            output_path = root / "processed/corner_passing_order.csv"
            self.assertTrue(output_path.exists())
            materialized = pd.read_csv(output_path)
            self.assertEqual(materialized["gate_no"].tolist(), [1, 2, 3])
            self.assertEqual(materialized["corner_4_position"].tolist(), [3, 2, 1])
            self.assertEqual(summary["race_id_min"], "101")
            self.assertEqual(summary["race_id_max"], "101")

    def test_select_table_columns_reuses_frame_for_noop_selection(self) -> None:
        frame = pd.DataFrame({"race_id": [101], "horse_id": ["h1"], "horse_key": ["k1"]})

        selected = _select_table_columns(frame, keep_columns=["race_id", "horse_id", "horse_key"], join_on=["race_id", "horse_id"])

        self.assertIs(selected, frame)

    def test_restrict_table_to_join_keys_preserves_matching_single_key_rows(self) -> None:
        table = pd.DataFrame({"race_id": [101, 102, 103], "value": [1, 2, 3]})
        base = pd.DataFrame({"race_id": [101, 103]})

        restricted = _restrict_table_to_join_keys(table, base, ["race_id"])

        self.assertEqual(restricted["race_id"].tolist(), [101, 103])
        self.assertEqual(restricted["value"].tolist(), [1, 3])

    def test_restrict_table_to_join_keys_preserves_matching_composite_rows(self) -> None:
        table = pd.DataFrame(
            {
                "race_id": [101, 101, 102],
                "horse_id": ["h1", "h2", "h3"],
                "value": [1, 2, 3],
            }
        )
        base = pd.DataFrame(
            {
                "race_id": [101, 102],
                "horse_id": ["h2", "h3"],
            }
        )

        restricted = _restrict_table_to_join_keys(table, base, ["race_id", "horse_id"])

        self.assertEqual(restricted[["race_id", "horse_id"]].values.tolist(), [[101, "h2"], [102, "h3"]])
        self.assertEqual(restricted["value"].tolist(), [2, 3])

    def test_normalize_digit_series_reuses_numeric_values_without_string_extract(self) -> None:
        series = pd.Series([1400.0, 1800.0, None])

        normalized = _normalize_digit_series(series)

        self.assertEqual(normalized.tolist()[:2], [1400.0, 1800.0])
        self.assertTrue(pd.isna(normalized.iloc[2]))

    def test_normalize_decimal_series_reuses_numeric_values_without_string_extract(self) -> None:
        series = pd.Series([1.4, 36.8, None])

        normalized = _normalize_decimal_series(series)

        self.assertEqual(normalized.tolist()[:2], [1.4, 36.8])
        self.assertTrue(pd.isna(normalized.iloc[2]))

    def test_normalize_digit_series_salvages_parenthetical_weight_strings(self) -> None:
        series = pd.Series(["474(-2)", "504(0)", None])

        normalized = _normalize_digit_series(series)

        self.assertEqual(normalized.tolist()[:2], [474.0, 504.0])
        self.assertTrue(pd.isna(normalized.iloc[2]))

    def test_normalize_decimal_series_keeps_direct_numeric_strings(self) -> None:
        series = pd.Series(["36.8", "1.4", None])

        normalized = _normalize_decimal_series(series)

        self.assertEqual(normalized.tolist()[:2], [36.8, 1.4])
        self.assertTrue(pd.isna(normalized.iloc[2]))

    def test_load_training_table_prefers_materialized_supplemental_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            supplemental_dir = root / "external"
            supplemental_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "date": ["2025-01-01"],
                    "race_id": [101],
                    "horse_id": ["h1"],
                    "horse_name": ["horse-a"],
                    "track": ["tokyo"],
                    "distance": [1600],
                    "gate_no": [1],
                    "rank": [1],
                }
            ).to_csv(raw_dir / "primary.csv", index=False)
            pd.DataFrame(
                {
                    "レースID": [101],
                    "1コーナー": ["1"],
                    "2コーナー": ["1"],
                    "3コーナー": ["1"],
                    "4コーナー": ["1"],
                }
            ).to_csv(supplemental_dir / "sample_corner_passing_order.csv", index=False)

            materialized_dir = root / "processed"
            materialized_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "race_id": [101],
                    "gate_no": [1],
                    "corner_1_position": [7],
                    "corner_2_position": [8],
                    "corner_3_position": [9],
                    "corner_4_position": [10],
                }
            ).to_csv(materialized_dir / "corner_passing_order.csv", index=False)

            dataset_config = {
                "supplemental_tables": [
                    {
                        "name": "corner_passing_order",
                        "pattern": "**/*corner_passing_order*.csv",
                        "search_dirs": ["external"],
                        "materialized_file": "processed/corner_passing_order.csv",
                        "join_on": ["race_id", "gate_no"],
                        "required_columns": ["race_id", "gate_no"],
                        "keep_columns": [
                            "race_id",
                            "gate_no",
                            "corner_1_position",
                            "corner_2_position",
                            "corner_3_position",
                            "corner_4_position",
                        ],
                        "dedupe_on": ["race_id", "gate_no"],
                        "merge_mode": "fill_missing",
                        "table_loader": "corner_passing_order",
                    }
                ]
            }

            loaded = load_training_table(raw_dir, dataset_config=dataset_config, base_dir=root)

            self.assertEqual(int(loaded.loc[0, "corner_1_position"]), 7)
            self.assertEqual(int(loaded.loc[0, "corner_4_position"]), 10)

    def test_append_external_tables_keeps_range_index_after_dedupe(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "race_id": [101, 102],
                "horse_id": ["h1", "h2"],
                "value": [1, 2],
            }
        )
        append_frame = pd.DataFrame(
            {
                "date": ["2025-01-02", "2025-01-03"],
                "race_id": [102, 103],
                "horse_id": ["h2", "h3"],
                "value": [99, 3],
            }
        )
        dataset_config = {
            "append_tables": [
                {
                    "name": "netkeiba_race_result",
                    "pattern": "**/*netkeiba_race_result*.csv",
                    "search_dirs": ["external"],
                    "required_columns": ["date", "race_id", "horse_id"],
                    "keep_columns": ["date", "race_id", "horse_id", "value"],
                    "dedupe_on": ["race_id", "horse_id"],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch(
                "racing_ml.data.dataset_loader._load_matching_table",
                return_value=append_frame,
            ):
                combined = _append_external_tables(
                    frame,
                    raw_dir,
                    dataset_config,
                    root,
                    recent_date_floor=pd.Timestamp("2025-01-01"),
                    max_rows=None,
                )

        self.assertIsInstance(combined.index, pd.RangeIndex)
        self.assertEqual(combined.index.start, 0)
        self.assertEqual(combined.index.step, 1)
        self.assertEqual(
            combined[["race_id", "horse_id", "value"]].values.tolist(),
            [[101, "h1", 1], [102, "h2", 2], [103, "h3", 3]],
        )

    def test_load_matching_table_skips_raw_candidate_when_filename_range_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            candidate = raw_dir / "19860105-20210731_laptime.csv"
            pd.DataFrame({"race_id": [202101010101], "value": [1]}).to_csv(candidate, index=False)
            base_frame = pd.DataFrame({"race_id": [202506040102, 202509050805]})

            with mock.patch("racing_ml.data.dataset_loader._load_candidate_table") as loader:
                loaded = _load_matching_table(
                    table_cfg={"pattern": "**/*laptime*.csv"},
                    search_roots=[raw_dir],
                    join_on=["race_id"],
                    keep_columns=["race_id", "value"],
                    required_columns=["race_id"],
                    base_frame=base_frame,
                    base_dir=root,
                )

            self.assertIsNone(loaded)
            loader.assert_not_called()

    def test_load_matching_table_skips_materialized_candidate_when_manifest_range_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed_dir = root / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)
            artifacts_dir = root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            materialized = processed_dir / "corner_passing_order.csv"
            materialized.write_text("race_id,gate_no,corner_1_position\n202101010101,1,1\n", encoding="utf-8")
            manifest = artifacts_dir / "corner_manifest.json"
            manifest.write_text(
                '{"race_id_date_start":"20210101","race_id_date_end":"20210731"}',
                encoding="utf-8",
            )
            base_frame = pd.DataFrame({"race_id": [202506040102, 202509050805], "gate_no": [1, 2]})

            with mock.patch("racing_ml.data.dataset_loader._load_materialized_candidate_table") as loader:
                loaded = _load_matching_table(
                    table_cfg={
                        "materialized_file": "processed/corner_passing_order.csv",
                        "materialized_manifest_file": "artifacts/corner_manifest.json",
                        "pattern": "**/*corner_passing_order*.csv",
                    },
                    search_roots=[root],
                    join_on=["race_id", "gate_no"],
                    keep_columns=["race_id", "gate_no", "corner_1_position"],
                    required_columns=["race_id", "gate_no"],
                    base_frame=base_frame,
                    base_dir=root,
                )

            self.assertIsNone(loaded)
            loader.assert_not_called()

    def test_resolve_exact_candidate_usecols_preserves_header_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "sample.csv"
            candidate.write_text(
                "\ufeffレースID,horse_id,horse_key,jockey_key\n101,h1,k1,j1\n",
                encoding="utf-8",
            )

            usecols = _resolve_exact_candidate_usecols(
                candidate,
                {"column_aliases": {"race_id": ["レースID"]}},
                join_on=["race_id", "horse_id"],
                keep_columns=["horse_key"],
                required_columns=["race_id", "horse_id", "horse_key"],
            )

        self.assertEqual(usecols, ["レースID", "horse_id", "horse_key"])


if __name__ == "__main__":
    unittest.main()
