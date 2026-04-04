from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from racing_ml.data.local_nankan_id_prep import _build_horse_key_frame
from racing_ml.data.local_nankan_primary import materialize_local_nankan_primary_from_config


class LocalNankanPrimaryMaterializeTest(unittest.TestCase):
    def test_materialize_keeps_jra_comparable_result_granularity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            result_path = base_dir / "data/external/local_nankan/results/local_race_result.csv"
            card_path = base_dir / "data/external/local_nankan/racecard/local_racecard.csv"
            pedigree_path = base_dir / "data/external/local_nankan/pedigree/local_pedigree.csv"
            output_path = base_dir / "data/local_nankan/raw/local_nankan_primary.csv"
            manifest_path = base_dir / "artifacts/reports/local_nankan_primary_materialize_manifest.json"
            result_keys_path = base_dir / "data/local_nankan/raw/local_nankan_race_result_keys.csv"
            race_card_output_path = base_dir / "data/local_nankan/raw/local_nankan_race_card.csv"
            pedigree_output_path = base_dir / "data/local_nankan/raw/local_nankan_pedigree.csv"

            result_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.parent.mkdir(parents=True, exist_ok=True)
            pedigree_path.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "race_id": "nar001",
                        "horse_id": "nar001:1",
                        "horse_key": "horse001",
                        "horse_name": "Alpha",
                        "rank": 1,
                        "odds": 2.5,
                        "popularity": 1,
                        "finish_time": "1:23.4",
                        "margin": "",
                        "closing_time_3f": 37.1,
                        "passing_order": "1-1-1-1",
                        "jockey_id": "j001",
                        "trainer_id": "t001",
                        "track": "大井",
                        "distance": 1400,
                        "weather": "晴",
                        "ground_condition": "良",
                        "sex": "牡",
                        "age": 3,
                        "carried_weight": 56.0,
                        "weight": 480,
                        "weight_change": -2,
                        "frame_no": 1,
                        "gate_no": 1,
                    }
                ]
            ).to_csv(result_path, index=False)

            pd.DataFrame(
                [
                    {
                        "race_id": "nar001",
                        "horse_id": "nar001:1",
                        "horse_key": "horse001",
                        "owner_name": "Owner A",
                        "breeder_name": "Breeder A",
                        "weight_change": -2,
                        "carried_weight": 56.0,
                        "popularity": 1,
                        "post_time": "15:20",
                        "scheduled_post_at": "2025-01-01T15:20:00+09:00",
                        "card_source_url": "https://www.nankankeiba.com/syousai/nar001.do",
                        "card_fetch_mode": "cache_manifest",
                        "card_snapshot_at": "2025-01-01T05:00:00Z",
                        "card_snapshot_relation": "pre_race",
                        "odds_source_url": "https://www.nankankeiba.com/oddsJS/nar001.do",
                        "odds_fetch_mode": "cache_manifest",
                        "odds_snapshot_at": "2025-01-01T05:05:00Z",
                        "odds_snapshot_relation": "pre_race",
                    }
                ]
            ).to_csv(card_path, index=False)

            pd.DataFrame(
                [
                    {
                        "horse_key": "horse001",
                        "sire_name": "Sire A",
                        "dam_name": "Dam A",
                        "damsire_name": "DamSire A",
                    }
                ]
            ).to_csv(pedigree_path, index=False)

            summary = materialize_local_nankan_primary_from_config(
                {"dataset": {"raw_dir": "data/local_nankan/raw"}},
                base_dir=base_dir,
                race_result_path=result_path,
                race_card_path=card_path,
                pedigree_path=pedigree_path,
                output_file=output_path,
                manifest_file=manifest_path,
                dry_run=False,
            )

            self.assertEqual(summary["status"], "completed")
            self.assertIn("generated_files", summary)

            frame = pd.read_csv(output_path)
            row = frame.iloc[0]
            self.assertEqual(frame.loc[0, "finish_time"], "1:23.4")
            self.assertEqual(float(str(row["closing_time_3f"])), 37.1)
            self.assertEqual(str(row["passing_order"]), "1-1-1-1")
            self.assertEqual(float(str(row["carried_weight"])), 56.0)
            self.assertEqual(int(str(row["weight_change"])), -2)
            self.assertEqual(str(row["owner_name"]), "Owner A")
            self.assertEqual(str(row["breeder_name"]), "Breeder A")
            self.assertEqual(str(row["sire_name"]), "Sire A")
            self.assertEqual(str(row["post_time"]), "15:20")
            self.assertEqual(str(row["card_fetch_mode"]), "cache_manifest")
            self.assertEqual(str(row["card_snapshot_relation"]), "pre_race")
            self.assertIn("oddsJS/nar001.do", str(row["odds_source_url"]))
            self.assertEqual(int(str(row["is_win"])), 1)

            result_keys = pd.read_csv(result_keys_path)
            race_card_output = pd.read_csv(race_card_output_path)
            pedigree_output = pd.read_csv(pedigree_output_path)

            self.assertEqual(str(result_keys.loc[0, "horse_key"]), "horse001")
            self.assertEqual(str(race_card_output.loc[0, "horse_id"]), "nar001:1")
            self.assertEqual(str(race_card_output.loc[0, "card_snapshot_relation"]), "pre_race")
            self.assertEqual(str(pedigree_output.loc[0, "sire_name"]), "Sire A")

    def test_materialize_falls_back_to_race_and_horse_key_for_non_runners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            result_path = base_dir / "data/external/local_nankan/results/local_race_result.csv"
            card_path = base_dir / "data/external/local_nankan/racecard/local_racecard.csv"
            output_path = base_dir / "data/local_nankan/raw/local_nankan_primary.csv"
            manifest_path = base_dir / "artifacts/reports/local_nankan_primary_materialize_manifest.json"

            result_path.parent.mkdir(parents=True, exist_ok=True)
            card_path.parent.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "race_id": "nar001",
                        "horse_id": "nar001:4",
                        "horse_key": "2000106224",
                        "horse_name": "Excluded Horse",
                        "rank": None,
                        "popularity": None,
                        "track": "大井",
                        "distance": 1400,
                    }
                ]
            ).to_csv(result_path, index=False)

            pd.DataFrame(
                [
                    {
                        "race_id": "nar001",
                        "horse_id": "nar001:除外",
                        "horse_key": "2000106224",
                        "horse_name": "Excluded Horse",
                        "owner_name": "Owner X",
                        "breeder_name": "Breeder X",
                        "odds": None,
                        "popularity": None,
                    }
                ]
            ).to_csv(card_path, index=False)

            summary = materialize_local_nankan_primary_from_config(
                {"dataset": {"raw_dir": "data/local_nankan/raw"}},
                base_dir=base_dir,
                race_result_path=result_path,
                race_card_path=card_path,
                output_file=output_path,
                manifest_file=manifest_path,
                dry_run=False,
            )

            self.assertEqual(summary["status"], "completed")
            frame = pd.read_csv(output_path)
            row = frame.iloc[0]
            self.assertEqual(str(row["owner_name"]), "Owner X")
            self.assertEqual(str(row["breeder_name"]), "Breeder X")
            self.assertEqual(float(row["odds"]), 0.0)


class LocalNankanHorseKeyPrepTest(unittest.TestCase):
    def test_build_horse_key_frame_excludes_non_numeric_seed_keys(self) -> None:
        seed_frame = pd.DataFrame(
            {
                "horse_key": ["LN_HORSE_0001", "LN_HORSE_0002", "2001109285"],
            }
        )
        target_output = Path("/tmp/nonexistent_local_nankan_race_result.csv")

        frame = _build_horse_key_frame(
            seed_frame,
            horse_key_column="horse_key",
            targets={
                "race_result": {"output_file": str(target_output)},
                "race_card": {"output_file": str(target_output)},
                "pedigree": {"output_file": str(target_output)},
            },
            base_dir=Path("/"),
            include_completed=False,
            limit=None,
        )

        self.assertEqual(frame["horse_key"].tolist(), ["2001109285"])


if __name__ == "__main__":
    unittest.main()
