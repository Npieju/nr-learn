from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

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
            self.assertEqual(int(str(row["is_win"])), 1)


if __name__ == "__main__":
    unittest.main()