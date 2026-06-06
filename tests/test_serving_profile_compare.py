from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_serving_profile_compare as profile_compare_script


class ServingProfileCompareCommandTest(unittest.TestCase):
    def test_smoke_command_passes_market_refresh_options(self) -> None:
        command = profile_compare_script._smoke_command(
            profile_name="current_recommended_serving",
            dates=["2025-12-28"],
            prediction_backend="replay-existing",
            artifact_suffix="market_refresh_probe",
            model_artifact_suffix=None,
            output_file=Path("artifacts/reports/serving_smoke_market_refresh_probe.json"),
            market_file=Path("artifacts/tmp/replay_market_refresh_probe_market.csv"),
            market_join_keys=["race_id", "gate_no"],
            market_columns=["odds", "popularity"],
        )

        self.assertIn("--market-file", command)
        self.assertIn("artifacts/tmp/replay_market_refresh_probe_market.csv", command)
        self.assertIn("--market-join-keys", command)
        self.assertIn("race_id,gate_no", command)
        self.assertIn("--market-columns", command)
        self.assertIn("odds,popularity", command)


if __name__ == "__main__":
    unittest.main()