from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.probability import diagnose_race_probabilities


class RaceProbabilityDiagnosticsTest(unittest.TestCase):
    def test_accepts_coherent_race_probabilities(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2", "r2"],
                "probability": [0.7, 0.3, 0.2, 0.3, 0.5],
            }
        )

        result = diagnose_race_probabilities(frame, "probability")

        self.assertTrue(result.probability_contract_ok)
        self.assertEqual(result.race_count, 2)
        self.assertEqual(result.race_sum_violation_count, 0)
        self.assertAlmostEqual(result.race_sum_mean or 0.0, 1.0)

    def test_rejects_non_unit_race_sums(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2"],
                "probability": [0.4, 0.3, 0.8, 0.4],
            }
        )

        result = diagnose_race_probabilities(frame, "probability")

        self.assertFalse(result.probability_contract_ok)
        self.assertEqual(result.race_sum_violation_count, 2)
        self.assertAlmostEqual(result.race_sum_min or 0.0, 0.7)
        self.assertAlmostEqual(result.race_sum_max or 0.0, 1.2)

    def test_rejects_non_finite_or_out_of_range_values(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": ["r1", "r1", "r2", "r2"],
                "probability": [0.5, np.nan, 1.2, -0.2],
            }
        )

        result = diagnose_race_probabilities(frame, "probability")

        self.assertFalse(result.probability_contract_ok)
        self.assertEqual(result.invalid_value_count, 3)
        self.assertEqual(result.valid_race_count, 0)


if __name__ == "__main__":
    unittest.main()
