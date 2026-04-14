from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import scripts.run_predict as predict_script


class RunPredictLocalNankanTrustGuardTest(unittest.TestCase):
    def test_historical_local_nankan_is_blocked_without_override(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_predict.py",
                "--config",
                "configs/model.yaml",
                "--data-config",
                "configs/data_local_nankan.yaml",
                "--feature-config",
                "configs/features.yaml",
                "--race-date",
                "2026-03-27",
            ],
        ), patch.object(
            predict_script,
            "load_yaml",
            return_value={"dataset": {"source_dataset": "local_nankan", "raw_dir": "data/local_nankan/raw"}},
        ), patch.object(
            predict_script,
            "require_local_nankan_trust_ready",
            side_effect=ValueError("blocked by trust guard"),
        ):
            exit_code = predict_script.main()

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()