from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.evaluation.scoring import compose_value_blend_probabilities


class ComposeValueBlendProbabilitiesTest(unittest.TestCase):
    def test_market_aware_alpha_branch_uses_market_logit_path(self) -> None:
        win_prob = np.array([0.70, 0.25], dtype=float)
        market_prob = np.array([0.55, 0.35], dtype=float)
        alpha_raw = np.array([0.40, -0.60], dtype=float)
        params = {
            "probability_path_mode": "market_aware_alpha_branch",
            "alpha_weight": 0.05,
            "alpha_scale": 0.5,
            "alpha_positive_only": True,
            "market_blend_weight": 0.97,
        }

        actual = compose_value_blend_probabilities(
            win_prob=win_prob,
            params=params,
            alpha_raw=alpha_raw,
            market_prob=market_prob,
        )

        alpha_signal = np.tanh(alpha_raw / 0.5)
        alpha_signal = np.maximum(alpha_signal, 0.0)
        expected_logit = ((1.0 - 0.97) * np.log(win_prob / (1.0 - win_prob))) + (
            0.97 * (np.log(market_prob / (1.0 - market_prob)) + (0.05 * alpha_signal))
        )
        expected = 1.0 / (1.0 + np.exp(-expected_logit))

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

    def test_market_aware_alpha_branch_skips_legacy_probability_blend(self) -> None:
        win_prob = np.array([0.65], dtype=float)
        market_prob = np.array([0.40], dtype=float)
        alpha_raw = np.array([0.75], dtype=float)
        params = {
            "probability_path_mode": "market_aware_alpha_branch",
            "alpha_weight": 0.05,
            "alpha_scale": 0.5,
            "market_blend_weight": 0.97,
        }

        actual = compose_value_blend_probabilities(
            win_prob=win_prob,
            params=params,
            alpha_raw=alpha_raw,
            market_prob=market_prob,
        )
        legacy = compose_value_blend_probabilities(
            win_prob=win_prob,
            params={
                "alpha_weight": 0.05,
                "alpha_scale": 0.5,
                "market_blend_weight": 0.97,
            },
            alpha_raw=alpha_raw,
            market_prob=market_prob,
        )

        self.assertFalse(np.allclose(actual, legacy))

    def test_support_preserving_residual_path_keeps_win_logit_as_base(self) -> None:
        win_prob = np.array([0.70, 0.25], dtype=float)
        market_prob = np.array([0.55, 0.35], dtype=float)
        alpha_raw = np.array([0.40, -0.60], dtype=float)
        params = {
            "probability_path_mode": "support_preserving_residual_path",
            "alpha_weight": 0.05,
            "alpha_scale": 0.5,
            "alpha_positive_only": True,
            "market_residual_weight": 0.08,
            "market_residual_scale": 0.75,
        }

        actual = compose_value_blend_probabilities(
            win_prob=win_prob,
            params=params,
            alpha_raw=alpha_raw,
            market_prob=market_prob,
        )

        win_logit = np.log(win_prob / (1.0 - win_prob))
        market_logit = np.log(market_prob / (1.0 - market_prob))
        alpha_signal = np.tanh(alpha_raw / 0.5)
        alpha_signal = np.maximum(alpha_signal, 0.0)
        market_signal = np.tanh((market_logit - win_logit) / 0.75)
        expected_logit = win_logit + (0.08 * market_signal) + (0.05 * alpha_signal)
        expected = 1.0 / (1.0 + np.exp(-expected_logit))

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

    def test_support_preserving_residual_path_skips_legacy_probability_blend(self) -> None:
        win_prob = np.array([0.65], dtype=float)
        market_prob = np.array([0.40], dtype=float)
        alpha_raw = np.array([0.75], dtype=float)

        actual = compose_value_blend_probabilities(
            win_prob=win_prob,
            params={
                "probability_path_mode": "support_preserving_residual_path",
                "alpha_weight": 0.05,
                "alpha_scale": 0.5,
                "market_residual_weight": 0.08,
                "market_residual_scale": 0.75,
            },
            alpha_raw=alpha_raw,
            market_prob=market_prob,
        )
        legacy = compose_value_blend_probabilities(
            win_prob=win_prob,
            params={
                "alpha_weight": 0.05,
                "alpha_scale": 0.5,
                "market_blend_weight": 0.97,
            },
            alpha_raw=alpha_raw,
            market_prob=market_prob,
        )

        self.assertFalse(np.allclose(actual, legacy))

    def test_support_preserving_residual_path_positive_only_clips_negative_market_residual(self) -> None:
        win_prob = np.array([0.65, 0.35], dtype=float)
        market_prob = np.array([0.40, 0.60], dtype=float)

        actual = compose_value_blend_probabilities(
            win_prob=win_prob,
            params={
                "probability_path_mode": "support_preserving_residual_path",
                "market_residual_weight": 0.05,
                "market_residual_scale": 0.75,
                "market_residual_positive_only": True,
            },
            market_prob=market_prob,
        )

        win_logit = np.log(win_prob / (1.0 - win_prob))
        market_logit = np.log(market_prob / (1.0 - market_prob))
        market_signal = np.tanh((market_logit - win_logit) / 0.75)
        market_signal = np.maximum(market_signal, 0.0)
        expected_logit = win_logit + (0.05 * market_signal)
        expected = 1.0 / (1.0 + np.exp(-expected_logit))

        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    unittest.main()