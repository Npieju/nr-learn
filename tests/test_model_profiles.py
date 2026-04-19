from __future__ import annotations

import unittest

from racing_ml.common.model_profiles import resolve_model_run_profile


class ModelProfilesTest(unittest.TestCase):
    def test_local_nankan_baseline_pre_race_ready_profile_resolves_to_dedicated_dataset(self) -> None:
        profile_name, model_config, data_config, feature_config = resolve_model_run_profile(
            "local_nankan_baseline_pre_race_ready",
            default_model_config="configs/model.yaml",
            default_data_config="configs/data.yaml",
            default_feature_config="configs/features.yaml",
        )

        self.assertEqual(profile_name, "local_nankan_baseline_pre_race_ready")
        self.assertEqual(model_config, "configs/model_local_baseline.yaml")
        self.assertEqual(data_config, "configs/data_local_nankan_pre_race_ready.yaml")
        self.assertEqual(feature_config, "configs/features_local_baseline.yaml")

    def test_support_preserving_probability_profile_resolves(self) -> None:
        profile_name, model_config, data_config, feature_config = resolve_model_run_profile(
            "current_recommended_serving_support_preserving_prob_race_norm",
            default_model_config="configs/model.yaml",
            default_data_config="configs/data.yaml",
            default_feature_config="configs/features.yaml",
        )

        self.assertEqual(profile_name, "current_recommended_serving_support_preserving_prob_race_norm")
        self.assertEqual(
            model_config,
            "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_race_norm.yaml",
        )
        self.assertEqual(data_config, "configs/data.yaml")
        self.assertEqual(feature_config, "configs/features_catboost_rich_high_coverage_diag.yaml")

    def test_support_preserving_probability_positive_only_profile_resolves(self) -> None:
        profile_name, model_config, data_config, feature_config = resolve_model_run_profile(
            "current_recommended_serving_support_preserving_prob_race_norm_positive_only",
            default_model_config="configs/model.yaml",
            default_data_config="configs/data.yaml",
            default_feature_config="configs/features.yaml",
        )

        self.assertEqual(profile_name, "current_recommended_serving_support_preserving_prob_race_norm_positive_only")
        self.assertEqual(
            model_config,
            "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_race_norm_positive_only.yaml",
        )
        self.assertEqual(data_config, "configs/data.yaml")
        self.assertEqual(feature_config, "configs/features_catboost_rich_high_coverage_diag.yaml")


if __name__ == "__main__":
    unittest.main()