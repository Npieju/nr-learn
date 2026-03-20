from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRunProfile:
    model_config: str
    data_config: str
    feature_config: str


MODEL_RUN_PROFILES: dict[str, ModelRunProfile] = {
    "current_best_eval": ModelRunProfile(
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_recommended_serving": ModelRunProfile(
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
}


def resolve_model_run_profile(
    profile_name: str | None,
    *,
    default_model_config: str,
    default_data_config: str,
    default_feature_config: str,
) -> tuple[str | None, str, str, str]:
    if profile_name is None:
        return None, default_model_config, default_data_config, default_feature_config

    profile = MODEL_RUN_PROFILES.get(profile_name)
    if profile is None:
        raise ValueError(
            f"Unknown model profile: {profile_name}. Available profiles: {sorted(MODEL_RUN_PROFILES)}"
        )

    return profile_name, profile.model_config, profile.data_config, profile.feature_config