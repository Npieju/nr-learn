from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRunProfile:
    description: str
    model_config: str
    data_config: str
    feature_config: str


MODEL_RUN_PROFILES: dict[str, ModelRunProfile] = {
    "current_best_eval": ModelRunProfile(
        description="Best nested evaluation mainline with May policy and runtime score override support.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_best_eval_2025_latest": ModelRunProfile(
        description="Best nested evaluation mainline with a 2025 full-year holdout split for post-backfill latest checks.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml",
        data_config="configs/data_2025_latest.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_recommended_serving": ModelRunProfile(
        description="Simplified serving default that matches mainline behavior across validated actual-date checks.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_long_horizon_serving": ModelRunProfile(
        description="Long-horizon operational serving alias that keeps baseline behavior outside September and applies the validated September Kelly-only guard.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_sep_selected_rows_kelly_only_candidate.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_bankroll_candidate": ModelRunProfile(
        description="Conservative serving candidate that keeps May-July Kelly overrides and lowers Aug+ portfolio blend.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_portfolio_lower_blend_hybrid_keep_kelly.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_ev_candidate": ModelRunProfile(
        description="Serving candidate that keeps May-July Kelly overrides and raises Aug+ portfolio EV floor to 1.0.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_portfolio_ev_only_hybrid_keep_kelly.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_sep_guard_candidate": ModelRunProfile(
        description="September-only de-risk candidate that falls straight from sparse September portfolio selection into Kelly fallback.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_sep_selected_rows_kelly_only_candidate.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
    "current_tighter_policy_search_candidate": ModelRunProfile(
        description="Candidate that tightens policy search thresholds so 2025 high-support broad betting falls back to stricter Kelly selection.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe.yaml",
        data_config="configs/data.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag.yaml",
    ),
}


LATEST_DATA_CONFIG = "configs/data_2025_latest.yaml"
LATEST_PROFILE_SUFFIX = "_2025_latest"


def _build_latest_profile_variants(base_profiles: dict[str, ModelRunProfile]) -> dict[str, ModelRunProfile]:
    latest_profiles: dict[str, ModelRunProfile] = {}
    for profile_name, profile in base_profiles.items():
        if profile.data_config != "configs/data.yaml":
            continue

        latest_profile_name = f"{profile_name}{LATEST_PROFILE_SUFFIX}"
        if latest_profile_name in base_profiles:
            continue

        latest_profiles[latest_profile_name] = ModelRunProfile(
            description=(
                f"{profile.description.rstrip('.')} with the 2025 latest holdout split and netkeiba backfill data."
            ),
            model_config=profile.model_config,
            data_config=LATEST_DATA_CONFIG,
            feature_config=profile.feature_config,
        )
    return latest_profiles


MODEL_RUN_PROFILES.update(_build_latest_profile_variants(MODEL_RUN_PROFILES))


def format_model_run_profiles() -> str:
    lines: list[str] = []
    for profile_name in sorted(MODEL_RUN_PROFILES):
        profile = MODEL_RUN_PROFILES[profile_name]
        lines.append(f"{profile_name}: {profile.description}")
        lines.append(f"  model_config={profile.model_config}")
        lines.append(f"  data_config={profile.data_config}")
        lines.append(f"  feature_config={profile.feature_config}")
    return "\n".join(lines)


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