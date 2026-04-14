from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRunProfile:
    description: str
    model_config: str
    data_config: str
    feature_config: str
    default_model_artifact_suffix: str | None = None


MODEL_RUN_PROFILES: dict[str, ModelRunProfile] = {
    "local_nankan_recommended": ModelRunProfile(
        description="Recommended local Nankan wf_runtime_narrow config for serving and replay with the stronger promoted r20260330 narrow runtime artifact suffix.",
        model_config="configs/model_local_baseline_wf_runtime_narrow.yaml",
        data_config="configs/data_local_nankan.yaml",
        feature_config="configs/features_local_baseline.yaml",
        default_model_artifact_suffix="r20260330_local_nankan_baseline_wf_runtime_narrow_v1",
    ),
    "local_nankan_value_blend_bootstrap": ModelRunProfile(
        description="Local Nankan bootstrap stack profile for result-ready NAR train, evaluate, and predict flows.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml",
        data_config="configs/data_local_nankan.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
    ),
    "local_nankan_value_blend_bootstrap_pre_race_ready": ModelRunProfile(
        description="Local Nankan bootstrap stack profile for pre-race-ready handoff artifacts before result-ready benchmark reruns.",
        model_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml",
        data_config="configs/data_local_nankan_pre_race_ready.yaml",
        feature_config="configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
    ),
    "local_nankan_baseline_pre_race_ready": ModelRunProfile(
        description="Local Nankan baseline profile for the dedicated strict pre-race-ready corpus after result arrival handoff.",
        model_config="configs/model_local_baseline.yaml",
        data_config="configs/data_local_nankan_pre_race_ready.yaml",
        feature_config="configs/features_local_baseline.yaml",
    ),
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
RECENT_DATA_CONFIGS: tuple[tuple[str, str, str], ...] = (
    (
        "configs/data_2025_recent_2018.yaml",
        "_2025_recent_2018",
        "with the 2025 holdout split and a recent-heavy training window starting at 2018-01-01",
    ),
    (
        "configs/data_2025_recent_2020.yaml",
        "_2025_recent_2020",
        "with the 2025 holdout split and a recent-heavy training window starting at 2020-01-01",
    ),
)


def _build_data_config_profile_variants(
    base_profiles: dict[str, ModelRunProfile],
    *,
    data_config: str,
    profile_suffix: str,
    description_suffix: str,
) -> dict[str, ModelRunProfile]:
    variant_profiles: dict[str, ModelRunProfile] = {}
    for profile_name, profile in base_profiles.items():
        if profile.data_config != "configs/data.yaml":
            continue

        variant_profile_name = f"{profile_name}{profile_suffix}"
        if variant_profile_name in base_profiles:
            continue

        variant_profiles[variant_profile_name] = ModelRunProfile(
            description=f"{profile.description.rstrip('.')} {description_suffix}.",
            model_config=profile.model_config,
            data_config=data_config,
            feature_config=profile.feature_config,
            default_model_artifact_suffix=profile.default_model_artifact_suffix,
        )
    return variant_profiles


MODEL_RUN_PROFILES.update(
    _build_data_config_profile_variants(
        MODEL_RUN_PROFILES,
        data_config=LATEST_DATA_CONFIG,
        profile_suffix=LATEST_PROFILE_SUFFIX,
        description_suffix="with the 2025 latest holdout split and netkeiba backfill data",
    )
)

for data_config, profile_suffix, description_suffix in RECENT_DATA_CONFIGS:
    MODEL_RUN_PROFILES.update(
        _build_data_config_profile_variants(
            MODEL_RUN_PROFILES,
            data_config=data_config,
            profile_suffix=profile_suffix,
            description_suffix=description_suffix,
        )
    )


def format_model_run_profiles() -> str:
    lines: list[str] = []
    for profile_name in sorted(MODEL_RUN_PROFILES):
        profile = MODEL_RUN_PROFILES[profile_name]
        lines.append(f"{profile_name}: {profile.description}")
        lines.append(f"  model_config={profile.model_config}")
        lines.append(f"  data_config={profile.data_config}")
        lines.append(f"  feature_config={profile.feature_config}")
        if profile.default_model_artifact_suffix:
            lines.append(f"  default_model_artifact_suffix={profile.default_model_artifact_suffix}")
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