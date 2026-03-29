# JRA Baseline Artifact Inventory

## 1. Purpose

この文書は、現行 JRA baseline を支える artifact の正本 inventory と gap audit をまとめたものである。

以後の experiment issue は、baseline 比較の起点としてここに列挙した artifact 群を参照する。

## 2. Baseline Identity

2026-03-29 時点で、JRA baseline の基準 identity は次で固定する。

- profile: `current_recommended_serving_2025_latest`
- revision: `r20260325_current_recommended_serving_2025_latest_benchmark_refresh`
- universe: `jra`
- public reference: `artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json`

## 3. Canonical Artifact Set

baseline を読むときの canonical set は次の 6 本である。

1. public reference
2. promotion gate
3. revision gate
4. evaluation manifest
5. evaluation summary
6. relevant compare dashboards

## 4. Canonical Paths

### 4.1 Public Reference

- `artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json`

この manifest は read order と参照先をまとめた最上位入口として使う。

### 4.2 Formal Baseline Artifacts

- `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/revision_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/evaluation_by_date_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.csv`

### 4.3 Public Snapshot Docs

- `docs/public_benchmark_snapshot.md`
- `docs/project_overview.md`
- `docs/benchmarks.md`

### 4.4 Operational Compare Set

- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_sep_full_month_2025_latest_profile.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_dec_tail_2025_latest_profile.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh_vs_current_tighter_policy_search_candidate_2025_latest_sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_vs_tighter_policy_candidate_fresh_vs_current_tighter_policy_search_candidate_2025_latest_dec_tail_2025_latest_vs_tighter_policy_candidate_fresh.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh_vs_current_recommended_serving_2025_recent_2018_sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_vs_recent2018_true_retrain_fresh_vs_current_recommended_serving_2025_recent_2018_dec_tail_2025_latest_vs_recent2018_true_retrain_fresh.json`

## 5. Baseline Reading Order

baseline を読む順序は次で固定する。

1. `public_benchmark_reference_current_recommended_serving_2025_latest.json`
2. baseline promotion gate
3. baseline revision gate
4. baseline evaluation manifest / summary
5. compare dashboards

## 6. Known Supporting Candidate Artifacts

baseline を評価するとき、比較相手として最低限押さえる候補は次の 2 本である。

### 6.1 Tighter Policy Candidate

- revision: `r20260326_tighter_policy_ratio003`
- promotion: `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
- evaluation summary: `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260326_tighter_policy_ratio003_wf_full_nested.json`

### 6.2 Tighter Policy Abs80 Candidate

- revision: `r20260327_tighter_policy_ratio003_abs80`
- promotion: `artifacts/reports/promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
- evaluation summary: `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260327_tighter_policy_ratio003_abs80_wf_full_nested.json`

## 7. Gap Audit

### 7.1 No Critical Gap

現時点では、baseline を読むための critical artifact は揃っている。

特に `public_benchmark_reference_current_recommended_serving_2025_latest.json` があるため、baseline identity の誤読はかなり減っている。

### 7.2 Manageable Gaps

- `artifacts/reports/evaluation_manifest.json` と `artifacts/reports/evaluation_summary.json` は latest pointer なので、将来 run で上書きされうる
- docs 側の baseline 参照は複数ファイルに分散しているため、新規 experiment が versioned artifact ではなく latest pointer を誤って参照する余地がある
- candidate compare は豊富だが、「baseline canonical set」が 1 枚にまとまっていなかった

## 8. Operating Rule From This Audit

今後の experiment では次を守る。

- baseline 比較には latest pointer より versioned artifact を優先する
- baseline identity は profile だけでなく revision まで書く
- JRA baseline の参照起点はこの inventory か public reference manifest にする
- issue / PR には、最低でも baseline promotion gate と evaluation summary を貼る

## 9. Immediate Recommendation

次の experiment issue からは、baseline 欄に少なくとも次を明記する。

- baseline profile
- baseline revision
- baseline promotion artifact
- baseline evaluation summary
- 必要な compare dashboard
