# Docs Index

この `docs/` は、初めて引き継ぐ人が「何を読めば全体像と現在地が分かるか」を最短で判断できるように整理している。

2026-03-29 以降の新規標準化は、既存 docs を参照しつつも、まず [strategy_v2.md](strategy_v2.md) と [autonomous_dev_standard.md](autonomous_dev_standard.md) を入口に進める。

## 方針

- 残すのは、構造理解と運用判断に必要な正本ドキュメントに限定する。
- 日付依存の実験メモ、セッションメモ、途中経過の説明資料は `docs/` に置かない。
- 古い判断経緯や細かな数値の追跡は `git log` と `artifacts/reports/` を正本にする。

## まず読む順番

引き継ぎ時は、まず次の 5 本を順に読む。

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [development_flow.md](development_flow.md)
4. [command_reference.md](command_reference.md)
5. [benchmarks.md](benchmarks.md)

対外説明が必要なら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) と [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) を追加で読む。実装の中身まで入る必要がある場合だけ [system_architecture.md](system_architecture.md) を読む。

latest 2025 の actual-date compare を再開したいだけなら、次の 3 段で足りる。

1. [serving_validation_guide.md](serving_validation_guide.md) の dashboard summary JSON 一覧から、September は `long_horizon -> tighter policy -> recent-2018` の順で見て、次に December control を確認する。
2. compare を回し直す必要があるときだけ [command_reference.md](command_reference.md) の latest 2025 compare 例を同じ順に使う。
3. formal support まで根拠を掘る必要があるときだけ [benchmarks.md](benchmarks.md) と `artifacts/reports/` の promotion gate / evaluation summary に降りる。

新しい issue 駆動運用を始めるときは、次の 3 本を最初に読む。

1. [strategy_v2.md](strategy_v2.md)
2. [autonomous_dev_standard.md](autonomous_dev_standard.md)
3. [initial_issue_backlog.md](initial_issue_backlog.md)

AI coding の標準化を進めるときは、追加で [ai_coding_best_practices.md](ai_coding_best_practices.md) を読む。

運用上の hard caution は [development_operational_cautions.md](development_operational_cautions.md) を使う。

progress 不足の棚卸しは [progress_coverage_audit.md](progress_coverage_audit.md) を使う。

現在 GitHub に起こすべき案件の直近キューは [github_issue_queue_current.md](github_issue_queue_current.md) を使う。

ML モデル開発プロセスを迷わず進めるには、[ml_model_development_standard.md](ml_model_development_standard.md) を正本として使う。

`ROI > 1.20` の success 条件を参照するときは、[roi120_kpi_definition.md](roi120_kpi_definition.md) を使う。

最初の実務 issue を切るときは、[issue_drafts_first_wave.md](issue_drafts_first_wave.md) と [ml_stage_checklist.md](ml_stage_checklist.md) を併用する。

JRA baseline の参照起点は [jra_baseline_artifact_inventory.md](jra_baseline_artifact_inventory.md) を使う。

feature 実験の優先順位づけには [feature_family_ranking.md](feature_family_ranking.md) を使う。

policy 実験の優先順位づけには [policy_family_shortlist.md](policy_family_shortlist.md) を使う。

次の policy 本線 issue 下書きは [next_issue_kelly_runtime_family.md](next_issue_kelly_runtime_family.md) を使う。

次の JRA feature child role split issue 下書きは [next_issue_class_rest_surface_conditional_actual_date_role_split.md](next_issue_class_rest_surface_conditional_actual_date_role_split.md) を使う。

次の最有力 experiment の実務下書きは [next_issue_tighter_policy_frontier.md](next_issue_tighter_policy_frontier.md) を使う。

revision gate 前の確認には [revision_gate_candidate_checklist.md](revision_gate_candidate_checklist.md) を使う。

`tighter policy` の実行手順は [tighter_policy_frontier_execution.md](tighter_policy_frontier_execution.md) を使う。

promoted anchor に対する challenger 判定には [policy_challenger_decision_checklist.md](policy_challenger_decision_checklist.md) を使う。

non-default promoted line の operator ordering を確認するときは [next_issue_analysis_first_promoted_candidate_ordering.md](next_issue_analysis_first_promoted_candidate_ordering.md) を使う。

次の serving / seasonal 本線 issue 下書きは [next_issue_seasonal_derisk_long_horizon.md](next_issue_seasonal_derisk_long_horizon.md) を使う。

seasonal de-risk の判断標準は [seasonal_derisk_decision_standard.md](seasonal_derisk_decision_standard.md) を使う。

seasonal family の次 issue 下書きは [next_issue_sep_guard_secondary_family.md](next_issue_sep_guard_secondary_family.md) を使う。

seasonal second-layer fallback の順位づけは [seasonal_secondary_fallback_standard.md](seasonal_secondary_fallback_standard.md) を使う。

runtime 改善の次 issue 下書きは [next_issue_primary_source_shaping.md](next_issue_primary_source_shaping.md) を使う。
primary tail cache の運用標準は [primary_tail_cache_operational_policy.md](primary_tail_cache_operational_policy.md) を使う。
primary tail cache の次 issue 下書きは [next_issue_primary_tail_cache_default_promotion.md](next_issue_primary_tail_cache_default_promotion.md) を使う。
runtime 後に experiment queue へ戻すときは [next_issue_post_runtime_benchmark_refresh.md](next_issue_post_runtime_benchmark_refresh.md) を使う。
runtime 後の最初の experiment 再開は [next_issue_tighter_policy_reentry_after_runtime.md](next_issue_tighter_policy_reentry_after_runtime.md) を使う。
feature 側の次 issue 下書きは [next_issue_class_rest_surface_interactions.md](next_issue_class_rest_surface_interactions.md) を起点にし、support hardening の記録は [next_issue_class_rest_surface_support_hardening.md](next_issue_class_rest_surface_support_hardening.md) と [next_issue_surface_plus_class_layoff_interactions.md](next_issue_surface_plus_class_layoff_interactions.md) を参照し、promoted 後の serving read は [next_issue_post_surface_plus_class_layoff_promotion.md](next_issue_post_surface_plus_class_layoff_promotion.md) と [next_issue_surface_plus_class_layoff_bet_rate_robustness.md](next_issue_surface_plus_class_layoff_bet_rate_robustness.md) を見て、widening failure の後は [next_issue_surface_plus_class_layoff_role_split.md](next_issue_surface_plus_class_layoff_role_split.md) を使う。
formal promoted line と operational default line の切り分けは [promoted_vs_operational_role_split_standard.md](promoted_vs_operational_role_split_standard.md) を使う。次の feature family reentry は [next_issue_jockey_trainer_combo_regime_extension.md](next_issue_jockey_trainer_combo_regime_extension.md) を使う。
first execution candidate は [next_issue_jockey_trainer_combo_style_distance_candidate.md](next_issue_jockey_trainer_combo_style_distance_candidate.md) を使う。
second child を narrow に再開するときは [next_issue_jockey_trainer_combo_closing_time_selective_candidate.md](next_issue_jockey_trainer_combo_closing_time_selective_candidate.md) を使う。
second child の role decision は [next_issue_jockey_trainer_combo_closing_time_role_split.md](next_issue_jockey_trainer_combo_closing_time_role_split.md) を使う。
execution が resource pressure で止まる場合は [next_issue_jockey_trainer_combo_resource_safe_execution.md](next_issue_jockey_trainer_combo_resource_safe_execution.md) を参照する。
first promoted child の role decision は [next_issue_jockey_trainer_combo_role_split.md](next_issue_jockey_trainer_combo_role_split.md) を使う。
NAR を JRA と無理に統合せず別 universe として進める標準は [nar_model_transfer_strategy.md](nar_model_transfer_strategy.md) を使う。
NAR の formal read で分母つき bet-rate を必須にする標準は [nar_bet_denominator_standard.md](nar_bet_denominator_standard.md) を使う。
NAR の issue comment / decision summary にそのまま使う read ひな形は [nar_formal_read_template.md](nar_formal_read_template.md) を使う。
NAR baseline の first execution issue は [next_issue_local_nankan_baseline_formalization.md](next_issue_local_nankan_baseline_formalization.md) を使う。
NAR baseline 完了後の denominator-first decision は [next_issue_nar_post_formal_read.md](next_issue_nar_post_formal_read.md) を使う。
NAR の `wf_feasibility` runtime follow-up は [next_issue_nar_wf_runtime_followup.md](next_issue_nar_wf_runtime_followup.md) を使う。
NAR の runtime compare 候補 config は [model_local_baseline_wf_runtime_narrow.yaml](../configs/model_local_baseline_wf_runtime_narrow.yaml) を起点にする。
NAR の first feature-family replay は [next_issue_nar_class_rest_surface_replay.md](next_issue_nar_class_rest_surface_replay.md) を使い、candidate feature config は [features_local_baseline_class_rest_surface_replay.yaml](../configs/features_local_baseline_class_rest_surface_replay.yaml) を起点にする。
NAR replay が no-op だった場合の feature-level 切り分けは [next_issue_nar_class_rest_surface_availability_audit.md](next_issue_nar_class_rest_surface_availability_audit.md) を使う。
build 済み replay features を actual candidate に戻す selection fix は [next_issue_nar_selection_fix_for_buildable_replay.md](next_issue_nar_selection_fix_for_buildable_replay.md) を使う。
NAR の class/rest replay が baseline 劣後で終わった後の next family は [next_issue_nar_jockey_trainer_combo_replay.md](next_issue_nar_jockey_trainer_combo_replay.md) を使う。
NAR の `jockey / trainer / combo` replay が formal `pass / promote` まで到達した後の ops fix は [next_issue_nar_wf_summary_path_alignment.md](next_issue_nar_wf_summary_path_alignment.md) を使う。
NAR の combo replay と path alignment fix 完了後の gate/frame replay read は [next_issue_nar_gate_frame_course_replay.md](next_issue_nar_gate_frame_course_replay.md) を使う。
gate/frame replay が baseline 劣後で終わった後の next family は [next_issue_nar_owner_signal_replay.md](next_issue_nar_owner_signal_replay.md) を使う。
NAR の高すぎる AUC / ROI を leak と optimism に分解する監査は [next_issue_nar_evaluation_integrity_audit.md](next_issue_nar_evaluation_integrity_audit.md) を使う。
market dependency を切り分けた後の本線である policy optimism 監査は [next_issue_nar_policy_optimism_audit.md](next_issue_nar_policy_optimism_audit.md) を使う。
policy optimism の exact phase compare を取る baseline rerun は [next_issue_nar_baseline_pathfixed_rerun_for_policy_audit.md](next_issue_nar_baseline_pathfixed_rerun_for_policy_audit.md) を使う。
policy optimism の exact compare 後に conservative short-circuit を入れる corrective issue は [next_issue_nar_promotion_alignment_short_circuit.md](next_issue_nar_promotion_alignment_short_circuit.md) を使う。
formal benchmark を held-out test metrics に揃える corrective issue は [next_issue_nar_holdout_formal_benchmark_alignment.md](next_issue_nar_holdout_formal_benchmark_alignment.md) を使う。
NAR の formal benchmark source は `#73` で held-out test metrics に揃えたため、以後の NAR formal read は valid-side aggregate ではなく held-out aggregate を正本とする。
held-out formal `ROI < 1.0` でも `promote` になる permissive threshold の corrective issue は [next_issue_nar_promotion_threshold_realignment.md](next_issue_nar_promotion_threshold_realignment.md) を使う。
local Nankan wrapper は `#74` 以後、promotion gate に held-out formal `weighted_roi >= 1.0` を default で要求する。
pace / closing-fit selective replay の formal read は [next_issue_pace_closing_fit_selective_candidate.md](next_issue_pace_closing_fit_selective_candidate.md) を使い、その後の operational role split は [next_issue_pace_closing_fit_actual_date_role_split.md](next_issue_pace_closing_fit_actual_date_role_split.md) を使う。
pace / closing-fit selective line の September fallback compare は [next_issue_pace_closing_fit_september_fallback_compare.md](next_issue_pace_closing_fit_september_fallback_compare.md) を使う。
tighter policy の September fallback ordering を再確認するときは [next_issue_tighter_policy_september_fallback_compare.md](next_issue_tighter_policy_september_fallback_compare.md) を使う。
JRA baseline に対する owner signal の marginal contribution を切り分けるときは [next_issue_owner_signal_ablation_audit.md](next_issue_owner_signal_ablation_audit.md) を使う。

tail loader optimization の gate 運用は [tail_loader_equivalence_gate_standard.md](tail_loader_equivalence_gate_standard.md) を使う。

## 正本

このプロジェクトの理解と運用に必要な中核文書は次の 8 本である。

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [development_flow.md](development_flow.md)
4. [benchmarks.md](benchmarks.md)
5. [evaluation_guide.md](evaluation_guide.md)
6. [command_reference.md](command_reference.md)
7. [system_architecture.md](system_architecture.md)
8. [public_benchmark_snapshot.md](public_benchmark_snapshot.md)

## 補助資料

必要なときだけ読む補助資料は次のとおりである。

1. [artifact_guide.md](artifact_guide.md)
2. [environment_notes.md](environment_notes.md)
3. [serving_validation_guide.md](serving_validation_guide.md)
4. [data_extension.md](data_extension.md)
5. [scripts_guide.md](scripts_guide.md)

## 文書の役割

### 正本

- [project_overview.md](project_overview.md)
  - プロジェクトの目的、現在の operational baseline、直近の正式通過候補、未決定事項をまとめた入口資料。
- [roadmap.md](roadmap.md)
  - いま何が完了していて、次に何を決めるかを管理する実行計画の正本。
- [development_flow.md](development_flow.md)
  - smoke / probe と full revision gate をどう分けるか、`revision` と commit をどの単位で切るかを定義する運用ルール。
- [benchmarks.md](benchmarks.md)
  - 何を良い結果とみなすか、現在の baseline/candidate をどう扱うかを定義する内部採用基準。
- [evaluation_guide.md](evaluation_guide.md)
  - `run_evaluate.py`、`stability_assessment`、`run_promotion_gate.py`、`revision` 単位の正式判断をまとめた評価ガイド。
- [command_reference.md](command_reference.md)
  - 日常で使う主要 CLI の入口を用途別にまとめた実行リファレンス。
- [system_architecture.md](system_architecture.md)
  - 実コードに対応したシステム構成、主要モジュール、学習から serving までの流れを整理した技術資料。
- [public_benchmark_snapshot.md](public_benchmark_snapshot.md)
  - 対外説明で使う current benchmark の要約と、現在の採用位置づけをまとめた公開向け snapshot。
- [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md)
  - 対外説明で benchmark 数字と operational role の関係を読むための公開向けガイド。

### 補助資料

- [artifact_guide.md](artifact_guide.md)
  - 各 CLI がどこに何を出力するか、正式判断でどの artifact を見るかを整理した資料。
- [environment_notes.md](environment_notes.md)
  - GPU / Docker / Notebook 周りの実務メモとトラブルシュートをまとめた補助資料。
- [serving_validation_guide.md](serving_validation_guide.md)
  - serving smoke / replay / compare / bankroll sweep / dashboard の導線をまとめた検証ガイド。
  - latest 2025 の actual-date compare を再開するときは、この文書の quickstart を最初に見る。
- [data_extension.md](data_extension.md)
  - Kaggle/JRA 主表に対して、netkeiba などの外部データをどう追加し、どう検証するかを整理した資料。
- [scripts_guide.md](scripts_guide.md)
  - `scripts/` 配下の CLI と補助 shell を用途別に引けるようにした索引資料。

## 更新するときの基準

- 現在地や優先順位が変わったら [roadmap.md](roadmap.md) を更新する。
- baseline / candidate の位置づけが変わったら [project_overview.md](project_overview.md) を更新する。
- 採用基準や formal result の読み方が変わったら [benchmarks.md](benchmarks.md) と [evaluation_guide.md](evaluation_guide.md) を更新する。
- 対外説明に出す数値が変わったら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) を更新する。
- benchmark 数字と運用上の位置づけの説明順が変わったら [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) も更新する。
- 実行手順や Git 運用が変わったら [development_flow.md](development_flow.md) と [command_reference.md](command_reference.md) を更新する。

## コードの入口

- 学習: [scripts/run_train.py](../scripts/run_train.py)
- 評価: [scripts/run_evaluate.py](../scripts/run_evaluate.py)
- 予測: [scripts/run_predict.py](../scripts/run_predict.py)
- バックテスト: [scripts/run_backtest.py](../scripts/run_backtest.py)
- revision gate: [scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
- latest revision gate wrapper: [scripts/run_netkeiba_latest_revision_gate.py](../scripts/run_netkeiba_latest_revision_gate.py)
- serving smoke: [scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- serving compare dashboard: [scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- serving compare aggregate: [scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)
- script 全体の索引: [scripts_guide.md](scripts_guide.md)

## 補足

- 実験の細かい経緯や旧判断を確認したい場合は、削除した文書名を探すのではなく、コミット履歴と artifact を起点にたどるほうが正確である。
- 今後 `docs/` に新しい文書を追加する前に、その内容が既存の 8 本の正本か 5 本の補助資料のどこへ統合できるかを先に確認する。
