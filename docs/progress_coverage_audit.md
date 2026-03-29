# Progress Coverage Audit

## 1. Purpose

この文書は、`docs/development_operational_cautions.md` で定義した

`数秒で終わらない source には progress 必須`

を、現行 repo に対して棚卸しした結果である。

## 2. Audit Rule

今回の簡易監査では、`scripts/run_*.py` を対象にして、少なくとも次のいずれかを持つかを確認した。

- `ProgressBar`
- `Heartbeat`
- `log_progress`

これは完全な品質監査ではないが、progress instrumentation の有無を機械的に把握する一次スクリーニングとしては十分である。

監査日:

- `2026-03-29`

## 3. Already Instrumented Examples

すでに progress を備えている代表例:

- `scripts/run_train.py`
- `scripts/run_evaluate.py`
- `scripts/run_revision_gate.py`
- `scripts/run_wf_threshold_sweep.py`
- `scripts/run_backfill_netkeiba.py`
- `scripts/run_local_backfill_then_benchmark.py`

これらは今後の新規 CLI で参照すべき実装例である。

## 4. Missing Or Weak-Coverage Candidates

2026-03-29 時点のこの監査対象では、簡易監査で progress helper が見当たらない `scripts/run_*.py` は解消済みである。

この監査の初版から次は progress 対応済みになった。

- `scripts/run_local_benchmark_gate.py`
- `scripts/run_local_coverage_snapshot.py`
- `scripts/run_local_data_source_validation.py`
- `scripts/run_local_evaluate.py`
- `scripts/run_local_feature_gap_report.py`
- `scripts/run_local_public_snapshot.py`
- `scripts/run_mixed_universe_compare.py`
- `scripts/run_mixed_universe_left_gap_audit.py`
- `scripts/run_mixed_universe_left_recovery_plan.py`
- `scripts/run_mixed_universe_numeric_compare.py`
- `scripts/run_mixed_universe_numeric_summary.py`
- `scripts/run_mixed_universe_readiness.py`
- `scripts/run_mixed_universe_recovery.py`
- `scripts/run_mixed_universe_schema.py`
- `scripts/run_mixed_universe_status_board.py`
- `scripts/run_netkeiba_pedigree_postprocess.py`
- `scripts/run_netkeiba_refresh_pedigree_missing.py`
- `scripts/run_public_benchmark_reference.py`

## 5. Priority Order

### P0: Wrapper / Gate / Recovery

最優先で progress を入れるべき候補:

- `scripts/run_mixed_universe_left_gap_audit.py`
- `scripts/run_mixed_universe_left_recovery_plan.py`

理由:

- multi-stage 実行が多い
- subprocess 呼び出しを含む
- operator が「今どの phase か」を見失いやすい
- blocked / failed 時の停止点が分かりにくい

### P1: Summary / Snapshot / Numeric Compare

次点で progress を入れる候補:

- remaining runtime review and real-job validation

理由:

- I/O と aggregation が長くなりやすい
- 出力はあるが途中経過が見えないと stalled に見える

### P2: Small Utility / Refresh

後回しでよいが、触るときに入れる候補:

- none in the current `scripts/run_*.py` audit scope

## 6. Standard Remediation Pattern

各 script の是正は次の最小パターンで行う。

1. `log_progress()` を追加する
2. `ProgressBar(total=...)` を開始時に置く
3. 長時間区間を `Heartbeat(...)` で囲む
4. `progress.update(...)` で phase 完了を明示する
5. `progress.complete(...)` で正常終了を閉じる

subprocess wrapper では、少なくとも次を出す。

- starting phase
- child command start
- child command end / return code
- next phase

## 7. Recommended Next Issue

次に切るべき process issue:

`[ops] Progress instrumentation hardening for long-running wrappers`

対象の第一弾:

- `scripts/run_local_benchmark_gate.py`
- `scripts/run_mixed_universe_recovery.py`
- `scripts/run_mixed_universe_compare.py`
- `scripts/run_mixed_universe_status_board.py`

2026-03-29 update:

第一弾と follow-up 束は対応済み。次は実ジョブで progress 粒度を見直し、必要なら message quality と failure-phase logging を整える段階である。

## 8. Spot Check Result

2026-03-29 に dry-run spot check を実施した。

対象:

- `scripts/run_mixed_universe_status_board.py --revision progress_spotcheck --dry-run`
- `scripts/run_mixed_universe_readiness.py --revision progress_spotcheck --dry-run`
- `scripts/run_mixed_universe_numeric_compare.py --revision progress_spotcheck --dry-run`
- `scripts/run_local_public_snapshot.py --revision progress_spotcheck --dry-run`

確認できたこと:

- start / phase / write / completion がログで判別できる
- partial / not_ready のような非成功終了でも stopped point が読める
- manifest path が completion message で読める
- progress 未整備時のような「無限ループか停止か分からない」状態はかなり解消した

次の実務段階:

- real run でも heartbeat 間隔と文言が十分かを確認する
- 長時間 child command の outer wrapper message が coarse すぎる箇所は粒度を追加する
- failure path にも `current_phase` と artifact exit が十分残るかを重点確認する

完了条件:

- progress start / heartbeat / completion が入る
- failure phase が分かる
- output artifact path がログから分かる
- long-running subprocess wrapper が stalled に見えない

2026-03-29 real-run update:

representative real run も追加で確認した。

対象:

- `scripts/run_public_benchmark_reference.py --output artifacts/tmp/public_benchmark_reference_realrun_spotcheck.json`
- `scripts/run_mixed_universe_status_board.py --revision r20260328_reference_bridge --output artifacts/tmp/mixed_universe_status_board_realrun_spotcheck.json`

確認できたこと:

- completed 系では `loading ... started/done`、phase update、manifest write、saved path まで一連で読める
- partial 系でも non-zero exit 前に `status=partial` と manifest path が出るため、停止ではなく正常な partial exit と判別できる
- operator が最低限ほしい `current_phase` と `recommended_action` は manifest 側で回収できる
- 今回の real run では message quality を理由に追加修正が必要な箇所は見当たらなかった

今回の real-run 結論:

- `scripts/run_*.py` の progress instrumentation は、dry-run だけでなく representative real run でも運用上十分に読める
- follow-up は新しい long-running wrapper 追加時の継続監査でよい

2026-03-29 WF diagnostics follow-up:

`run_wf_feasibility_diag.py` については、formal JRA run で fold 内探索が長く見えるケースがあったため、追加の checkpoint visibility を入れた。

追加した内容:

- fold 完了時に `feasible_candidates` と `elapsed` を出す
- fold 内の search chunk でも定期 checkpoint を出す
- checkpoint には `fold`, `search_step`, `blend_weight`, `min_edge`, `min_prob`, `odds`, `candidate_rows` を含める

smoke 確認:

- `timeout 120 .venv/bin/python scripts/run_wf_feasibility_diag.py --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90.yaml --data-config configs/data_2025_latest.yaml --feature-config configs/features_catboost_rich_high_coverage_diag.yaml --artifact-suffix wf_progress_smoke --model-artifact-suffix r20260326_tighter_policy_ratio003 --pre-feature-max-rows 40000 --wf-mode fast`

確認できたこと:

- fold 開始後に fold 内 checkpoint が実際に出る
- operator が「fold は進んでいるが search space が重い」状態を stalled と誤認しにくくなる
- 少なくとも smoke 範囲ではログ量は許容内だった

2026-03-29 parent-child streaming follow-up:

`run_revision_gate.py` 配下で child CLI の progress が親から見えにくい問題に対して、parent-child streaming を追加した。

追加した内容:

- child Python を `python -u` で起動する
- `capture_output=True` ではなく live forwarding で stdout / stderr を親へ流す
- 失敗分類用には child 出力を保持しつつ、operator 向けにはリアルタイム表示する

smoke 確認:

- `timeout 150 .venv/bin/python scripts/run_revision_gate.py --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_odds25.yaml --data-config configs/data_2025_latest.yaml --feature-config configs/features_catboost_rich_high_coverage_diag.yaml --revision r20260329_streaming_smoke_abs90_odds25 --train-artifact-suffix r20260329_streaming_smoke_abs90_odds25 --skip-train --evaluate-model-artifact-suffix r20260326_tighter_policy_ratio003 --evaluate-max-rows 30000 --evaluate-pre-feature-max-rows 40000 --evaluate-wf-mode fast --evaluate-wf-scheme nested --promotion-min-feasible-folds 3`

確認できたこと:

- parent の `run_revision_gate.py` 配下で child `run_evaluate.py` の heartbeat が live に見える
- `loading training table running...` と `building features running...` のような child progress が親で読める
- 今後の challenger run では `ps` / `lsof` へ逃げずに child progress を追いやすい
