# コマンドリファレンス

## 1. この文書の役割

この文書は、`nr-learn` の主要 CLI を用途別に引けるようにした実行リファレンスである。

網羅的な内部仕様は各 script の `--help` と実装を参照し、この文書では日常運用で使う入口だけを整理する。

長時間かかるコマンドは、現在は `ProgressBar` または `Heartbeat` により途中経過が見える前提で整備している。

CLI の基本挙動:

- operator 向け script は、config 不足、入力不足、output path の取り違えをなるべく早い段階で検出する。
- 想定内の失敗は concise な `failed: ...` で返し、unexpected exception のときだけ traceback を出す。
- `output file` を受ける引数には file path を渡し、`output dir` を受ける引数には directory path を渡す。

## 2. まず使うコマンド

### 2.1 データ取り込み

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_ingest.py --config configs/data.yaml
```

関連:

- [../scripts/run_validate_data_sources.py](../scripts/run_validate_data_sources.py)
- [../scripts/run_feature_gap_report.py](../scripts/run_feature_gap_report.py)
- [../scripts/run_netkeiba_coverage_snapshot.py](../scripts/run_netkeiba_coverage_snapshot.py)

### 2.2 学習

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_best_eval
```

profile 一覧:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --list-profiles
```

### 2.3 評価

正式判断の読み筋は [evaluation_guide.md](evaluation_guide.md) を参照する。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --profile current_best_eval --max-rows 120000
```

昇格判断の基本は、短窓の単発結果ではなく `stability_assessment=representative` を満たす評価である。

### 2.4 予測

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_predict.py --profile current_best_eval --race-date 2021-07-31
```

### 2.5 バックテスト

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backtest.py --profile current_best_eval
```

## 3. 正式な revision 評価

短い smoke / probe と、正式な改善判断は分けて扱う。

正式な判断は次で行う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --profile current_best_eval \
  --revision r20260321a \
  --evaluate-max-rows 200000 \
  --evaluate-wf-mode full
```

重い実行前に orchestration だけ確認したいときは次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --profile current_best_eval \
  --revision r20260321a \
  --evaluate-max-rows 200000 \
  --evaluate-wf-mode full \
  --dry-run
```

軽量 smoke として実際に通したいときは、train 側も行数を絞れる。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --profile current_best_eval \
  --revision r20260321_smoke \
  --train-max-train-rows 5000 \
  --train-max-valid-rows 1000 \
  --evaluate-pre-feature-max-rows 5000 \
  --evaluate-max-rows 5000 \
  --evaluate-wf-mode off
```

関連:

- [../scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
- [../scripts/run_promotion_gate.py](../scripts/run_promotion_gate.py)
- [development_flow.md](development_flow.md)
- [evaluation_guide.md](evaluation_guide.md)

補足:

- `run_revision_gate.py` は train、evaluate、promotion gate の各段階を progress 付きで出力する。
- `--dry-run` を付けると、重い train / evaluate を実行せずに planned command と revision manifest だけを確認できる。
- `--train-max-train-rows` と `--train-max-valid-rows` を使うと、real run でも lightweight smoke を組める。

## 4. serving 検証

基本の流れと各 artifact の読み方は [serving_validation_guide.md](serving_validation_guide.md) を参照する。

### 4.1 smoke

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py --profile current_recommended_serving --date 2024-09-14
```

### 4.2 2 候補比較

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving \
  --right-profile current_bankroll_candidate \
  --date 2024-09-16 \
  --date 2024-09-21 \
  --date 2024-09-22 \
  --date 2024-09-28 \
  --date 2024-09-29 \
  --window-label late_sep \
  --run-bankroll-sweep \
  --run-dashboard
```

関連:

- [../scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- [../scripts/run_serving_smoke_compare.py](../scripts/run_serving_smoke_compare.py)
- [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py)
- [../scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- [../scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)
- [serving_validation_guide.md](serving_validation_guide.md)

補足:

- `run_serving_profile_compare.py` は左右 smoke、compare、bankroll sweep、dashboard を段階ごとに出力する。
- provenance 用の `serving_smoke_profile_compare_*.json` も出し、途中 step が失敗した場合も可能な限り実行済み step と失敗位置を残す。
- `--dashboard-summary-output`、`--dashboard-chart-output`、`--dashboard-csv-output` は directory ではなく file path を渡す。
- `--left-summary-output`、`--right-summary-output`、`--compare-json-output`、`--compare-csv-output`、`--bankroll-json-output`、`--bankroll-csv-output`、`--manifest-output` も同様に file path 前提である。

共通注意:

- `--output`, `--output-file`, `--summary-path`, `--summary-csv`, `--manifest-output` のような引数に directory を渡すと fail-fast する。
- 逆に `--output-dir` のような directory 前提の引数には file path を渡さない。

## 5. netkeiba 系の代表コマンド

ID 準備:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_prepare_netkeiba_ids.py \
  --data-config configs/data.yaml \
  --crawl-config configs/crawl_netkeiba_template.yaml \
  --target race_result \
  --start-date 2020-01-01
```

収集:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_collect_netkeiba.py \
  --config configs/crawl_netkeiba_template.yaml \
  --target race_result \
  --limit 50
```

backfill:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backfill_netkeiba.py \
  --data-config configs/data.yaml \
  --crawl-config configs/crawl_netkeiba_template.yaml \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --date-order desc \
  --race-batch-size 100 \
  --pedigree-batch-size 500
```

補足:

- netkeiba 系の `run_prepare_netkeiba_ids.py`、`run_collect_netkeiba.py`、`run_backfill_netkeiba.py`、`run_netkeiba_benchmark_gate.py` は、いずれも進捗または heartbeat を出す。

## 6. 補助コマンド

### 6.1 A/B 比較

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_ab_compare.py \
  --base-profile current_best_eval \
  --challenger-profile current_recommended_serving \
  --max-rows 30000
```

### 6.2 ダッシュボード生成

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_dashboard.py
```

### 6.3 value stack tuning

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_tune_value_stack.py \
  --summary-path artifacts/reports/tune_value_stack_summary.json
```

### 6.4 進捗表示の軽量 smoke

progress の退行確認だけをしたいときは、再学習や再推論を伴わない既存 artifact ベースのコマンドを優先する。

ingest / diagnostics / manifest:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_ingest.py --config configs/data.yaml

/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_coverage_snapshot.py \
  --config configs/data.yaml \
  --tail-rows 200 \
  --output artifacts/reports/netkeiba_coverage_snapshot_smoke.json

/workspaces/nr-learn/.venv/bin/python scripts/run_validate_evaluation_manifest.py \
  --manifest artifacts/reports/evaluation_manifest.json \
  --output artifacts/reports/evaluation_manifest_validation_smoke.json
```

WF 後段チェーン:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_compare.py \
  --reports \
  artifacts/reports/wf_threshold_sweep_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_20240601_20240929_wf_full_nested.json \
  artifacts/reports/wf_threshold_sweep_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may_20240601_20240929_wf_full_nested.json \
  --output artifacts/reports/wf_threshold_compare_progress_smoke.json \
  --summary-csv artifacts/reports/wf_threshold_compare_progress_smoke.csv \
  --fold-summary-csv artifacts/reports/wf_threshold_compare_progress_smoke_folds.csv

/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_signature_report.py \
  --compare-report artifacts/reports/wf_threshold_compare_current_profiles_h1_vs_h2_2024.json \
  --output artifacts/reports/wf_threshold_signature_report_progress_smoke.json \
  --summary-csv artifacts/reports/wf_threshold_signature_report_progress_smoke.csv

/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_mitigation_focus.py \
  --compare-report artifacts/reports/wf_threshold_compare_current_profiles_h1_vs_h2_2024.json \
  --shortlist-report artifacts/reports/wf_threshold_mitigation_shortlist_current_profiles_h1_vs_h2_2024.json \
  --output artifacts/reports/wf_threshold_mitigation_focus_progress_smoke.json \
  --summary-csv artifacts/reports/wf_threshold_mitigation_focus_progress_smoke.csv

/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_mitigation_policy_probe.py \
  --focus-report artifacts/reports/wf_threshold_mitigation_focus_current_profiles_h1_vs_h2_2024.json \
  --drilldown-report artifacts/reports/wf_threshold_signature_drilldown_current_profiles_h1_vs_h2_2024.json \
  --output artifacts/reports/wf_threshold_mitigation_policy_probe_progress_smoke.json \
  --summary-csv artifacts/reports/wf_threshold_mitigation_policy_probe_progress_smoke.csv

/workspaces/nr-learn/.venv/bin/python scripts/run_generate_serving_candidates_from_mitigation_probe.py \
  --policy-probe artifacts/reports/wf_threshold_mitigation_policy_probe_current_profiles_h1_vs_h2_2024.json \
  --output-json artifacts/reports/generated_serving_candidates_from_mitigation_probe_progress_smoke.json \
  --output-yaml artifacts/reports/generated_serving_candidates_from_mitigation_probe_progress_smoke.yaml
```

補足:

- この節のコマンドは、progress が出るかを見るための軽量確認であり、評価改善の根拠には使わない。
- `run_wf_threshold_compare.py` だけは fold 集計を再生成するため、他より少し重いが再学習や推論は伴わない。

## 7. artifact の見方

主要な出力先:

- 学習モデル: `artifacts/models/`
- train / evaluate / backtest report: `artifacts/reports/`
- prediction: `artifacts/predictions/`
- dashboard: `artifacts/reports/dashboard/`

特に正式判断で見るもの:

- `artifacts/reports/evaluation_summary.json`
- `artifacts/reports/evaluation_manifest.json`
- `artifacts/reports/promotion_gate_report.json` または `promotion_gate_<revision>.json`

## 8. 補足

- 高頻度で使う script 以外は [../scripts](../scripts) を起点に探す。
- 用途別の script 一覧は [scripts_guide.md](scripts_guide.md) を参照する。
- 詳しい運用ルールは [development_flow.md](development_flow.md) を参照する。
- benchmark の判断基準は [benchmarks.md](benchmarks.md) を参照する。
- artifact の見方は [artifact_guide.md](artifact_guide.md) を参照する。
- GPU / Docker / Notebook 周りは [environment_notes.md](environment_notes.md) を参照する。