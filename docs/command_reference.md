# コマンドリファレンス

## 1. この文書の役割

この文書は、`nr-learn` の主要 CLI を用途別に引けるようにした実行リファレンスである。

網羅的な内部仕様は各 script の `--help` と実装を参照し、この文書では日常運用で使う入口だけを整理する。

長時間かかるコマンドは、現在は `ProgressBar` または `Heartbeat` により途中経過が見える前提で整備している。

CLI の基本挙動:

- operator 向け script は、config 不足、入力不足、output path の取り違えをなるべく早い段階で検出する。
- 想定内の失敗は concise な `failed: ...` で返し、unexpected exception のときだけ traceback を出す。
- `output file` を受ける引数には file path を渡し、`output dir` を受ける引数には directory path を渡す。

## 2. Git 管理

Git の運用方針そのものは [development_flow.md](development_flow.md) を正本とし、ここでは日常作業で実際に使う入口だけをまとめる。

### 2.1 状態確認

変更に入る前と、まとまった変更を終えた直後は次を確認する。

```bash
git status
git diff --stat
```

特定ファイルだけ差分を見たいときの例:

```bash
git diff -- docs/roadmap.md docs/command_reference.md
```

### 2.2 履歴確認

直近の判断や revision の流れを追いたいときは、artifact と合わせて次を使う。

```bash
git log --oneline --decorate -20
```

特定ファイルの変更履歴を見るときの例:

```bash
git log --oneline -- docs/roadmap.md
```

### 2.3 commit

この repo では、smoke の途中状態ではなく、意味のあるまとまりで commit する。

```bash
git add docs/roadmap.md docs/command_reference.md
git commit -m "docs: sync roadmap after tighter policy revision gate"
```

補足:

- commit message は revision 名、profile 名、または変更の意図が追える形にする。
- 正式判断に紐づく変更なら、`r20260326_tighter_policy_ratio003` のような revision slug を含めてもよい。

### 2.4 push

shared remote が使える作業では、push までを完了条件に含める。

```bash
git push origin <branch>
```

権限や認証で push できない場合は、その場で止めずに理由を記録して共有する。

## 3. まず使うコマンド

### 3.1 データ取り込み

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_ingest.py --config configs/data.yaml
```

関連:

- [../scripts/run_validate_data_sources.py](../scripts/run_validate_data_sources.py)
- [../scripts/run_feature_gap_report.py](../scripts/run_feature_gap_report.py)
- [../scripts/run_netkeiba_coverage_snapshot.py](../scripts/run_netkeiba_coverage_snapshot.py)

### 3.2 学習

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_best_eval
```

2025 backfill 済みデータで回すときは、同じ profile family に `_2025_latest` を付ける。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_best_eval_2025_latest
```

recent-heavy な学習 window を試すときは、`_2025_recent_2018` または `_2025_recent_2020` を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_recommended_serving_2025_recent_2018
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_recommended_serving_2025_recent_2020
```

profile 一覧:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --list-profiles
```

### 3.3 評価

正式判断の読み筋は [evaluation_guide.md](evaluation_guide.md) を参照する。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --profile current_best_eval --max-rows 120000
```

最新 backfill を反映した holdout で評価するときの代表例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --profile current_best_eval_2025_latest --max-rows 120000
```

昇格判断の基本は、短窓の単発結果ではなく `stability_assessment=representative` を満たす評価である。

### 3.4 予測

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_predict.py --profile current_best_eval --race-date 2021-07-31
```

最新データ側の race day を使うときの例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_predict.py --profile current_recommended_serving_2025_latest --race-date 2025-12-28
```

補足:

- `run_predict.py` 自体には `--artifact-suffix` はなく、prediction artifact は date 基準の canonical 名で出る。
- window 単位の provenance を残したいときは、predict 単体ではなく `run_serving_smoke.py` 側で `--artifact-suffix` と `--output-file` を明示する。

### 3.5 バックテスト

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backtest.py --profile current_best_eval
```

2025 latest split に合わせた prediction artifact を使う場合も、同じ `_2025_latest` profile を指定してよい。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backtest.py --profile current_best_eval_2025_latest
```

### 3.6 current stable `_2025_latest` family

`_2025_latest` suffix は `data_2025_latest.yaml` を付けた派生 profile を広く作るが、2026-03-27 時点で current stable family として日常的に参照するのは次の 4 本である。

| profile | 種別 | 代表コマンド | 代表 artifact / window |
| --- | --- | --- | --- |
| `current_best_eval_2025_latest` | evaluation mainline | `run_train.py`, `run_evaluate.py`, `run_revision_gate.py` | latest formal evaluation / revision gate |
| `current_recommended_serving_2025_latest` | operational baseline | `run_predict.py`, `run_serving_smoke.py`, `run_serving_profile_compare.py` | `current_recommended_serving_2025_latest_<window_or_purpose>` |
| `current_long_horizon_serving_2025_latest` | seasonal de-risk alias | `run_serving_profile_compare.py` | `sep_full_month_2025_latest_profile`, `dec_tail_2025_latest_profile` |
| `current_tighter_policy_search_candidate_2025_latest` | analysis-first defensive candidate | `run_revision_gate.py`, `run_serving_profile_compare.py`, `run_wf_threshold_sweep.py` | `r20260326_tighter_policy_ratio003`, `r20260327_tighter_policy_ratio003_abs80` |

補足:

- `current_bankroll_candidate_2025_latest`、`current_ev_candidate_2025_latest`、`current_sep_guard_candidate_2025_latest` も profile 解決自体はできる。
- ただし current latest docs の主導線では使わず、必要なときだけ legacy mitigation / alias 系として参照する。
- `current_sep_guard_candidate_2025_latest` 相当の役割は、latest docs では `current_long_horizon_serving_2025_latest` という名称で扱う。

## 4. 正式な revision 評価

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

既存の学習済み artifact を再利用した threshold-only revision 例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs80.yaml \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --revision r20260327_tighter_policy_ratio003_abs80 \
  --train-artifact-suffix r20260327_tighter_policy_ratio003_abs80 \
  --skip-train \
  --evaluate-model-artifact-suffix r20260326_tighter_policy_ratio003 \
  --evaluate-pre-feature-max-rows 300000 \
  --evaluate-max-rows 120000 \
  --evaluate-wf-mode full \
  --evaluate-wf-scheme nested \
  --promotion-min-feasible-folds 3
```

関連:

- [../scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
- [../scripts/run_promotion_gate.py](../scripts/run_promotion_gate.py)
- [development_flow.md](development_flow.md)
- [evaluation_guide.md](evaluation_guide.md)

補足:

- `run_revision_gate.py` は train、evaluate、promotion gate の各段階を progress 付きで出力する。
- 現在の `run_revision_gate.py` は evaluate の後に matching `wf_feasibility_diag` も自動実行するので、promotion gate に必要な WF summary を別途手で作らなくてよい。
- `--dry-run` を付けると、重い train / evaluate を実行せずに planned command と revision manifest だけを確認できる。
- `--train-max-train-rows` と `--train-max-valid-rows` を使うと、real run でも lightweight smoke を組める。
- `--skip-train` と `--evaluate-model-artifact-suffix` を組み合わせると、設定だけ変えた threshold-only revision を既存学習済み model artifact に対して formal 評価できる。

## 5. serving 検証

基本の流れと各 artifact の読み方は [serving_validation_guide.md](serving_validation_guide.md) を参照する。

### 5.1 smoke

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py --profile current_recommended_serving --date 2024-09-14
```

latest data の末尾 window をまとめて見る例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py \
  --profile current_recommended_serving_2025_latest \
  --artifact-suffix current_recommended_serving_2025_latest_tail_dec_window \
  --output-file artifacts/reports/serving_smoke_current_recommended_serving_2025_latest_tail_dec_window.json \
  --date 2025-12-06 \
  --date 2025-12-07 \
  --date 2025-12-13 \
  --date 2025-12-14 \
  --date 2025-12-20 \
  --date 2025-12-21 \
  --date 2025-12-27 \
  --date 2025-12-28
```

この latest tail window では `2025-12-06`、`2025-12-20`、`2025-12-27` で `policy_bets=1` を確認した。

artifact 命名の実務ルール:

- latest baseline の window artifact は `current_recommended_serving_2025_latest_<window_or_purpose>` の形で suffix を切る。
- `run_serving_smoke.py` の summary 既定ファイル名は `serving_smoke_<profile>.json` で、`--artifact-suffix` だけでは変わらない。
- そのため window ごとに summary を分けたいときは、`--artifact-suffix` と同じ slug を `--output-file artifacts/reports/serving_smoke_<slug>.json` にも入れる。
- `--prediction-backend replay-existing` を使うと canonical prediction CSV を再利用しつつ、suffix 付き replay backtest artifact だけを増やせる。

### 5.2 2 候補比較

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
- suffix 付き true retrain model を比較したいときは `--left-model-artifact-suffix` / `--right-model-artifact-suffix` を使う。
- この用途では `--prediction-backend fresh` が必要で、`replay-existing` では canonical prediction CSV を再利用するだけになる。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving_2025_latest \
  --right-profile current_recommended_serving_2025_recent_2018 \
  --right-model-artifact-suffix r20260327_recent_2018_component_retrain \
  --prediction-backend fresh \
  --date 2025-09-06 \
  --date 2025-09-07 \
  --date 2025-09-13 \
  --date 2025-09-14 \
  --date 2025-09-20 \
  --date 2025-09-21 \
  --date 2025-09-27 \
  --date 2025-09-28 \
  --window-label sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh \
  --run-bankroll-sweep \
  --run-dashboard
```

共通注意:

- `--output`, `--output-file`, `--summary-path`, `--summary-csv`, `--manifest-output` のような引数に directory を渡すと fail-fast する。
- 逆に `--output-dir` のような directory 前提の引数には file path を渡さない。

### 5.3 latest 2025 の候補群をどう比較するか

latest 2025 系は、まず `current_recommended_serving_2025_latest` を baseline に固定し、September difficult window で defensive candidate を横比較し、そのあと December tail の control window で baseline 維持を確認する順に進める。

実務上の参照順は次で固定する。

1. `current_long_horizon_serving_2025_latest`
2. `current_tighter_policy_search_candidate_2025_latest`
3. `current_recommended_serving_2025_recent_2018` true retrain

理由は次のとおりである。

- `current_long_horizon_serving_2025_latest` は baseline path を最も崩さない seasonal de-risk alias で、実運用寄りの比較起点になる。
- `current_tighter_policy_search_candidate_2025_latest` は latest 2025 の formal support を持つ defensive candidate で、support 改善と actual-date de-risk をあわせて読める。
- recent-2018 true retrain は September difficult window では strongest de-risk 側だが、December tail では baseline に劣後するため broad replacement 判定には使わない。

September difficult window で long-horizon alias を最初に見る例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving_2025_latest \
  --right-profile current_long_horizon_serving_2025_latest \
  --prediction-backend replay-existing \
  --date 2025-09-06 \
  --date 2025-09-07 \
  --date 2025-09-13 \
  --date 2025-09-14 \
  --date 2025-09-20 \
  --date 2025-09-21 \
  --date 2025-09-27 \
  --date 2025-09-28 \
  --window-label sep_full_month_2025_latest_profile \
  --run-bankroll-sweep \
  --run-dashboard
```

September difficult window で tighter policy candidate を fresh compare する例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving_2025_latest \
  --right-profile current_tighter_policy_search_candidate_2025_latest \
  --prediction-backend fresh \
  --date 2025-09-06 \
  --date 2025-09-07 \
  --date 2025-09-13 \
  --date 2025-09-14 \
  --date 2025-09-20 \
  --date 2025-09-21 \
  --date 2025-09-27 \
  --date 2025-09-28 \
  --window-label sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh \
  --run-bankroll-sweep \
  --run-dashboard
```

December tail の control window で baseline 維持を確認する例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving_2025_latest \
  --right-profile current_tighter_policy_search_candidate_2025_latest \
  --prediction-backend fresh \
  --date 2025-12-06 \
  --date 2025-12-07 \
  --date 2025-12-13 \
  --date 2025-12-14 \
  --date 2025-12-20 \
  --date 2025-12-21 \
  --date 2025-12-27 \
  --date 2025-12-28 \
  --window-label dec_tail_2025_latest_vs_tighter_policy_candidate_fresh \
  --run-bankroll-sweep \
  --run-dashboard
```

recent-2018 true retrain を September difficult window に載せるときは、既存の canonical prediction を再利用せず、suffix 付き model artifact を fresh 推論で読む。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving_2025_latest \
  --right-profile current_recommended_serving_2025_recent_2018 \
  --right-model-artifact-suffix r20260327_recent_2018_component_retrain \
  --prediction-backend fresh \
  --date 2025-09-06 \
  --date 2025-09-07 \
  --date 2025-09-13 \
  --date 2025-09-14 \
  --date 2025-09-20 \
  --date 2025-09-21 \
  --date 2025-09-27 \
  --date 2025-09-28 \
  --window-label sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh \
  --run-bankroll-sweep \
  --run-dashboard
```

判断ルール:

- September difficult window では、baseline より net / bankroll の損失圧縮があるかを見る。
- December tail のような control window では、candidate が formal に通っていても baseline 優位を崩さないかを見る。
- threshold frontier の改善と actual-date role の変更は分けて読む。`0.03/80` のような formal support 拡張だけで serving default は切り替えない。

### 5.4 latest 2025 quick artifact map

latest 2025 の compare を回したあと、まず見る artifact は次の 4 系統だけでよい。

| 見たいもの | まず見る artifact |
| --- | --- |
| baseline vs long-horizon の September 読み | `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_sep_full_month_2025_latest_profile.json` |
| baseline vs long-horizon の December control 読み | `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_dec_tail_2025_latest_profile.json` |
| baseline vs tighter policy candidate の September / December 読み | `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh_vs_current_tighter_policy_search_candidate_2025_latest_sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh.json` と `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_vs_tighter_policy_candidate_fresh_vs_current_tighter_policy_search_candidate_2025_latest_dec_tail_2025_latest_vs_tighter_policy_candidate_fresh.json` |
| baseline vs recent-2018 true retrain の September / December 読み | `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh_vs_current_recommended_serving_2025_recent_2018_sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh.json` と `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_vs_recent2018_true_retrain_fresh_vs_current_recommended_serving_2025_recent_2018_dec_tail_2025_latest_vs_recent2018_true_retrain_fresh.json` |

formal support 側を見たいときだけ、次に降りる。

- baseline: `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- tighter policy: `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json` と `artifacts/reports/promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
- recent-2018: `artifacts/reports/promotion_gate_r20260327_recent_2018_component_retrain.json`

## 6. netkeiba 系の代表コマンド

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
- 重い latest / backfill 系 evaluate では `run_netkeiba_latest_revision_gate.py --evaluate-pre-feature-max-rows 300000`、または `run_netkeiba_benchmark_gate.py --pre-feature-max-rows 300000` を使うと、feature build 前に row 数を抑えられる。

## 7. 補助コマンド

### 7.1 A/B 比較

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_ab_compare.py \
  --base-profile current_best_eval \
  --challenger-profile current_recommended_serving \
  --max-rows 30000
```

### 7.2 ダッシュボード生成

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_dashboard.py
```

### 7.3 value stack tuning

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_tune_value_stack.py \
  --summary-path artifacts/reports/tune_value_stack_summary.json
```

### 7.4 進捗表示の軽量 smoke

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
/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_sweep.py \
  --wf-summary artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_wf_full_nested.json \
  --min-bet-ratio-values 0.03 \
  --min-bets-abs-values 100,80 \
  --output artifacts/reports/wf_threshold_sweep_current_tighter_policy_search_candidate_2025_latest_ratio003_100_vs_80.json \
  --summary-csv artifacts/reports/wf_threshold_sweep_current_tighter_policy_search_candidate_2025_latest_ratio003_100_vs_80.csv

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
- `run_wf_threshold_sweep.py` は `min_bet_ratio` と `min_bets_abs` を同時に変える frontier 比較に使う。`0.03/100` と `0.03/80` のような compound point 比較はまずこちらで読む。
- `run_wf_threshold_compare.py` だけは fold 集計を再生成するため、他より少し重いが再学習や推論は伴わない。
- `run_wf_threshold_compare.py` は source summary 側の baseline `min_bet_ratio` を引き継ぐので、`min_bets_abs` 単独の見比べには向くが、ratio と absolute threshold を同時に変える比較の正本にはしない。

## 8. artifact の見方

主要な出力先:

- 学習モデル: `artifacts/models/`
- train / evaluate / backtest report: `artifacts/reports/`
- prediction: `artifacts/predictions/`
- dashboard: `artifacts/reports/dashboard/`

特に正式判断で見るもの:

- `artifacts/reports/evaluation_summary.json`
- `artifacts/reports/evaluation_manifest.json`
- `artifacts/reports/promotion_gate_report.json` または `promotion_gate_<revision>.json`

## 9. 補足

- 高頻度で使う script 以外は [../scripts](../scripts) を起点に探す。
- 用途別の script 一覧は [scripts_guide.md](scripts_guide.md) を参照する。
- 詳しい運用ルールは [development_flow.md](development_flow.md) を参照する。
- benchmark の判断基準は [benchmarks.md](benchmarks.md) を参照する。
- artifact の見方は [artifact_guide.md](artifact_guide.md) を参照する。
- GPU / Docker / Notebook 周りは [environment_notes.md](environment_notes.md) を参照する。