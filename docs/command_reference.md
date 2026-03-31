# コマンドリファレンス

## 1. この文書の役割

この文書は、`nr-learn` の主要 CLI を用途別に引けるようにした実行リファレンスである。

網羅的な内部仕様は各 script の `--help` と実装を参照し、この文書では日常運用で使う入口だけを整理する。

長時間かかるコマンドは、現在は `ProgressBar` または `Heartbeat` により途中経過が見える前提で整備している。

長時間 run の確認先は terminal ではなく log file を優先する。

- `run_revision_gate.py` の既定 live log: `artifacts/logs/revision_gate_<revision>.log`
- `run_local_revision_gate.py` の既定 live log: `artifacts/logs/local_revision_gate_<revision>.log`
- 任意 path に変えたいときだけ `--log-file` を付ける

CLI の基本挙動:

- operator 向け script は、config 不足、入力不足、output path の取り違えをなるべく早い段階で検出する。
- 想定内の失敗は concise な `failed: ...` で返し、unexpected exception のときだけ traceback を出す。
- `output file` を受ける引数には file path を渡し、`output dir` を受ける引数には directory path を渡す。

tail loader の aggressive 最適化を試す前の equivalence 確認には、次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_tail_loader_equivalence.py \
  --raw-dir data/raw \
  --tail-rows 10000 \
  --left-reader current \
  --right-reader deque_trim
```

`--fail-on-diff` を付けると non-zero exit で gate に使える。判定基準は `--fail-gate` で選ぶ。

- `exact`: raw frame が完全一致のときだけ pass
- `canonical`: raw frame が完全一致、または canonical dtype drift のみなら pass
- `value`: 値差分がなければ dtype drift があっても pass

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_tail_loader_equivalence.py \
  --raw-dir data/raw \
  --tail-rows 10000 \
  --left-reader current \
  --right-reader deque_trim \
  --fail-on-diff \
  --fail-gate canonical
```

reduced smoke の summary 同値を確認するときは、次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_summary_equivalence.py \
  --left-summary artifacts/reports/evaluation_summary_left.json \
  --right-summary artifacts/reports/evaluation_summary_right.json \
  --fail-on-diff
```

`output_files` と `run_context` の artifact suffix / manifest path は標準で無視する。

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

### 3.0 latest 2025 actual-date compare の再開順

latest 2025 の compare を再開するときは、いきなりコマンドを打たず次の順で進める。

1. まず [serving_validation_guide.md](serving_validation_guide.md) の dashboard summary JSON 一覧を見て、September difficult window を `long_horizon -> tighter policy -> recent-2018` の順で読む。
2. 差分を回し直す必要があるときだけ、この文書の latest 2025 compare コマンド例を同じ順で使う。
3. support の裏付けまで掘る必要があるときだけ [evaluation_guide.md](evaluation_guide.md) と promotion gate artifact に降りる。

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

resource-safe rerun を強制したいときは、`run_train.py` の既定 preflight をそのまま使う。重い train / evaluate / revision gate / local collection が並走していると、CLI は fail-fast で停止する。どうしても override が必要なときだけ `--allow-concurrent-heavy-jobs` を付ける。

rerun 前に quiet heavy-job lane が空いているかだけ確認したいときは、次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_execution_capacity_status.py \
  --script-pattern scripts/run_train.py \
  --output artifacts/reports/execution_capacity_status_train.json
```

空くまで待ってから次へ進めたいときは、次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_execution_capacity_wait.py \
  --script-pattern scripts/run_train.py \
  --timeout-seconds 3600 \
  --poll-interval-seconds 30 \
  --output artifacts/reports/execution_capacity_wait_train.json
```

quiet lane が空いた瞬間にそのまま train を始めたいときは、次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train_when_capacity_ready.py \
  --timeout-seconds 3600 \
  --poll-interval-seconds 30 \
  --wait-output artifacts/reports/execution_capacity_wait_train.json \
  -- \
  --config configs/model_catboost_win_high_coverage_diag.yaml \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_regime_extension.yaml \
  --artifact-suffix r20260330_jockey_trainer_combo_style_distance_v1
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

live progress は既定で `artifacts/logs/revision_gate_r20260321a.log` にも書かれる。

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
- VS Code から進捗を追うときは `artifacts/logs/revision_gate_<revision>.log` を開く。
- 既定 path を変えたいときだけ `--log-file <path>` を付ける。
- `--dry-run` を付けると、重い train / evaluate を実行せずに planned command と revision manifest だけを確認できる。planned manifest には `read_order`, `current_phase`, `recommended_action`, `highlights` も入る。
- `--train-max-train-rows` と `--train-max-valid-rows` を使うと、real run でも lightweight smoke を組める。
- `--skip-train` と `--evaluate-model-artifact-suffix` を組み合わせると、設定だけ変えた threshold-only revision を既存学習済み model artifact に対して formal 評価できる。

supplemental materialize の初期コマンド:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_materialize_supplemental_table.py \
  --data-config configs/data_2025_latest.yaml \
  --table-name corner_passing_order \
  --manifest-file artifacts/reports/supplemental_materialize_corner_passing_order.json
```

`netkeiba_race_card` を materialized path として再生成する例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_materialize_supplemental_table.py \
  --data-config configs/data_2025_latest.yaml \
  --table-name netkeiba_race_card \
  --output-file data/processed/supplemental/netkeiba_race_card.csv \
  --manifest-file artifacts/reports/supplemental_materialize_netkeiba_race_card.json
```

materialized supplemental path を opt-in で評価する smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py \
  --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_kelly_runtime_edge005.yaml \
  --data-config configs/data_2025_latest_materialized_corner.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --model-artifact-suffix r20260326_tighter_policy_ratio003 \
  --artifact-suffix perf_smoke_materialized_corner_v1 \
  --max-rows 5000 \
  --pre-feature-max-rows 10000 \
  --wf-mode fast \
  --wf-scheme nested
```

補足:

- `scripts/run_materialize_supplemental_table.py` は progress と manifest を出し、`--table-kind supplemental|append|auto` で config table を再利用用 CSV に前展開できる。
- append table の例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_materialize_supplemental_table.py \
  --data-config configs/data_2025_latest.yaml \
  --table-name netkeiba_race_result \
  --table-kind append \
  --output-file data/processed/append/netkeiba_race_result.csv \
  --manifest-file artifacts/reports/materialize_netkeiba_race_result_append.json
```

- primary tail cache の例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_primary_tail_cache_status.py \
  --data-config configs/data_2025_latest_primary_tail_cache.yaml \
  --tail-rows 10000
```

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_primary_tail_cache_refresh_if_needed.py \
  --data-config configs/data_2025_latest_primary_tail_cache.yaml \
  --tail-rows 10000
```

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_materialize_primary_tail_cache.py \
  --data-config configs/data_2025_latest_primary_tail_cache.yaml \
  --tail-rows 10000 \
  --output-file data/processed/primary/race_result_tail10000_exact.pkl \
  --manifest-file artifacts/reports/primary_tail_cache_tail10000.json
```

- status command の exit code は `fresh=0`, `stale/missing/tail_mismatch/cache_invalid=2`, `not_configured=1` である。
- `run_primary_tail_cache_refresh_if_needed.py` は `fresh` なら no-op、`stale/missing/...` なら materialize を実行して再判定する。
- refresh runbook は `status -> refresh_if_needed -> 必要なら reduced smoke` の順で使う。
- `primary_tail_cache_file` / `primary_tail_cache_manifest_file` を data config に追加すると、`load_training_table_tail(...)` は requested `tail_rows` と一致する cache manifest がある場合に pickle cache を優先する。
- [configs/data_2025_latest.yaml](/workspaces/nr-learn/configs/data_2025_latest.yaml) は primary tail cache を default mainline として含む。
- tracked candidate config [configs/data_2025_latest_primary_tail_cache.yaml](/workspaces/nr-learn/configs/data_2025_latest_primary_tail_cache.yaml) は historical A/B と explicit alias 用に残している。
- cache manifest には `source_dataset`, `source_dataset_size_bytes`, `source_dataset_mtime_ns` が入り、raw primary source が変わった場合は stale cache とみなして raw tail read へ fallback する。
- default config は `data/processed/supplemental/corner_passing_order.csv` を優先し、存在しない場合だけ raw supplemental CSV へ fallback する。
- default config は `data/processed/supplemental/netkeiba_race_card.csv` と `data/processed/supplemental/netkeiba_race_result_keys.csv` も優先し、存在しない場合だけ raw source へ fallback する。
- `configs/data_2025_latest_materialized_corner.yaml` は A/B や追加検証用の明示 config として残している。
- `configs/data_2025_latest_materialized_racecard.yaml` は historical A/B 記録用に残しているが、`netkeiba_race_card` 自体はすでに default mainline に昇格済みである。
- `configs/data_2025_latest_materialized_result_keys.yaml` は `netkeiba_race_result_keys` materialized path の A/B 記録用 config であり、source 自体はすでに default mainline に昇格済みである。

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
- `current_tighter_policy_search_candidate_2025_latest` は latest 2025 の同一 family 上で exposure を絞る defensive candidate で、実運用の比較順としては recent-heavy retrain より一段単純である。
- recent-2018 true retrain は September difficult window では strongest de-risk 側だが、学習窓の再構成を伴うため current reading では analysis-first fallback に置く。December tail でも broad replacement 判定には使わない。

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

local_nankan ID 準備の初期 smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_prepare_local_nankan_ids.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --race-id-source race_list \
  --target race_result \
  --start-date 2026-03-25 \
  --end-date 2026-03-27 \
  --limit 50
```

seed CSV を使う従来 smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_prepare_local_nankan_ids.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --seed-file data/external/local_nankan/seeds/local_nankan_seed.csv \
  --target race_result \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --limit 50
```

補足:

- `run_prepare_local_nankan_ids.py` は provider crawler 実装前の入口として、operator が用意した seed CSV から `race_ids.csv` と `horse_keys.csv` を生成する。
- `--race-id-source race_list` を使うと、`calendar/<quarter>.do` と `program/<meeting_id>.do` から date range 内の `race_id` を直接 discovery できる。
- 初期 smoke では `race_result` または `race_card` から始め、pedigree は seed に `horse_key` 列が揃ってから有効化する。

local_nankan collect planning の初期 smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_collect_local_nankan.py \
  --config configs/crawl_local_nankan_template.yaml \
  --target race_result \
  --limit 50 \
  --dry-run
```

補足:

- `run_collect_local_nankan.py --dry-run` は target ごとの ID 件数、output path、manifest path を planned shape で書き出す。
- provider fetch 本体はまだ未実装なので、`--dry-run` なしの実行は blocked manifest を返し、次アクションを `implement_source_provider` として残す。

local_nankan backfill planning の初期 smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backfill_local_nankan.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --seed-file data/external/local_nankan/seeds/local_nankan_seed.csv \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --limit 50 \
  --dry-run
```

補足:

- `run_backfill_local_nankan.py --dry-run` は 1 cycle 分の `prepare -> collect` を planned shape で manifest 化する。
- provider 実装前の通常実行は blocked manifest を返すが、後で fetch 実装を差し込む場所を backfill 単位で固定できる。

local_nankan を 6 か月ずつ順送りで backfill する長期取得:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backfill_local_nankan.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --data-config configs/data_local_nankan.yaml \
  --race-id-source race_list \
  --start-date 2006-03-29 \
  --end-date 2026-03-28 \
  --date-order asc \
  --chunk-months 6 \
  --max-date-windows 1 \
  --sleep-sec-between-windows 2.0 \
  --manifest-file artifacts/reports/local_nankan_backfill_20y_windowed.json \
  --materialize-after-collect
```

補足:

- `--chunk-months 6` を付けると、指定した全 date range を 6 か月ごとの window に分割して順に実行する。
- `2006-03-29 .. 2026-03-28` は 40 window に分割されるので、`--max-date-windows 1` なら 1 回の起動で 6 か月分だけ進む。
- `local_nankan` crawl config の既定 `delay_sec` は 0.05 秒まで下げてある。大規模 backfill を前提にしているので、明示的に上げない限り低 delay のまま回る。
- `run_collect_local_nankan.py` と `run_backfill_local_nankan.py` は `--delay-sec`, `--timeout-sec`, `--retry-count`, `--retry-backoff-sec`, `--overwrite/--no-overwrite` で request 設定を runtime override できる。config を編集せず、その場の回収条件だけ切り替えたいときに使う。
- aggregate manifest には `requested_window_count`, `executed_window_count`, `window_reports[]` が入り、各 window の個別 manifest も `..._windowNNN_YYYYMMDD_YYYYMMDD.json` として残る。
- 同じ aggregate manifest path で再実行すると、既に `status=completed` になっている window は自動で skip され、次の未完了 window から再開する。
- 実行中の aggregate manifest も逐次更新され、`active_window`, `active_window_date_window`, `completed_window_count`, `current_phase`, `last_updated_at` から現在位置を確認できる。window 内の個別 manifest も cycle ごとに `running` 更新される。
- 少しずつ取得したい場合は `--date-order asc` を基本にして古い window から順送りにする。新しい側から埋めたいときだけ `desc` を使う。
- `--materialize-after-collect` を併用すると、各 window 後に primary raw materialize まで進められる。collect 出力は累積しつつ、window ごとの materialize manifest も残る。
- 標準出力でも window 開始・完了・sleep が出るが、長時間実行では manifest を見るほうが確実である。
- さらに cycle ごとに `phase=prepare_ids_completed` と `phase=collect_completed` の要約が出る。`pending_breakdown=horse_keys=8, race_ids=0` のように残件の内訳が見えるので、何がボトルネックかを terminal だけでも追いやすい。
- `--target race_result` または `--target race_card` を使うと、長期 backfill を race 系だけ先に進められる。pedigree が長く残る場合は、先に race 系を回収してから `--target pedigree` を別運用にするのが扱いやすい。

local_nankan を 6 か月ずつ回しつつ window 単位で GitHub 向け archive を作る入口:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_archive_local_nankan_window.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --data-config configs/data_local_nankan.yaml \
  --race-id-source race_list \
  --target race_result \
  --start-date 2006-03-29 \
  --end-date 2026-03-28 \
  --date-order asc \
  --chunk-months 6 \
  --max-date-windows 1 \
  --manifest-file artifacts/reports/local_nankan_backfill_race_result_20y.json \
  --archive-root artifacts/archives/local_nankan \
  --archive-index artifacts/archives/local_nankan/archive_index.json
```

補足:

- `run_archive_local_nankan_window.py` は先に `run_backfill_local_nankan.py` を 1 window ぶん実行し、その run で新規または更新された window だけを archive 化する。
- archive 本体は `artifacts/archives/local_nankan/windowNNN_YYYYMMDD_YYYYMMDD/` に出し、同内容の `tar.gz` も `artifacts/archives/local_nankan/tarballs/` に作る。
- race 系 CSV は累積 output から当該 6 か月 window の `date` で slice して保存する。pedigree はその window の race rows で参照された `horse_key` だけを抜き出す。
- `artifacts/archives/local_nankan/archive_index.json` は積み上げ index で、各 half-year archive の window slug、期間、row count、tarball path を 1 本で追える。
- GitHub へ残す単位は、基本的に window directory、tarball、archive index の 3 つでよい。run ごとの orchestration は `artifacts/reports/local_nankan_archive_run.json` を見れば足りる。
- まず command 解決だけ確認したいときは `--dry-run` を付ける。backfill も archive も planned shape で止まり、tarball は作らない。

race 系だけを優先して 6 か月ずつ進める例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backfill_local_nankan.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --race-id-source race_list \
  --target race_result \
  --start-date 2006-03-29 \
  --end-date 2026-03-28 \
  --date-order asc \
  --chunk-months 6 \
  --max-date-windows 1 \
  --limit 64 \
  --manifest-file artifacts/reports/local_nankan_backfill_race_result_20y.json
```

local_nankan backfill から primary raw materialize まで一続きで確認する smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_backfill_local_nankan.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --data-config artifacts/tmp/local_nankan_smoke/data_local_nankan_smoke.yaml \
  --seed-file data/external/local_nankan/seeds/local_nankan_seed.csv \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --limit 50 \
  --materialize-after-collect \
  --race-result-path artifacts/tmp/local_nankan_smoke/results/local_race_result.csv \
  --race-card-path artifacts/tmp/local_nankan_smoke/racecard/local_racecard.csv \
  --pedigree-path artifacts/tmp/local_nankan_smoke/pedigree/local_pedigree.csv \
  --materialize-manifest-file artifacts/tmp/local_nankan_smoke/local_nankan_backfill_materialize.json
```

補足:

- `--materialize-after-collect` を付けると、backfill は各 cycle の `collect_summary` の後に `materialize_summary` も残す。
- provider 未実装で collect が blocked でも、既存 external outputs から primary raw が materialize できれば backfill manifest は `status=partial`, `current_phase=materialized_primary_raw`, `recommended_action=run_local_preflight` を返す。

local_nankan backfill から benchmark まで一続きで handoff する smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_local_backfill_then_benchmark.py \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --data-config artifacts/tmp/local_nankan_smoke/data_local_nankan_smoke.yaml \
  --seed-file data/external/local_nankan/seeds/local_nankan_seed.csv \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --limit 50 \
  --race-result-path artifacts/tmp/local_nankan_smoke/results/local_race_result.csv \
  --race-card-path artifacts/tmp/local_nankan_smoke/racecard/local_racecard.csv \
  --pedigree-path artifacts/tmp/local_nankan_smoke/pedigree/local_pedigree.csv \
  --wrapper-manifest-output artifacts/tmp/local_nankan_smoke/local_backfill_then_benchmark_manifest.json \
  --backfill-manifest-output artifacts/tmp/local_nankan_smoke/local_nankan_backfill_handoff_manifest.json \
  --materialize-manifest-output artifacts/tmp/local_nankan_smoke/local_nankan_primary_handoff_manifest.json \
  --preflight-output artifacts/tmp/local_nankan_smoke/data_preflight_handoff.json \
  --benchmark-manifest-output artifacts/tmp/local_nankan_smoke/benchmark_gate_handoff.json \
  --snapshot-output artifacts/tmp/local_nankan_smoke/coverage_snapshot_handoff.json \
  --skip-train \
  --skip-evaluate
```

補足:

- `run_local_backfill_then_benchmark.py` は backfill を `--materialize-after-collect` 付きで実行し、`current_phase=materialized_primary_raw` に到達したらそのまま local benchmark gate を起動する。
- wrapper manifest も `read_order`, `current_phase`, `recommended_action`, `highlights` を持つので、backfill 側で止まったのか benchmark 側で止まったのかを 1 本で追える。

local_nankan primary raw materialize の初期 smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_materialize_local_nankan_primary.py \
  --data-config configs/data_local_nankan.yaml \
  --race-result-path artifacts/tmp/local_nankan_smoke/results/local_race_result.csv \
  --race-card-path artifacts/tmp/local_nankan_smoke/racecard/local_racecard.csv \
  --pedigree-path artifacts/tmp/local_nankan_smoke/pedigree/local_pedigree.csv \
  --output-file artifacts/tmp/local_nankan_smoke/raw/local_nankan_primary.csv
```

補足:

- `run_materialize_local_nankan_primary.py` は external outputs から `data/local_nankan/raw` 相当の primary CSV を組み立てる bridge である。
- `race_result` が無い場合は `status=not_ready` で止まり、`recommended_action=populate_external_results` を返す。
- `race_card` と `pedigree` は optional enrichment として扱い、存在すれば fill する。

local_nankan benchmark gate で primary raw materialize を先に差し込む smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_local_benchmark_gate.py \
  --data-config artifacts/tmp/local_nankan_smoke/data_local_nankan_smoke.yaml \
  --race-result-path artifacts/tmp/local_nankan_smoke/results/local_race_result.csv \
  --race-card-path artifacts/tmp/local_nankan_smoke/racecard/local_racecard.csv \
  --pedigree-path artifacts/tmp/local_nankan_smoke/pedigree/local_pedigree.csv \
  --materialize-primary-before-gate \
  --materialize-manifest-file artifacts/tmp/local_nankan_smoke/local_nankan_primary_from_gate.json \
  --manifest-output artifacts/tmp/local_nankan_smoke/benchmark_gate_local_nankan.json \
  --preflight-output artifacts/tmp/local_nankan_smoke/data_preflight_local_nankan.json \
  --skip-train \
  --skip-evaluate
```

補足:

- `--materialize-primary-before-gate` を付けると、benchmark gate wrapper は preflight 前に `run_materialize_local_nankan_primary.py` を呼ぶ。
- materialize が `status=not_ready` でも wrapper は benchmark gate を続行し、最終的な blocker は通常どおり preflight manifest と benchmark manifest に残す。

local_nankan revision gate で backfill handoff を先に差し込む dry-run smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_local_revision_gate.py \
  --revision local_nankan_handoff_smoke \
  --crawl-config configs/crawl_local_nankan_template.yaml \
  --data-config artifacts/tmp/local_nankan_smoke/data_local_nankan_smoke.yaml \
  --model-config configs/model_local_baseline.yaml \
  --feature-config configs/features_local_baseline.yaml \
  --seed-file data/external/local_nankan/seeds/local_nankan_seed.csv \
  --start-date 2024-01-01 \
  --end-date 2024-01-07 \
  --limit 50 \
  --backfill-before-benchmark \
  --race-result-path artifacts/tmp/local_nankan_smoke/results/local_race_result.csv \
  --race-card-path artifacts/tmp/local_nankan_smoke/racecard/local_racecard.csv \
  --pedigree-path artifacts/tmp/local_nankan_smoke/pedigree/local_pedigree.csv \
  --lineage-output artifacts/tmp/local_nankan_smoke/local_revision_gate_handoff_dry.json \
  --backfill-wrapper-output artifacts/tmp/local_nankan_smoke/local_backfill_then_benchmark_for_revision_dry.json \
  --backfill-manifest-output artifacts/tmp/local_nankan_smoke/local_nankan_backfill_for_revision_dry.json \
  --materialize-manifest-output artifacts/tmp/local_nankan_smoke/local_nankan_primary_for_revision_dry.json \
  --benchmark-manifest-output artifacts/tmp/local_nankan_smoke/benchmark_gate_for_revision_dry.json \
  --data-preflight-output artifacts/tmp/local_nankan_smoke/data_preflight_for_revision_dry.json \
  --snapshot-output artifacts/tmp/local_nankan_smoke/coverage_snapshot_for_revision_dry.json \
  --dry-run
```

live progress は既定で `artifacts/logs/local_revision_gate_local_nankan_handoff_smoke.log` にも書かれる。

補足:

- `--backfill-before-benchmark` を付けると、revision lineage は benchmark gate の直前で `run_local_backfill_then_benchmark.py` を planned/real path に含める。
- そのため lineage manifest からも `backfill_wrapper_manifest`, `backfill_manifest`, `primary_materialize_manifest`, `benchmark_manifest` を 1 本の read_order で追える。

mixed readiness / status board で local handoff blocker を読む smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_mixed_universe_readiness.py \
  --revision local_nankan_handoff_smoke \
  --left-public-snapshot artifacts/tmp/local_nankan_smoke/local_public_snapshot_handoff_from_dry.json \
  --output artifacts/tmp/local_nankan_smoke/mixed_universe_readiness_handoff_from_dry.json

/workspaces/nr-learn/.venv/bin/python scripts/run_mixed_universe_status_board.py \
  --revision local_nankan_handoff_smoke \
  --left-universe local_nankan \
  --right-universe jra \
  --public-snapshot artifacts/tmp/local_nankan_smoke/local_public_snapshot_handoff_from_dry.json \
  --readiness-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_readiness_handoff_from_dry.json \
  --output artifacts/tmp/local_nankan_smoke/mixed_universe_status_board_handoff_from_dry.json
```

補足:

- readiness は `left_readiness_blocked` の highlights に upstream `backfill_handoff_summary` の `status`, `current_phase`, `recommended_action` も出す。
- status board は explicit path 指定を使うと tmp smoke artifact をそのまま読める。public snapshot が `status=completed` でも `current_phase=lineage_planned` を持つ場合、board は `status=partial`, `current_phase=public_snapshot` で止まる。

mixed gap audit / recovery plan で upstream handoff blocker を first step にする smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_mixed_universe_left_gap_audit.py \
  --revision local_nankan_handoff_smoke \
  --left-universe local_nankan \
  --right-universe jra \
  --numeric-compare-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_numeric_compare_handoff_from_dry.json \
  --left-lineage-manifest artifacts/tmp/local_nankan_smoke/local_revision_gate_handoff_dry.json \
  --output artifacts/tmp/local_nankan_smoke/mixed_universe_left_gap_audit_handoff_from_dry.json

/workspaces/nr-learn/.venv/bin/python scripts/run_mixed_universe_left_recovery_plan.py \
  --revision local_nankan_handoff_smoke \
  --left-universe local_nankan \
  --right-universe jra \
  --gap-audit-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_left_gap_audit_handoff_from_dry.json \
  --output artifacts/tmp/local_nankan_smoke/mixed_universe_left_recovery_plan_handoff_from_dry.json
```

補足:

- gap audit は local revision lineage に `backfill_handoff_payload` が残っていると、`current_phase=local_backfill_then_benchmark`, `recommended_action=run_local_backfill_then_benchmark` を返す。
- recovery plan もその blocker を manual first step として先頭へ並べ、`plan_steps[0].step=run_local_backfill_then_benchmark` を返す。

mixed recovery wrapper でも upstream handoff blocker を first step にする dry-run smoke:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_mixed_universe_status_board.py \
  --revision local_nankan_handoff_smoke \
  --left-universe local_nankan \
  --right-universe jra \
  --public-snapshot artifacts/tmp/local_nankan_smoke/local_public_snapshot_handoff_from_dry.json \
  --readiness-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_readiness_handoff_from_dry.json \
  --numeric-compare-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_numeric_compare_handoff_from_dry.json \
  --left-gap-audit-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_left_gap_audit_handoff_from_dry.json \
  --left-recovery-plan-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_left_recovery_plan_handoff_from_dry.json \
  --output artifacts/tmp/local_nankan_smoke/mixed_universe_status_board_handoff_from_dry_v2.json

/workspaces/nr-learn/.venv/bin/python scripts/run_mixed_universe_recovery.py \
  --revision local_nankan_handoff_smoke \
  --left-universe local_nankan \
  --right-universe jra \
  --status-board-manifest artifacts/tmp/local_nankan_smoke/mixed_universe_status_board_handoff_from_dry_v2.json \
  --output artifacts/tmp/local_nankan_smoke/mixed_universe_recovery_handoff_from_dry_v3.json \
  --dry-run
```

補足:

- explicit tmp board が `recommended_action=run_local_backfill_then_benchmark` を返している場合、recovery wrapper も first step を `local_backfill_then_benchmark` に切り替える。
- その後段で local revision lineage と mixed downstream manifests を同じ anchor から再生成する planned steps を続ける。

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
