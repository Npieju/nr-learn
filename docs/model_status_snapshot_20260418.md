# モデル指標 snapshot 2026-04-18

## 位置づけ

- 本ファイルは 2026-04-18 時点の JRA / NAR 現況をまとめた snapshot である。
- JRA は public benchmark の current snapshot を正本として読む。
- NAR は separate universe の readiness track として読み、JRA の primary KPI 判定には混ぜない。
- 数値の意味づけは `docs/public_benchmark_operational_reading_guide.md` と `docs/github_issue_queue_current.md` を優先する。

## 要約

- JRA は `current_recommended_serving_2025_latest` が引き続き operational default で、AUC `0.8401`、top1 ROI `0.8070`、nested WF weighted ROI `0.7628` を維持している。
- JRA では defensive candidate と formal promoted line も存在するが、bet rate が `3%` 台に留まるため、ROI だけで default 置換を正当化できる段階ではない。
- NAR は `#101` formal rerun により、AUC `0.8625`、top1 ROI `0.8402`、nested WF weighted ROI `3.9661` を確認できている。
- 一方で NAR の architecture parity 本線である `#103` value-blend bootstrap は現時点で baseline 未満で、promotion gate は `hold` のままである。
- 現状の最重要論点は、JRA では low-frequency な ROI をどこまで信頼できるか、NAR では provenance を維持したまま `#101` baseline を超える architecture / policy line を作れるかである。

## JRA 現状

### current line

| line | role | AUC | top1 ROI | EV top1 ROI | nested / formal ROI | bets / races | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `current_recommended_serving_2025_latest` | operational default | `0.8401` | `0.8070` | `0.5568` | nested WF `0.7628` | `544 / 16215 = 3.35%` | current default |
| `r20260326_tighter_policy_ratio003` | analysis-first defensive candidate | `0.8401` | `0.8070` | `0.5568` | formal benchmark `1.1728` | `500 / 16215 = 3.08%` | `promote`, representative |
| `r20260330_surface_plus_class_layoff_interactions_v1` | formal promoted, not default | `0.8416` | `0.8002` | `0.7100` | formal benchmark `1.1380` | `519 / 16215 = 3.20%` | formal pass, conservative role |

### 良好な点

- default line が formal gate を通過した状態で維持できており、AUC と ROI が大きく崩れていない。
- tighter policy 系は same-model のまま formal benchmark ROI `1.1728` を出しており、難しい regime 向けの defensive option を持てている。
- `surface_plus_class_layoff` は AUC `0.8416` と EV top1 ROI `0.7100` を示しており、feature family 改善が formal line まで届くことは確認できている。

### JRA の統計的な見方

- AUC や logloss の読みには、`n_rows=120000`、`n_races=8702`、nested WF では `test_races_total=16215` という母数があるため、順位付け能力そのものをゼロから疑う段階ではない。
- ただし policy ROI の読みは別で、current default の nested WF bets は `544`、強候補でも `500-519` bets に留まる。これは policy の期待値を強く主張するにはまだ細い。
- さらに JRA 側は candidate を policy search と regime compare のあとで選んでいるため、見かけの ROI には選択バイアスが乗る。単純に `500 bets あるから十分` とは言いにくい。
- したがって現時点の JRA は、「AUC は一定の再現性を示しているが、ROI はまだ provisional」であり、過学習や window luck の可能性を十分には捨てきれない、というのが妥当な読みである。

### 今後の課題

- 強い候補が出ても bet rate が `3%` 台と低く、broad default へ昇格できるだけの運用安定性が不足している。
- September difficult window では defensive option が有効でも、December tail のような control window では baseline 優位が残るため、全期間置換の根拠にはまだ弱い。
- public 説明では default line と formal promoted line を分けて扱う必要があり、単純な `weighted ROI` の大小だけで運用判断できない。
- 特に JRA の対外説明では、「モデル品質は維持しているが、投資可能な edge の統計的裏付けはまだ薄い」と表現するのが安全である。

## NAR 現状

### current line

| line | role | AUC | top1 ROI | EV top1 ROI | nested WF weighted ROI | bets | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `#101 r20260415_local_nankan_pre_race_ready_formal_v1` | trust-ready baseline reference | `0.8625` | `0.8402` | `2.1111` | `3.9661` | `778` | completed |
| `#103 r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1` | architecture bootstrap candidate | `0.7386` | `0.7956` | `0.4710` | `0.6579` | `3429` | completed, `hold` |

### 良好な点

- `#120` strict provenance trust gate と `#101` formal rerun により、historical trust-ready corpus 上で NAR benchmark を再度読める状態までは到達している。
- baseline reference の `#101` は AUC `0.8625`、top1 ROI `0.8402`、nested WF weighted ROI `3.9661` と、separate universe の readiness benchmark としては十分に強い。
- NAR は JRA と違って bets の分母が細すぎるわけではなく、`#101` では `778` bets を伴って nested WF が完走している。

### 今後の課題

- `#103` value-blend bootstrap は AUC `0.7386`、nested WF weighted ROI `0.6579` に留まり、current baseline `#101` を大きく下回っている。
- fast gate でも `wf_feasible_fold_count=0`、`formal_benchmark_weighted_roi=null` で `hold` になっており、architecture parity は未達である。
- NAR は依然として separate universe の readiness track であり、JRA benchmark の代替や primary KPI 判定には使えない。
- provenance の trust read は改善したが、historical source timing 側には `#121` の unresolved caution が残るため、対外説明では過去 ROI を trust-carrying evidence として過大評価しないほうがよい。

## 総評

JRA は「運用基準線は維持できているが、より強い候補を default に昇格させるだけの bet-rate と regime 一貫性、そして統計的な確信がまだ足りない」段階である。NAR は「benchmark reference は作れたが、その上に JRA 相当の architecture parity line を積み上げる段階で失速している」状態で、課題の中心は `#103` の support / policy / train-time state の切り分けにある。

したがって 2026-04-18 時点の読みは次の 2 行に要約できる。

- JRA は current default 維持が妥当で、強候補は defensive / conservative role に留める。
- JRA は current default 維持が妥当で、強候補は defensive / conservative role に留める。ROI の対外説明は控えめにする。
- NAR は `#101` baseline を benchmark reference としつつ、`#103` の formal parity 未達を最優先課題として扱う。

## 出典

- `docs/public_benchmark_operational_reading_guide.md`
- `docs/public_benchmark_snapshot.md`
- `docs/github_issue_queue_current.md`
- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260326_tighter_policy_ratio003_wf_full_nested.json`
- `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
- `artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json`
- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json`
- `artifacts/reports/promotion_gate_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_fast_gate.json`