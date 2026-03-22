# Serving 検証ガイド

## 1. この文書の役割

この文書は、`nr-learn` における serving 系の確認手順をまとめたガイドである。

対象は次の 4 つである。

1. representative date の smoke
2. 複数日 window の profile compare
3. replay ベースの軽量検証
4. bankroll / dashboard / aggregate の読み方

## 2. 基本方針

- serving 検証は、runtime の挙動確認と候補の絞り込みに使う。
- 短い smoke / compare は昇格判断そのものではなく、`development_flow.md` で定義した smoke / probe の一部として扱う。
- 正式な昇格判断は、別途 `run_evaluate.py` と `run_promotion_gate.py` を通した revision 単位で行う。

## 3. 主に使う stable profile

- `current_best_eval`
  - nested evaluation 上の主力候補
- `current_recommended_serving`
  - 現時点の簡易運用候補
- `current_bankroll_candidate`
  - 強い de-risk を狙う conservative candidate
- `current_ev_candidate`
  - `current_bankroll_candidate` よりは攻める intermediate candidate

## 4. 代表日の smoke

特定日だけ runtime 挙動を確認したいときは、まず `run_serving_smoke.py` を使う。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py \
  --profile current_recommended_serving \
  --date 2024-09-14
```

built-in case を持たない profile では `--date` が必須である。

主な出力:

- `artifacts/reports/serving_smoke_<suffix>.json`
- suffix 付き prediction / backtest artifact

## 5. 複数日 window の比較

2 候補をまとめて比較するときは `run_serving_profile_compare.py` を使う。

例:

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

この 1 run で次が揃う。

1. 左右の `serving_smoke_*.json`
2. compare JSON / CSV
3. bankroll sweep JSON / CSV
4. dashboard JSON / PNG / CSV
5. provenance 用 manifest

この provenance manifest は `serving_smoke_profile_compare_*.json` に保存され、途中 step が失敗した場合も可能な限り実行済み step と失敗位置を残す。

明示的に output path を渡す場合は、summary / compare / bankroll / dashboard の各出力とも directory ではなく file path を渡す。

## 6. replay ベースの軽量検証

shared feature build が重い window では、既存 prediction CSV を再利用する replay モードを使う。

### 6.1 smoke 側で replay を使う

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py \
  --profile current_bankroll_candidate \
  --prediction-backend replay-existing \
  --date 2024-09-16 \
  --date 2024-09-21
```

### 6.2 prediction から直接 replay summary を作る

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_replay_from_predictions.py \
  --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml \
  --prediction-files artifacts/predictions/predictions_20240916.csv \
  --artifact-suffix replay_probe
```

この方法では heavy feature rebuild を避けたまま、`serving_smoke_*` 互換の summary を作れる。

## 7. bankroll sweep の見方

bankroll 観点まで見たいときは `run_serving_stateful_bankroll_sweep.py` を使う。

ポイント:

- default の floor grid には `1.01` が含まれる
- pure stage path も別候補として常に評価する

つまり、threshold grid が足りず「常に candidate を使う」path を見落とすことはない。

見るべき項目:

1. `best_result.final_bankroll`
2. `best_result.selection_mode`
3. `best_result.selected_label`
4. pure path と baseline-only path の差

`selection_mode=pure_stage` なら、実質的に「常にその profile を使うのが最良だった」ことを意味する。

## 8. dashboard と aggregate

### 8.1 window ごとの dashboard

compare manifest から window 単位の可視化を作るときは `run_serving_compare_dashboard.py` を使う。

出力:

- summary JSON
- 日次 net / bankroll path の PNG
- 日別 CSV

### 8.2 複数 window の aggregate

期間横断で trade-off を見るときは `run_serving_compare_aggregate.py` を使う。

入力は次のどちらでもよい。

- dashboard summary 群
- compare manifest 群

aggregate では、window ごとの net delta / pure bankroll delta / best sweep bankroll を横断比較する。

## 9. 候補の読み方

現時点の保守的候補の読み方は次のとおりである。

  - strongest de-risk candidate
  - ベット数をかなり絞って bankroll を守る寄り
  - intermediate candidate
  - `current_bankroll_candidate` よりはベットするが、baseline より損失を抑えやすい

### 9.1 直近の runtime comparison

2026-03-22 に、次の 5 日で `current_recommended_serving` と `current_best_eval` を `fresh` backend で比較した。

- `2024-09-16`
- `2024-09-21`
- `2024-09-22`
- `2024-09-28`
- `2024-09-29`

実行は [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py) で行い、compare と bankroll sweep を保存した。

観測結果は次のとおりである。

- shared dates は 5/5 で全件 `ok`
- `differing_score_source_dates=[]`
- `differing_policy_dates=[]`
- total policy bets は両者とも `31`
- mean policy ROI は両者とも `0.108`
- total policy net は両者とも `-25.6`
- bankroll sweep でも pure path / threshold grid を含めて差は出なかった

つまり、この window では `current_best_eval` の追加 score override source は runtime 上の差分を生まず、`current_recommended_serving` が simpler candidate として優先しやすい。
つまり、比較の軸は単純な net だけではなく、次の 3 つで見る。

1. bets
2. total net
3. final bankroll

## 10. どこまでを serving で判断するか

serving compare で判断してよいのは、主に次である。

- runtime が壊れていないか
- policy change が actual calendar でどう効くか
- conservative candidate が bankroll をどの程度守るか

serving だけで決めないものは次である。

- 正式な昇格判断
- benchmark の代表値
- revision の確定

これらは `development_flow.md` に従って full evaluation / promotion gate で判断する。

## 11. 関連 script

- [../scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- [../scripts/run_serving_smoke_compare.py](../scripts/run_serving_smoke_compare.py)
- [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py)
- [../scripts/run_serving_replay_from_predictions.py](../scripts/run_serving_replay_from_predictions.py)
- [../scripts/run_serving_stateful_bankroll_sweep.py](../scripts/run_serving_stateful_bankroll_sweep.py)
- [../scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- [../scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)