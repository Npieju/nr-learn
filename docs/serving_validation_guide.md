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

2026-03-22 時点の aggregate JSON には、mean だけでなく window ごとの方向と分類も含める。

- `net_delta_direction`
  `right_minus_left` の net 差分の符号。`positive` / `negative` / `zero` / `unknown` を取る。
- `pure_bankroll_delta_direction`
  `right_minus_left` の pure final bankroll 差分の符号。`positive` / `negative` / `zero` / `unknown` を取る。
- `tradeoff_classification`
  net と bankroll の符号をまとめた window 単位の regime label。

実務上の解釈は次のとおりである。

- `positive_net_positive_bankroll`
  候補側が net と bankroll の両面で baseline を上回った。候補を押し上げる strongest evidence。
- `negative_net_positive_bankroll`
  候補側は bankroll 改善と引き換えに net を落としている。de-risk 候補としては有効だが、無条件の置換根拠にはならない。
- `negative_net_negative_bankroll`
  候補側が net と bankroll の両面で不利。baseline 維持を支持する counterexample として扱う。
- `positive_net_negative_bankroll`
  候補側が net は伸ばすが bankroll を悪化させる。攻め寄り候補としては意味があるが、drawdown 制御目的では注意が必要。

summary 側では `tradeoff_classification_counts` と `windows_by_tradeoff_classification` で、どの regime が何本あったかを確認する。

## 9. 候補の読み方

現時点の保守的候補の読み方は次のとおりである。

- `current_bankroll_candidate`
  strongest de-risk candidate。ベット数をかなり絞って bankroll を守る寄り。
- `current_ev_candidate`
  intermediate candidate。`current_bankroll_candidate` よりはベットするが、baseline より損失を抑えやすい。

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

### 9.2 bankroll 重視候補との比較

2026-03-22 に、同じ 5 日で `current_recommended_serving` と `current_bankroll_candidate` を `fresh` backend で比較した。

- `2024-09-16`
- `2024-09-21`
- `2024-09-22`
- `2024-09-28`
- `2024-09-29`

実行は [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py) で行い、compare と bankroll sweep を保存した。

観測結果は次のとおりである。

- shared dates は 5/5 で全件 `ok`
- `differing_score_source_dates=[]`
- `differing_policy_dates` は 5/5 で、policy は全日で分岐した
- total policy bets は `current_recommended_serving=31`、`current_bankroll_candidate=1`
- mean policy ROI は comparable な日で `current_recommended_serving=0.108`、`current_bankroll_candidate=0.0`
- total policy net は `current_recommended_serving=-25.6`、`current_bankroll_candidate=-1.0`
- pure stage の bankroll path では `current_recommended_serving=0.2780` に対し、`current_bankroll_candidate=0.9583` だった
- best sweep result も `selection_mode=pure_stage` かつ `selected_label=current_bankroll_candidate` で、threshold grid を含めても bankroll 重視では `current_bankroll_candidate` が優位だった

この window では、`current_bankroll_candidate` は期待値を取りに行くよりも、ほぼノーベットでドローダウンを抑える役割が明確だった。`current_recommended_serving` は同じ score source を使いながら、より積極的にベットして損失も大きくなった。

### 9.3 EV 重視候補との比較

2026-03-22 に、同じ 5 日で `current_recommended_serving` と `current_ev_candidate` を `fresh` backend で比較した。

- `2024-09-16`
- `2024-09-21`
- `2024-09-22`
- `2024-09-28`
- `2024-09-29`

実行は [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py) で行い、compare / bankroll sweep / dashboard を保存した。

観測結果は次のとおりである。

- shared dates は 5/5 で全件 `ok`
- `differing_score_source_dates=[]`
- `differing_policy_dates` は 5/5 で、policy は全日で分岐した
- total policy bets は `current_recommended_serving=31`、`current_ev_candidate=12`
- mean policy ROI は `current_recommended_serving=0.108`、`current_ev_candidate=0.0`
- total policy net は `current_recommended_serving=-25.6`、`current_ev_candidate=-12.0`
- pure stage の bankroll path では `current_recommended_serving=0.2780` に対し、`current_ev_candidate=0.5832` だった
- best sweep result も `selection_mode=pure_stage` かつ `selected_label=current_ev_candidate` で、threshold grid を含めてもこの window では `current_ev_candidate` を全日使う path が最良だった

この window では、`current_ev_candidate` は `current_recommended_serving` よりベット数を `31 -> 12` まで減らして損失を半減させつつ、`current_bankroll_candidate` ほど極端には絞らない中間候補として機能した。late-September のこの 5 日では、役割の並びは `current_recommended_serving` が最も攻め、`current_ev_candidate` が中間、`current_bankroll_candidate` が最も防御的と読める。

### 9.4 複数 window aggregate の読み方

2026-03-22 に、`run_serving_compare_aggregate.py` で次の 2 種類の aggregate を作った。

- `current_recommended_serving` vs `current_bankroll_candidate`
- `current_recommended_serving` vs `current_ev_candidate`

入力 window は最初に次の 2 本で始め、その後 August weekend と May weekend を足して 4 本に拡張した。

- `tail_weekends`
- late-September 5 日 window
- `aug_weekends_20260322`
- `may_weekends_20260322`

aggregate artifact は次に保存した。

- `artifacts/reports/dashboard/serving_compare_aggregate_recommended_vs_bankroll_4windows_20260322.json`
- `artifacts/reports/dashboard/serving_compare_aggregate_recommended_vs_ev_4windows_20260322.json`

観測結果は次のとおりである。

- 4-window では、`current_bankroll_candidate` の pure bankroll delta は `2/4` windows で正、mean は `net -2.6500`, `bankroll +0.1520`
- 4-window では、`current_ev_candidate` の pure bankroll delta も `2/4` windows で正、mean は `net -5.6500`, `bankroll +0.0354`
- net delta が正だったのは両候補とも late-September 側だけだった
- `tail_weekends` では `current_recommended_serving` が total net で優位だったが、両候補とも pure final bankroll は改善した
- late-September では `current_bankroll_candidate` と `current_ev_candidate` の両方が、net と bankroll の両面で `current_recommended_serving` を上回った
- August weekends では逆に `current_recommended_serving` が net と bankroll の両面で両候補を上回った
- May weekends では両候補とも `current_recommended_serving` と policy / net / bankroll が一致し、差分は観測されなかった

classification で見ると、両 aggregate とも 4 本の window は次の 4 分割になった。

- late-September は `positive_net_positive_bankroll`
- `tail_weekends` は `negative_net_positive_bankroll`
- `aug_weekends_20260322` は `negative_net_negative_bankroll`
- `may_weekends_20260322` は `zero_net_zero_bankroll`

この 4 分割が示すのは、候補側の効き方が regime 依存なだけでなく、regime によっては差分自体が消えるということである。したがって、4-window mean を単独で読むのではなく、どの class が何本あるかを先に確認する。

したがって、現時点の実務的な読みは次のとおりである。

1. `current_bankroll_candidate` は drawdown の強い局面では最も defensive だが、常に優位ではない
2. `current_ev_candidate` は依然として中間候補だが、August のような好調 window では baseline を下回る
3. `current_recommended_serving` は単なる baseline ではなく、特定 window では net と bankroll の両面で最良になる
4. May のように candidate 側が baseline と同一挙動へ収束する window もあるため、「候補は常に防御的に振れる」とは限らない
5. したがって、短い serving compare の結論をそのまま一般化せず、複数 regime の window を跨いで読む必要がある

### 9.5 August weekend の反証 window

2026-03-22 に、次の 6 日で `current_recommended_serving` を `replay-existing` で再比較した。

- `2024-08-03`
- `2024-08-04`
- `2024-08-10`
- `2024-08-11`
- `2024-08-17`
- `2024-08-18`

結果は late-September と逆で、`current_recommended_serving` が両候補を上回った。

- vs `current_bankroll_candidate`: total policy net `20.1` vs `-2.0`、pure final bankroll `1.1765` vs `0.9317`
- vs `current_ev_candidate`: total policy net `20.1` vs `-9.0`、pure final bankroll `1.1765` vs `0.7482`
- 両 compare とも best sweep result は `selected_label=current_recommended_serving` だった

この window は、de-risk 候補が常に有利とは限らないことを示す反証になっている。Aug のように baseline 側が十分に利益を出せている局面では、ベット抑制そのものが機会損失になる。

つまり、比較の軸は単純な net だけではなく、次の 3 つで見る。

1. bets
2. total net
3. final bankroll

### 9.6 May weekend の zero-delta window

2026-03-22 に、次の 6 日で `current_recommended_serving` を `replay-existing` で再比較した。

- `2024-05-04`
- `2024-05-05`
- `2024-05-11`
- `2024-05-12`
- `2024-05-18`
- `2024-05-19`

結果は August や late-September とも異なり、`current_bankroll_candidate` / `current_ev_candidate` の両方が `current_recommended_serving` と完全一致した。

- 両 compare とも `differing_score_source_dates=[]`
- 両 compare とも `differing_policy_dates=[]`
- total policy bets はすべて `6`
- total policy net はすべて `-6.0`
- pure final bankroll はすべて `0.9906`
- best sweep result も baseline と同じ path を選び、`tradeoff_classification` は `zero_net_zero_bankroll` になった

この window は、「候補が defensive に効く window」と「baseline が優位な window」に加えて、「candidate を切り替えても実質差が出ない window」が存在することを示している。つまり、stable alias が違っていても、実 calendar 上では同じ policy に収束する regime がある。

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