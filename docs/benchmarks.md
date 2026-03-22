# ベンチマークと採用基準

## 1. この文書の役割

この文書は、`nr-learn` で何を良い結果とみなすかを定義するための正本である。

日常運用の区切り方や `revision` の切り方は、[development_flow.md](development_flow.md) を参照する。
formal な evaluation 導線は、[evaluation_guide.md](evaluation_guide.md) を参照する。

ここで決めるのは次の 3 点である。

1. 何の指標で評価するか。
2. どの数値を目標とするか。
3. どの条件を満たしたら採用するか。

## 2. 基本姿勢

- 単発の高 ROI は、そのまま採用根拠にしない。
- 同一期間で最適化と評価を兼ねない。
- raw 指標だけでなく、実際に買った場合の件数と純収益まで確認する。
- より単純な構成で同じ observed behavior を再現できるなら、単純な構成を優先する。

## 3. 主要指標

### 3.1 Weighted ROI

時系列検証の各期間で得た ROI を、件数を加味してまとめた指標である。

- `1.0` を超えると、検証全体で回収率が 100% を超えたことを意味する。
- ただし bets が極端に少ない場合は採用しない。

### 3.2 Bets

実際に賭けた件数である。

- 少なすぎる ROI は再現性が低い。
- `Weighted ROI` と必ずセットで解釈する。

### 3.3 Total Net

回収総額から投資総額を引いた値である。

- mean ROI だけでは、件数差による見かけの改善を見落とすことがある。
- serving 比較では `total net` を重要指標として扱う。

### 3.4 AUC / pseudo-R²

予測の並び順品質と、市場オッズに対する追加情報の有無を見る。

- AUC は予測順位の良さを見る。
- pseudo-R² は市場に対する独自情報の強さを見る。

## 4. 外部比較の目安

Benter 系の比較では、重要なのは単純な ROI よりも `ΔR² = R²_combined - R²_public` である。

当面の目安は次のとおりとする。

- 最低到達ライン:
  - `ΔR² > 0.009`
- 強いライン:
  - `ΔR² ≈ 0.018`

また、短期サンプルで `ROI > 1.0` が出ても、それだけで長期運用の目標達成とはみなさない。

## 5. 現在の内部 benchmark ladder

2024 年データの nested walk-forward における主要候補は次のとおりである。

| 候補 | Weighted ROI | Bets | 位置づけ |
| --- | ---: | ---: | --- |
| no-lineage public-free baseline | `0.5788` | `603` | 市場情報を切った基準線 |
| liquidity high-coverage | `0.9346` | `700` | 流動性改善の転換点 |
| regime hybrid | `0.9915` | `731` | 単純な運用候補 |
| regime modelswitch f1 policy may | `1.0073` | `713` | 現在の主力候補 |

この数値だけを見ると最後の候補が最良だが、構成の複雑さも同時に増しているため、運用判断では simpler rollback 候補との比較を必ず行う。

### 5.1 直近の revision comparison

2026-03-22 時点の full revision gate では、次の 2 候補を同条件で通した。

| revision | profile | status | decision | 備考 |
| --- | --- | --- | --- | --- |
| `r20260322a` | `current_best_eval` | `pass` | `promote` | `score_source_count=2`。`may_runtime_liquidity` override を読む |
| `r20260322b` | `current_recommended_serving` | `pass` | `promote` | `score_source_count=1`。単一 source で通過 |

この 2 run では、`evaluate --max-rows 120000 --wf-mode fast --wf-scheme nested` のトップライン指標は実質同一だった。

- `top1_roi`: `0.796298...`
- `ev_top1_roi`: `0.660712...`
- `auc`: `0.834390...`

したがって、現時点の運用判断では `current_recommended_serving` を simpler candidate として優先的に扱ってよい。
ただし、両者とも promotion gate warning として `probe_only` な walk-forward slice が残るため、将来の benchmark 更新では引き続き stricter full evaluation を優先する。

### 5.2 de-risk candidate の formal gate

2026-03-22 に、de-risk 候補 2 本も formal な revision gate に載せた。

| revision | profile | status | decision | 備考 |
| --- | --- | --- | --- | --- |
| `r20260322c` | `current_bankroll_candidate` | `block` | `hold` | representative evaluate は通るが、matching WF で feasible fold `0/5` |
| `r20260322d` | `current_ev_candidate` | `block` | `hold` | representative evaluate は通るが、matching WF で feasible fold `0/5` |

この 2 候補では共通して次が確認できた。

- `evaluate` 側の `stability_assessment` は `representative`
- matching な `wf_feasibility_diag` を付けると promotion gate は `wf_feasible_fold_count=0` で block
- dominant failure reason は `min_bets`
- fold ごとの binding source は全て `min_bets_abs=100` で、ratio 側ではなかった
- `max_infeasible_bets_observed=58` に留まり、threshold 未達が明確だった

同日付の threshold sweep で、support gap も数値化した。

- 両候補とも `min_bets_abs=58` で初めて feasible fold が `1/5` に到達した
- 両候補とも `min_bets_abs=40` で feasible fold が `4/5` になり、`3/5` 条件はここで満たす
- 両候補とも `min_bets_abs=30` まで下げると feasible fold が `5/5` になった
- fold 別の最初の到達 threshold は共通で、fold 3 が `58`、fold 1 が `45`、fold 2 と fold 4 が `40`、fold 5 が `30` だった

さらに threshold compare から mitigation probe を組み立てると、runtime 候補としては次の 2 本に収束した。

- dominant blocked signature は `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95` で、84 blocked occurrence を占めた
- そのうち `74/84` occurrence では `portfolio / blend_weight=0.6 / min_prob=0.03 / top_k=1 / min_ev=1.0` がより高い final bankroll を示した
- 残り `10/84` occurrence では `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=1.0` が pure bankroll で上回った
- mitigation probe からは runtime-ready candidate として `portfolio_lower_blend` と `portfolio_ev_only` の 2 本を書き出せたが、`74/10` の staged hybrid は現行 runtime の単一 policy/date override では直接表現できなかった

つまり、この 2 候補の formal block は「bankroll 側か EV 側か」の違いではなく、同じ support frontier に乗っている問題として読むのが正しい。

したがって、`current_bankroll_candidate` と `current_ev_candidate` は serving 上の de-risk 候補としては有力でも、現時点では benchmark 更新や正式昇格の候補とは扱わない。運用上は rollback / defensive override の候補に留め、昇格判断は support を増やす別の改善が入ってから再評価する。

## 6. serving 比較の判断基準

actual calendar の比較では、次を確認する。

1. `policy_bets`
2. `policy_roi`
3. `total return`
4. `total net`

現在の判断原則は次のとおりである。

- 追加 return があっても `total net` が悪化するなら採用しない。
- 主力候補と observed behavior が一致する単純候補があるなら、rollback ではそちらを優先する。
- representative date だけでなく、複数日 window で再確認する。

### 6.1 直近の複数 window aggregate

2026-03-22 時点で、`current_recommended_serving` を基準に次の 2 系統を複数 window aggregate で比較した。

- `current_bankroll_candidate`
- `current_ev_candidate`

入力 window は次の 4 本である。

- `tail_weekends`
- late-September 5 日 window
- `aug_weekends_20260322`
- `may_weekends_20260322`

aggregate の要点は次のとおりである。

- `current_bankroll_candidate` は pure bankroll delta が `2/4` windows で正、`1/4` は zero、4-window mean は `net -2.6500`, `bankroll +0.1520`
- `current_ev_candidate` も pure bankroll delta が `2/4` windows で正、`1/4` は zero、4-window mean は `net -5.6500`, `bankroll +0.0354`
- 両候補とも net delta が正だったのは late-September 側だけだった
- `tail_weekends` では `current_recommended_serving` が total net で優位で、候補側は pure final bankroll のみ改善した
- `aug_weekends_20260322` では `current_recommended_serving` が net と bankroll の両面で両候補を上回った
- `may_weekends_20260322` では両候補とも `current_recommended_serving` と完全一致した

aggregate JSON の `tradeoff_classification` で言い換えると、window 構成は `++`, `-+`, `--`, `00` が 1 本ずつである。

- late-September は `positive_net_positive_bankroll`
- `tail_weekends` は `negative_net_positive_bankroll`
- `aug_weekends_20260322` は `negative_net_negative_bankroll`
- `may_weekends_20260322` は `zero_net_zero_bankroll`

このため、候補側を「bankroll 改善候補」とだけ要約するのは粗すぎる。現状の evidence は、「一部 regime では効き、別 regime では baseline より悪化し、さらに差分が消える regime もある」である。

したがって、現時点の運用上の位置づけは次のとおりである。

1. `current_bankroll_candidate` は rollback / de-risk の有力候補だが、全 regime で優位とは言えない
2. `current_ev_candidate` は中間候補だが、現時点では `current_recommended_serving` を安定して上回る根拠は弱い
3. `current_recommended_serving` は baseline として維持し、候補置換は regime ごとの裏付けが増えるまで慎重に扱う

### 6.2 threshold 由来 single-policy probe の位置づけ

2026-03-22 に mitigation probe から生成した `portfolio_lower_blend` / `portfolio_ev_only` の single-policy config を actual-date smoke で再確認した。

- late-September 5 日 window では baseline `current_recommended_serving` が `31 bets / total net -25.6`、`portfolio_lower_blend` は `1 bet / -1.0`、`portfolio_ev_only` は `12 bets / -12.0`
- August weekends 6 日 window では baseline が `34 bets / total net +20.1`、`portfolio_lower_blend` は `2 bets / -2.0`、`portfolio_ev_only` は `9 bets / -9.0`
- late-September では defensive loss reduction は見えたが、August 側では baseline の利益局面を大きく取り逃した

このため、generated single-policy probe は「formal gate blockage の原因 signature を actual calendar でどう抑えるか」を観察する補助 artifact にはなるが、現時点では stable profile 登録候補ではない。運用判断は引き続き `current_recommended_serving` を baseline に置き、generated probe は selector 拡張や staged hybrid 実装の前段 evidence として扱う。

### 6.3 staged mitigation probe の位置づけ

既存の staged probe config も same-window actual-date smoke で確認したが、運用判断は変わらなかった。

- late-September では `staged_mitigation_probe` が `12 bets / total net -12.0`、`staged_mitigation_ev_guard_probe` が `10 bets / -10.0`
- August weekends では `staged_mitigation_probe` が `9 bets / -9.0`、`staged_mitigation_ev_guard_probe` が `6 bets / -6.0`
- baseline `current_recommended_serving` は同じ window で `late-September 31 bets / -25.6`、`August weekends 34 bets / +20.1`

つまり、staged probe は runtime feasibility 自体は示したが、realized calendar behavior は single-policy probe の defensiveness を少し調整した範囲に留まり、baseline replacement の根拠にはならなかった。現時点では staged も stable profile 候補ではなく、fallback 条件の再設計や selector logic の evidence 化が先に必要である。

補足すると、observed stage path もこの判断を裏づけている。

- `staged_mitigation_probe` は observed windows で実質 `portfolio_ev_only` のままで、stage diversification を生まなかった
- `staged_mitigation_ev_guard_probe` だけが August の一部日付で `kelly_fallback_1` / `kelly_fallback_2` を発火させた
- それでも August aggregate は baseline に届かなかったため、「fallback が発火すること」と「運用候補として十分に強いこと」は別問題だと分かった

加えて、staged runtime artifact には fallback reason も残すようにした。

- 例として `2024-08-11` の `staged_mitigation_ev_guard_probe` では、`portfolio_ev_only:max_expected_value_below|portfolio_lower_blend:no_selection` を経て `kelly_fallback_1` に到達したことが summary / backtest JSON の両方から確認できる
- これにより、今後の論点は単に「Kelly fallback が出たか」ではなく、「どの trigger が profitable regime を潰しているか」を actual-date artifact から直接監査することになる

同日付の `run_serving_stage_path_compare.py` でもこの差分を横比較した。

- late-September 5 日 window では `differing_stage_dates` が全 5 日だった
- August weekends 6 日 window では `differing_stage_dates` が `2024-08-03`, `2024-08-10`, `2024-08-11`, `2024-08-18` だった
- fallback reason まで入れた再比較では、late-September は `differing_stage_fallback_reason_dates` も全 5 日、August weekends は `2024-08-03`, `2024-08-11`, `2024-08-18` だった
- late-September の `staged_mitigation_ev_guard_probe` は 5/5 日すべてで `portfolio_ev_only:max_expected_value_below` を起点に Kelly fallback へ流れており、しかも 4/5 日は `portfolio_lower_blend:no_selection` で止まっていた
- August の fallback reason は 3 日だけで、`2024-08-03` のみ `portfolio_lower_blend:max_expected_value_below` まで進んだうえで `kelly_fallback_2` に落ちていた

したがって、現状の論点は「stage path が分岐しているか」ではなく、「その分岐条件が profitable regime を十分に拾えているか」である。

### 6.4 staged EV guard threshold sweep の判断

2026-03-22 に `scripts/run_staged_ev_guard_threshold_sweep.py` を追加し、`staged_mitigation_ev_guard_probe` の先頭 2 stage にある `max_expected_value_below` を actual-date replay で sweep した。

- stage1 grid: `1.08`, `1.06`, `1.04`, `1.02`
- stage2 grid: `1.03`, `1.01`, `0.99`
- late-September 5 日 window と August weekends 6 日 window の 2 本で実行

結果は、threshold 感度の解釈をかなり絞った。

- late-September では全 12 variant が `10 bets / total net -10.0` で横並びだった
- ただし fallback 頻度は `5/5 -> 4/5 -> 3/5` と段階的に減り、lower-blend selection も 1 日だけ復活した
- August では `s1 in {1.08,1.06,1.04}` がすべて `6 bets / total net -6.0` で横並びだった
- August の `s1=1.02` は全 stage2 variant で `8 bets / total net -8.0` に悪化した
- stage2 を `1.03 -> 1.01/0.99` に下げると `2024-08-03` の `kelly_fallback_2` は消えたが、window 全体の net は改善しなかった

この evidence から、今の staged bottleneck は次のように読むのが妥当である。

1. EV guard は確かに aggressive で、不要な fallback を起こしている日がある
2. ただし、今回の実 calendar では moderate な threshold 緩和だけでは baseline 置換の根拠になる profit 改善は出ない
3. さらに、stage1 を `1.02` まで下げるのは August regime では明確に悪化なので unsafe 側である

したがって、次の selector 改善は単純な threshold lowering で押し切るべきではない。候補は「August profitable regime を見分ける追加条件」か、「bankroll state / date-context を含む richer fallback 条件」である。

### 6.5 staged signal diagnostic の判断

同日、`scripts/run_staged_policy_signal_diagnostic.py` も追加し、August weekends の fallback dates を race-level signal で再確認した。

この診断は threshold sweep より重要な反証を与えた。

- `2024-08-03` と `2024-08-11` では、`portfolio_ev_only` は実際に race を選んでいるが、その選択自体が 0 hit だった
- `2024-08-18` では `portfolio_ev_only` が 4 race を選び、median `max_expected_value` は約 `1.023` だったが、やはり 0 hit だった
- `2024-08-17` はさらに厳しく、staged probe は全 stage `no_selection` だった一方、baseline `current_recommended_serving` は `7 bets / net +32.6` を出していた
- baseline は `2024-08-18` でも `8 bets / net +2.7` を出しており、staged の sparse portfolio family が profitable day を取り逃していることが分かった

この evidence により、現在の staged probe を「August は EV guard が少し厳しすぎるだけ」と解釈するのは不正確になった。より正確には、August gap は次の 2 層に分かれる。

1. 一部の race では EV guard が unnecessary fallback を起こしている
2. それとは別に、early portfolio stages 自体が profitable day を十分に再現できていない

したがって、次の selector 改善で優先すべきなのは `max_expected_value_below` の微調整ではなく、`portfolio_ev_only` / `portfolio_lower_blend` の policy family 自体を見直すこと、または baseline 側 profitable regime を stage policy に持ち込める regime-aware split を導入することである。

### 6.6 portfolio family sweep の判断

この仮説は `scripts/run_portfolio_policy_family_sweep.py` でさらに絞れた。August weekends の単純 portfolio replay sweepでは、baseline August policy family がそのまま最適だった。

- best variant は `blend=0.8 / min_prob=0.03 / min_ev=0.95` で、`34 bets / total net +20.1`
- 同じ `blend=0.8 / min_prob=0.03` でも `min_ev=1.0` に上げるだけで `9 bets / total net -9.0`
- `blend=0.8 / min_prob=0.05 / min_ev=0.95` は still positive だが `+8.6` まで縮む
- `blend=0.6` variants はすべて negative で、best でも `-1.0`

この差は profitable dates で特に大きい。

- `2024-08-17` は baseline family だと `7 bets / net +32.6`、`min_ev=1.0` に上げると `0 bets / net 0.0`
- `2024-08-18` は baseline family だと `8 bets / net +2.7`、`min_ev=1.0` だと `4 bets / net -4.0`

したがって、August で staged probe が弱い理由はもうかなり具体的である。

1. stage1 の `min_expected_value=1.0` が profitable August races を落としすぎている
2. stage2 の `blend=0.6` も recovery ではなく、むしろ profitability を削る方向に働いている
3. つまり、次の実験は threshold 調整ではなく「stage1 を baseline August portfolio family にどこまで寄せるか」で始めるべきである

### 6.7 baseline-aligned staged probe の判断

この次段として、stage1 を baseline August family へ戻した `runtime_staged_aug_baseline_stage1_probe` を actual-date replay で確認した。

- stage1 は `blend=0.8 / min_prob=0.03 / min_ev=0.95`
- stage2 以下は既存 staged probe と同じ lower-blend / Kelly fallback を残した
- fallback 条件は置かず、no-selection のときだけ次段へ進む構成にした

結果は解釈しやすい。

- August weekends 6 日 window では baseline `current_recommended_serving` と完全一致し、`34 bets / total net +20.1`
- ただし 6/6 日すべてが `portfolio_aug_baseline:selected` で、stage2 以下は一度も使われなかった
- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して、この probe は `38 bets / total net -32.6` とさらに悪化した
- late-September でも 5/5 日すべてが `portfolio_aug_baseline:selected` で、de-risk fallback は一切発火しなかった

したがって、この probe は「August gap の主因が stage1 family の strictness にある」ことは裏づけたが、運用候補にはならない。なぜなら、August を救う方法がそのまま late-September の over-exposure につながり、しかも staged wrapper 自体は何も制御していないからである。

ここからの正しい次段は、単なる staged shell の維持ではなく、次のどちらかである。

1. August 相当 regime を判別できる date-context / regime-aware split を selector に持ち込む
2. baseline-aligned stage1 の上に、late-September だけを抑える explicit fallback guard を追加する

### 6.8 selected_rows guard の判断

上の 2 に対する最初の小さい probe として、baseline-aligned stage1 に `fallback_when.selected_rows_at_most: 5` を付けた `runtime_staged_aug_baseline_stage1_selected_rows_guard_probe` も replay した。

結果は trade-off としては明快だが、採用候補にはならない。

- late-September 5 日 window では `10 bets / total net -10.0` まで縮み、baseline `31 bets / -25.6` よりかなり defensive になった
- 5/5 日すべてで `portfolio_aug_baseline:selected_rows_at_most` が発火し、最終段は `portfolio_lower_blend` か `kelly_fallback_1` だった
- しかし August weekends 6 日 window は `6 bets / total net -6.0` まで崩れ、baseline `34 bets / +20.1` を完全に失った
- 特に profitable day の `2024-08-17` は `7 bets / +32.6` から `0 bets / 0.0` へ落ちた

したがって、selected-count ベースの guard は late-September exposure を抑える効果自体はあるが、August 側では profitable regime を blunt に切り落としてしまう。運用候補として次に試すべきなのは、件数だけを見る guard ではなく、date-context か別の race-level signal を含む regime-aware 条件である。

### 6.9 September-only regime split candidate の判断

この次段として、September だけ `selected_rows_at_most: 5` 付き staged policy を使う `..._serving_sep_selected_rows_guard_candidate.yaml` を replay した。month-aware override を使って blunt guard を September に閉じ込める形である。

3 window で見ると挙動はかなり素直だった。

- May weekends 6 日 window は baseline と完全一致し、`6 bets / total net -6.0`
- August weekends 6 日 window も baseline と完全一致し、`34 bets / total net +20.1`
- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して `10 bets / total net -10.0`
- June-August の `tail_weekends` 19 日 window でも baseline と完全一致し、`55 bets / total net -0.9`、`differing_policy_dates=[]`

つまり、この candidate は今までの probe と違って、September の de-risk だけをほぼ純粋に取り出せている。`selected_rows_at_most` 自体は blunt でも、date-context で適用範囲を縛れば operationally useful な serving candidate になりうる。

stateful bankroll sweep を同じ 4 window で重ねても、この読みはさらに強まった。

- late-September では total net は baseline `-25.6` に対して candidate `-10.0` まで改善したが、pure path の final bankroll は baseline `0.2780`、candidate `0.0` で、best path は baseline 側に残った
- May weekends は pure path / threshold grid ともに baseline / candidate が `0.9906` で完全一致した
- August weekends も pure path / threshold grid ともに baseline / candidate が `1.1765` で完全一致した
- `tail_weekends` も pure path / threshold grid ともに baseline / candidate が `0.7432` で完全一致した

この段階での読みは次のとおりである。

1. regime-aware split は、単独 guard や staged shell より有望である
2. current evidence の範囲では、May/August/tail を壊さず late-September の net だけは改善できている
3. ただし late-September の pure bankroll は悪化するため、この candidate は `positive_net_negative_bankroll` な trade-off 候補として扱うのが正しい
4. 次に進めるなら、この candidate を aggregate / dashboard / wider-window compare の中で再確認するのが自然である

### 6.10 September Kelly-only fallback candidate の判断

上の September-only candidate を aggregate まで伸ばすと、弱点はかなりはっきりした。late-September の pure bankroll 崩壊は `2024-09-28` に `portfolio_lower_blend:selected` が 1 回入ったことに集中しており、他の 4 日は `kelly_fallback_1` でほぼフラットだった。

このため、September override から `portfolio_lower_blend` を外し、`selected_rows_at_most: 5` 発火後は Kelly fallback に直落としする `..._serving_sep_selected_rows_kelly_only_candidate.yaml` を replay した。

4 window aggregate の結果は、前の candidate より明確に強い。

- late-September 5 日 window は bets / net が `10 bets / total net -10.0` のまま維持された
- 同じ late-September で pure path final bankroll は baseline `0.2780` に対して `0.9915` まで改善した
- May weekends、August weekends、`tail_weekends` はすべて baseline と完全一致し、net / pure bankroll とも delta `0.0` だった
- aggregate の `tradeoff_classification` は `positive_net_positive_bankroll` が late-September 1 本、残り 3 window は `zero_net_zero_bankroll` だった

この candidate の読みはかなり単純である。

1. September-only split 自体は有効だったが、問題は `selected_rows` guard ではなく fallback 先の `portfolio_lower_blend` だった
2. Kelly-only fallback に置き換えると、validated 4 windows では net を保ったまま pure bankroll も改善できた
3. 現時点では、この variant が September de-risk line の中で最初の `positive_net_positive_bankroll` 候補である

加えて、旧 `..._serving_sep_selected_rows_guard_candidate.yaml` との direct late-September compare でも dominance が確認できた。

- shared policy bets は両者とも `10`
- shared total net も両者とも `-10.0`
- しかし pure path final bankroll は旧 candidate `0.0` に対して Kelly-only candidate `0.9915`

したがって、September de-risk line の current frontier はもう `selected_rows guard を使うかどうか` ではない。実務上の論点は「guard 発火後に portfolio lower-blend を残す理由があるか」であり、現時点の evidence では答えは `ない` である。

## 7. 現在の評価方針

### 7.1 nested evaluation

本命候補の判定は、`run_evaluate.py` による nested walk-forward を基準に行う。

理由は次の 2 点である。

- 同じ期間で最適化と評価を兼ねないため。
- fold ごとの no-bet や gating failure まで記録できるため。

### 7.2 serving validation

運用候補の確認は、`run_serving_smoke.py` と `run_serving_smoke_compare.py` を基準に行う。

理由は次の 2 点である。

- summary から復元した serving ルールが、実日付でも同じように動くか確認できるため。
- nested 上の改善が、actual calendar でも意味を持つかを見られるため。

ただし、短い serving validation は smoke / probe であり、正式な採用判断は `stability_assessment=representative` と promotion gate を通した revision 単位で行う。

## 8. 採用ルール

最終的な採用判断は、次の順序で行う。

1. nested walk-forward で改善しているか。
2. bets が少なすぎないか。
3. actual calendar で total net が悪化していないか。
4. 同等挙動を再現するより単純な候補がないか。

この 4 条件を満たして初めて、本命または rollback 候補として扱う。