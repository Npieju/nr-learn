# Next Issue: Difficult-Regime Operator Routing Surface-vs-Pace

## Summary

`#93` の analysis-first promoted candidate ordering により、non-default promoted line の current operator read は次で固定された。

- first operator reference:
  - `r20260330_surface_plus_class_layoff_interactions_v1`
- secondary difficult-regime specialist:
  - `r20260403_pace_closing_fit_selective_v1`

ただし、ここで固定されたのは ordering であり、routing rule ではない。

- `surface_plus_class_layoff` は September downside control と December carry を同時に残すが、low-exposure で conservative
- `pace_closing_fit` は September de-risk specialist としては clean だが、December carry は再現しない

したがって次の narrow JRA hypothesis は、この 2 line を broad replacement 候補として再昇格させることではなく、September difficult regime の中で explicit routing 境界を定義できるかを existing artifact から判定することである。

## Objective

`surface_plus_class_layoff` と `pace_closing_fit` を difficult-regime operator reference として横並びに再読し、September difficult window の中で explicit routing rule を定義できるか、それとも current ordering のまま compare reference に留めるべきかを判定する。

## Hypothesis

if `pace_closing_fit` の優位が September difficult window の narrower subset に局在し、`surface_plus_class_layoff` は broader difficult window でより安定している, then the two lines can be separated into `primary difficult-regime reference` and `narrower specialist` with an explicit routing boundary rather than a single flat ordering.

## Current Read

- `docs/issue_library/next_issue_analysis_first_promoted_candidate_ordering.md` では
  - first operator reference は `r20260330_surface_plus_class_layoff_interactions_v1`
  - secondary difficult-regime specialist は `r20260403_pace_closing_fit_selective_v1`
  に固定済みである
- `surface_plus_class_layoff` は
  - September `8 bets / -8.0 / pure bankroll 0.7333`
  - December `13 bets / +20.0 / pure bankroll 1.6551`
  で、4 line の中で唯一 September downside control と December carry を同時に残した
- `pace_closing_fit` は
  - September `3 bets / -3.0 / pure bankroll 0.892891589506173`
  - December `0 bets / 0.0 / pure bankroll 1.0`
  で、September specialist としては clean だが control window では carry を作らない
- current docs は ordering までは固定しているが、operator が difficult window 内でどちらを先に参照すべきかの routing wording は未固定である

## In Scope

- `r20260330_surface_plus_class_layoff_interactions_v1`
- `r20260403_pace_closing_fit_selective_v1`
- current baseline `current_recommended_serving_2025_latest`
- existing September difficult window compare artifacts
- 必要なら late-September narrow reread
- routing wording:
  - `broad difficult-regime reference`
  - `narrow specialist`
  - `no explicit routing; keep flat ordering`

## Non-Goals

- new retrain / model rebuild
- policy rewrite
- serving default の変更
- NAR work
- pruning judgment の再利用

## Success Metrics

- difficult-regime operator read が ordering だけでなく routing wording まで 1 本に固定される
- `bets / total net / pure bankroll` で September broad と narrower subset の差が説明できる
- next action が `explicit routing note` か `keep as flat ordering` の二択に絞れる

## Validation Plan

1. existing ordering doc と role split docs から September / December read を再掲する
2. broad September difficult window では `surface_plus_class_layoff` と `pace_closing_fit` の relative tradeoff を `bets / total net / pure bankroll` で比較する
3. 必要なら late-September narrow window の artifact を rereadし、`pace_closing_fit` の優位が narrower subset に集中しているかを見る
4. routing boundary が defendable なら wording を固定し、弱ければ current flat ordering を維持する

## Actual-Date Reread

Broad September difficult window では `pace_closing_fit` が `surface_plus_class_layoff` より defensive に見える。

- `r20260330_surface_plus_class_layoff_interactions_v1`
  - `8 bets`
  - `total net = -8.0`
  - `pure bankroll = 0.7333`
- `r20260403_pace_closing_fit_selective_v1`
  - `3 bets`
  - `total net = -3.0`
  - `pure bankroll = 0.892891589506173`

ただし、この差は `pace_closing_fit` が same-window でより sparse に suppress していることでも説明できる。どちらも low-exposure analysis-first line であり、broad September だけでは `pace_closing_fit` を先に参照すべき routing boundary までは作れない。

Narrower September evidence も routing を固定するほど強くない。

- late-September 5-day compare では `r20260403_pace_closing_fit_selective_v1` は
  - `2 bets`
  - `total net = -2.0`
  - `pure bankroll = 0.9184027777777779`
- ただし same window の `current_sep_guard_candidate_2025_latest` は
  - `6 bets`
  - `total net = -6.0`
  - `pure bankroll = 0.9857901270555528`

つまり `pace_closing_fit` は late-September でも narrow specialist らしい defensive read は保つが、existing narrow-window artifact でも pure bankroll の最良 line にはなっていない。また、`surface_plus_class_layoff` 側に symmetric late-September superiority artifact はなく、`pace_closing_fit` を `surface_plus_class_layoff` より先に自動採用する閾値は existing evidence だけでは定義できない。

December control window は引き続き `surface_plus_class_layoff` を broad operator reference に残す根拠になる。

- `r20260330_surface_plus_class_layoff_interactions_v1`
  - `13 bets`
  - `total net = +20.0`
  - `pure bankroll = 1.6551`
- `r20260403_pace_closing_fit_selective_v1`
  - `0 bets`
  - `total net = 0.0`
  - `pure bankroll = 1.0`

## Result

existing artifact reread だけでは explicit difficult-regime routing boundary は defend できなかった。

## Decision Summary

current decision は次で固定する。

- `r20260330_surface_plus_class_layoff_interactions_v1` を broad difficult-regime first reference のまま維持する
- `r20260403_pace_closing_fit_selective_v1` は narrower September specialist / second reading に留める
- explicit narrow routing rule は追加しない
- operator wording は `surface first, pace second` の flat ordering を維持する

理由は 3 つある。

1. broad September で `pace_closing_fit` はより defensive だが、改善は `3 bets` まで suppress した結果でも説明でき、routing boundary としては弱い
2. late-September narrow evidence でも `pace_closing_fit` は `sep_guard` を pure bankroll で更新しておらず、stable narrow override の形になっていない
3. cross-window では `surface_plus_class_layoff` だけが September downside control と December positive carry を同時に残している

したがって、この 2 line の practical read は ordering のままで十分である。September difficult window の deeper reread が必要なときは `pace_closing_fit` を second reading として参照してよいが、`surface_plus_class_layoff` から `pace_closing_fit` へ自動で切り替える explicit routing wording までは導入しない。

## Expected Artifacts

- operator routing reread memo
- optional late-September compare note
- queue / issue-thread に貼れる decision summary

## Stop Condition

- narrower subset を切っても `pace_closing_fit` の優位が routing rule として安定しない
- improvement が exposure suppression にしか見えない
- `surface_plus_class_layoff` first / `pace_closing_fit` second の flat ordering を崩す根拠が作れない

## Primary References

- `docs/issue_library/next_issue_analysis_first_promoted_candidate_ordering.md`
- `docs/issue_library/next_issue_post_surface_plus_class_layoff_promotion.md`
- `docs/issue_library/next_issue_surface_plus_class_layoff_bet_rate_robustness.md`
- `docs/issue_library/next_issue_pace_closing_fit_actual_date_role_split.md`
- `artifacts/reports/promotion_gate_r20260330_surface_plus_class_layoff_interactions_v1.json`
- `artifacts/reports/promotion_gate_r20260403_pace_closing_fit_selective_v1.json`

## Starting Expectation

default expectation は `surface_plus_class_layoff` を broad difficult-regime reference、`pace_closing_fit` を narrower September specialist として retain し、explicit routing を作れない場合は current flat ordering を維持することである。