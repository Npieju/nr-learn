# Tighter Policy Candidate Matrix

## 1. Purpose

この文書は、`tighter policy` family の次候補を少数に絞るための matrix である。

## 2. Starting Point

基準の 2 本:

- `r20260326_tighter_policy_ratio003`
- `r20260327_tighter_policy_ratio003_abs80`

narrow threshold sweep の結果、`min_bet_ratio=0.03` では `min_bets_abs=90` と `80` の両方が `5/5` feasible folds を満たした。したがって、次の anchor は `80` ではなく、より strict な `90` を採用する。

## 3. Candidate Rows

| candidate | base config | min_bet_ratio | min_bets_abs | serving min_prob | serving odds_max | serving min_ev | intent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | `ratio003_abs90` | `0.03` | `90` | `0.06` | `18` | `1.0` | strictest `5/5` support anchor |
| B | `ratio003_abs90_minprob005` | `0.03` | `90` | `0.05` | `18` | `1.0` | slightly wider selection around the strict anchor |
| C | `ratio003_abs90_odds25` | `0.03` | `90` | `0.06` | `25` | `1.0` | keep anchor support, widen odds band |

## 4. Pruning Rule

最初の round では 3 本以上増やさない。

次のいずれかに当てはまる candidate は落とす。

- role が A とほぼ同じ
- drawdown を悪化させるだけ
- support が増えない
- compare で broad replacement と誤読しやすい

## 5. 2026-03-29 Result Update

Candidate A は formal run 完了後、単なる strict anchor 候補ではなく promoted anchor と読む。

確定した結果:

- revision: `r20260329_tighter_policy_ratio003_abs90`
- revision gate: `pass / promote`
- promotion gate: `pass / promote`
- formal benchmark weighted ROI: `1.1042287961989103`
- feasible folds: `5/5`
- bets total: `598`

したがって、以後の読みは次で固定する。

- A:
  - promoted challenger-ready anchor
- B:
  - A に対する `min_prob` widening challenger
- C:
  - A に対する `odds_max` widening challenger

以後 B/C は「pre-A baseline を超えるか」ではなく、「promoted A を上回るか、あるいは明確に補完するか」で評価する。

### 5.1 Candidate B Result

- revision:
  - `r20260329_tighter_policy_ratio003_abs90_minprob005`
- evaluation-summary read:
  - promoted A と decisive fields が実質同値
- operational note:
  - run 中に portfolio path の `rank=NaN` bug を露出させ、これは別 issue で修正済み
- decision:
  - promoted anchor 置換候補にはしない
  - no-op / equivalence lesson として扱う

### 5.2 Candidate C Result

- revision:
  - `r20260329_tighter_policy_ratio003_abs90_odds25`
- revision gate:
  - `pass / promote`
- promotion gate:
  - `pass / promote`
- equivalence guard:
  - `different`
- evaluation-layer read:
  - `wf_nested_test_roi_weighted`
    - `0.9153552319682917 -> 0.916134557600003`
  - `wf_nested_test_roi_mean`
    - `0.8923979938465045 -> 0.8933616912671729`
- formal benchmark read:
  - weighted ROI:
    - A `1.1042287961989103`
    - C `1.1038859989058847`
  - feasible folds:
    - both `5/5`
  - bets total:
    - both `598`
- decision:
  - near-par challenger reference として残す
  - operational anchor は A を維持する
