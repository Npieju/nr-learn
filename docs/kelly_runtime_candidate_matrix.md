# Kelly Runtime Candidate Matrix

## 1. Purpose

この文書は、`kelly-centered runtime family` を次の本線 issue として進めるための最小 candidate matrix である。

`tighter policy search` family の first-wave は完了し、formal benchmark の fold winners も実質すべて `kelly` だった。したがって次は、portfolio を主軸にした serving から一歩進めて、`kelly` 自体を baseline defensive fallback family として formal に読む。

## 2. Anchor

比較対象の operational anchor は次で固定する。

- promoted anchor:
  - `r20260329_tighter_policy_ratio003_abs90`
- reference promotion:
  - `artifacts/reports/promotion_gate_r20260329_tighter_policy_ratio003_abs90.json`

## 3. Why Kelly Next

既存 formal benchmark の fold-level winners は次の通り、繰り返し `kelly` に寄っている。

- `promotion_gate_r20260326_tighter_policy_ratio003.json`
  - `kelly, kelly, kelly, kelly`
- `promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
  - `kelly, kelly, kelly, kelly, kelly`
- `promotion_gate_r20260329_tighter_policy_ratio003_abs90.json`
  - `kelly, kelly, kelly, kelly, kelly`
- `promotion_gate_r20260329_tighter_policy_ratio003_abs90_odds25.json`
  - `kelly, kelly, kelly, kelly, kelly`

したがって、次に formal 化する価値が高いのは「portfolio 主体 family の widening」ではなく、「runtime kelly family の simplest baseline candidates」である。

## 4. Candidate Rows

| candidate | base config | serving strategy | min_prob | odds_max | min_edge | intent |
| --- | --- | --- | ---: | ---: | ---: | --- |
| K1 | `kelly_runtime_base25` | pure kelly | `0.05` | `25` | `0.0` | most interpretable pure Kelly baseline |
| K2 | `kelly_runtime_minprob003` | pure kelly | `0.03` | `25` | `0.0` | wider defensive Kelly candidate |
| K3 | `kelly_runtime_edge005` | pure kelly | `0.03` | `25` | `0.005` | slightly stricter Kelly filter |

全候補共通:

- `blend_weight=0.8`
- `odds_min=1.0`
- `fractional_kelly=0.25`
- `max_fraction=0.02`
- evaluation policy constraints は promoted A と同じ `min_bet_ratio=0.03 / min_bets_abs=90`

## 5. Reading Rule

最初の round では、次の問いだけに絞る。

1. pure Kelly family は promoted A と同等以上の formal benchmark ROI を維持できるか
2. feasible folds `5/5` を維持できるか
3. drawdown / bankroll の読みが A より単純で説明しやすいか

## 6. Decision Rule

- `promote over anchor`
  - A より weighted ROI が改善
  - feasible folds を維持
  - guardrail を壊さない
- `keep as candidate`
  - A は超えないが defensive runtime role がより明確
- `reject`
  - A より primary KPI が弱く、role の追加価値も薄い

## 7. Immediate Next Step

この 3 本だけを `run_revision_gate.py --dry-run` に載せ、最初の formal candidate を 1 本に絞る。
