# Next Issue: NAR Class-Rest-Surface Replay

Current queue note:

- この draft は未完了の future option だが、current active NAR issue ではない。
- issue ladder 上では Stage 2 feature-family parity に属し、着手順は `#101` strict `pre_race_only` benchmark rebuild 完了後、`#103` value-blend architecture bootstrap の後段である。
- したがって現時点では未起票 draft として保持し、current queue には昇格しない。

## Summary

`#57` で local Nankan baseline の denominator-first formal read は確定した。

current best line は `configs/model_local_baseline_wf_runtime_narrow.yaml` を使う separate-universe baseline で、formal read は `bets_total=3525`, `test_races_total=28997`, `bet_rate=12.16%`, `weighted_roi=3.6903`, `wf_feasible_fold_count=3` だった。

したがって次の line は broad widening ではなく、JRA Tier A family を NAR 向けに narrow replay することになる。baseline には jockey/trainer history はすでに入っているため、first replay family は class/rest/surface の core 追加に絞る。

## Objective

local Nankan baseline に class/rest/surface の core 6 features を narrow add-on し、support を壊さず ROI / AUC を維持または改善できるかを formal に確認する。

## Hypothesis

if NAR baseline に `horse_days_since_last_race`, `horse_weight_change`, `horse_distance_change`, `horse_surface_switch`, `race_class_score`, `horse_class_change` を追加する, then JRA 由来の Tier A family を NAR separate-universe に narrow replay しつつ、support を保ったまま predictive strength を改善できる。

## Current Read

- NAR baseline formal read は `pass / promote` まで到達した
- denominator-first read は `3525 / 28997 races = 12.16%`
- current baseline feature set は 13 features で、class/rest/surface family は未投入
- `docs/nar_model_transfer_strategy.md` では JRA Tier A family の narrow replay を next wave としている
- baseline には `jockey_last_30_win_rate`, `trainer_last_30_win_rate` がすでにあるため、next add-on は class/rest/surface のほうが分離しやすい

## In Scope

- `configs/features_local_baseline_class_rest_surface_replay.yaml`
- `configs/model_local_baseline_wf_runtime_narrow.yaml`
- `configs/data_local_nankan.yaml`
- `scripts/run_local_revision_gate.py`
- `scripts/run_revision_gate.py`
- local Nankan formal artifacts

## Non-Goals

- JRA / NAR integration
- NAR operational default の再定義
- jockey/trainer-combo family の widening
- lineage / pedigree family の追加

## Success Metrics

- denominator-first formal read を維持したまま candidate の `pass / hold / reject` を判定できる
- `wf_feasible_fold_count` を baseline 以上で維持する
- `bet_rate` を大きく毀損せず、`AUC` または `weighted ROI` の改善余地を確認できる
- next NAR child issue を 1 measurable hypothesis に narrow できる

## Validation Plan

- dry-run で full local revision lane を確認する
- true train/evaluate/wf/promotion を matching tuple で回す
- formal read は `docs/nar_formal_read_template.md` に従って `bets / races / bet-rate` を最優先で読む

## Stop Condition

- support が baseline より明確に悪化する
- gain が低 bet-rate compression にしか見えない
- feature add-on が narrow replay ではなく broad family widening になってしまう
