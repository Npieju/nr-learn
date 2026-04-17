# Next Issue: local Nankan Market Provenance Fail-Closed Gate

## Summary

South-Kanto-only NAR line の reliability を担保するうえで、最優先の unresolved point は `market provenance` である。

current status:

- trust semantics repair 後の `#120` current alias は `strict_trust_ready=true` に到達している
- この issue の current role は fail-closed contract と historical evidence を保持することであり、`#101` / `#103` の current blocker を説明する primary source ではない

2026-04-06 の current local primary に strict provenance audit を当てた結果、次が確認された。

- input: `data/local_nankan/raw/local_nankan_primary.csv`
- row_count: `729107`
- race_count: `62503`
- required provenance columns missing:
  - `scheduled_post_at`
  - `card_snapshot_at`
  - `card_snapshot_relation`
  - `odds_snapshot_at`
  - `odds_snapshot_relation`
- market timing bucket:
  - `pre_race=0`
  - `unknown=729107`
  - `post_race=0`
- `unknown_ratio=1.0`
- `pre_race_rows=0`

したがって current South-Kanto line は、`odds/popularity` を含む market signal が「事前取得 odds である」とまだ証明できていない。

この状態では、data leak / odds source retrieval anomaly / stale materialization のいずれであっても benchmark を続行すべきではない。

## Objective

local Nankan market features を `provable pre-race market capture` に限定し、provenance 未証明の dataset では benchmark / revision / promotion が fail-closed で停止する状態を完成させる。

## Hypothesis

if current high ROI の主要 risk が `provable pre-race market provenance` の欠落にある, then strict provenance gate を benchmark 前に必須化し、provenance-aware local backfill/materialize を再実行することで、remaining ROI を `defensible` か `still suspicious` のどちらかに狭められる。

## In Scope

- local benchmark gate の strict provenance preflight
- local revision gate から見える provenance blocker artifact の固定
- provenance-aware local backfill / materialize の再実行条件整理
- rebuilt primary に対する strict provenance audit 再実行
- `#101` / `#103` を provenance trust gate の後段に再配置すること

## Non-Goals

- full NAR multi-region expansion
- value-blend architecture bootstrap の即時再開
- JRA/NAR mixed-universe compare
- current ROI を defense するための broad explanation 作成

## Success Metrics

- local benchmark gate が provenance 未証明 line を `not_ready` で fail-closed する
- rebuilt local primary で required provenance columns が揃う
- rebuilt local primary で `unknown_ratio=0` を満たす
- rebuilt local primary で `pre_race_rows>0` を満たす
- その後に残る ROI を leak / odds anomaly / policy optimism のどれに寄せるか再判定できる

## Validation Plan

1. current local primary に strict provenance audit を実行し blocker を artifact 化する
2. local benchmark gate へ provenance preflight を追加し、snapshot/train/evaluate 前に stop することを確認する
3. provenance-aware source から local backfill / materialize をやり直す
4. rebuilt primary に同じ strict provenance audit を再実行する
5. strict trust ready を満たした場合のみ `#101` strict `pre_race_only` benchmark handoff へ戻す

## Current Evidence

- current canonical provenance read:
  - `artifacts/reports/local_nankan_provenance_audit.json`
- provenance audit artifact:
  - `artifacts/reports/local_nankan_provenance_audit_issue120.json`
- benchmark gate block artifact:
  - `artifacts/reports/benchmark_gate_local_nankan_issue120.json`
- refreshed primary materialize artifact:
  - `artifacts/reports/local_nankan_primary_materialize_issue120_refresh.json`
- refreshed provenance audit artifact:
  - `artifacts/reports/local_nankan_provenance_audit_issue120_refresh.json`
- racecard provenance repair artifact:
  - `artifacts/reports/local_nankan_racecard_provenance_repair_issue120.json`
- repaired primary materialize artifact:
  - `artifacts/reports/local_nankan_primary_materialize_issue120_repaired.json`
- repaired provenance audit snapshot:
  - `artifacts/reports/local_nankan_provenance_audit_issue120_repaired.json`

current benchmark block read は次である。

- `status=not_ready`
- `current_phase=provenance_preflight`
- `error_code=market_provenance_not_ready`
- `recommended_action=rebuild_local_primary_with_provenance_columns`

stale primary corrective の結果、`local_nankan_primary.csv` 自体には provenance columns が再materializeで復帰した。ただし refreshed audit では次を確認した。

- provenance columns are present, but all rows are null in current result-ready primary
- external racecard provenance rows are only `562 rows / 24 races`
- provenance date window is `2026-04-06 .. 2026-04-07`
- joined result rows are `0`

その後、raw_html cache から historical racecard provenance を repair した結果、次も確認した。

- repaired racecard provenance non-null:
  - `card_snapshot_at=732503`
  - `odds_snapshot_at=732503`
- repaired strict provenance audit:
  - `pre_race=0`
  - `unknown=257`
  - `post_race=728850`

したがって current root blocker は merge bug や stale primary ではなく、`historical market cache の大半が post-race capture であり、result-ready historical rows に provable pre-race market capture が存在しない` ことである。

## Stop Condition

- rebuilt primary でも provenance columns はあるが `post_race` or `unknown` が残り、capture timing anomaly が source layer にあると判明する
- その場合は next issue を `crawler/source timing corrective` に 1 本化する

現時点の read では、次 issue を narrow するなら first candidate は `crawler/source timing corrective` である。

## Decision Rule

- `#120` current alias は strict trust ready に到達しており、trust 判定は `classification_basis=pre_race_feature_availability` を一次参照にする
- current benchmark reference は `#101` formal rerun `r20260415_local_nankan_pre_race_ready_formal_v1` であり、`#103` はこの issue ではなく Stage 1 architecture parity issue として扱う
- rebuilt provenance-aware line でも high ROI が異常に残る場合のみ、次の root-cause issue を `leak residual` か `policy optimism residual` のどちらか 1 本に narrowed する
