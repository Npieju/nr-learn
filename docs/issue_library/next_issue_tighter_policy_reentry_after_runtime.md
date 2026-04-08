# Next Issue: Tighter Policy Reentry After Runtime Refresh

Historical note:

- この draft は `#40` として experiment reentry decision まで完了している。
- runtime refresh 後の tighter policy reentry は completed queue-transition reference として扱い、この文書は historical issue source として使う。

## Summary

runtime queue で primary tail cache まで default mainline に昇格し、current baseline smoke は faster path 上でも drift なく再確認できた。次は experiment queue を再開し、formal support が最も強い `tighter policy search` family に戻る。

## Objective

runtime 改善後の current baseline を土台に、`tighter policy search` family を再び primary experiment line として再開し、September difficult regime 向けの defensive improvement を formal に詰める。

## In Scope

- `tighter policy search` family の current baseline 再接続
- promoted baseline との比較前提の issue framing
- `abs80` / `abs90` 系の defensive frontier 再読
- next formal candidate の narrowing

## Non-Goals

- new runtime optimization
- recent-heavy retrain family の再開
- seasonal alias の再整理
- NAR work

## Success Criteria

- next experiment line が `tighter policy search` に固定される
- current promoted baseline に対する compare framing が明文化される
- next formal candidate が 1 本に絞られる

## Starting Context

current reading:

- baseline remains `current_recommended_serving_2025_latest`
- `tighter policy search` is still the strongest formal-support defensive family
- runtime-default smoke on `configs/data_2025_latest.yaml` is now `loading training table 0m02s`, total `0m15s`
- recent-heavy family remains analysis-first and regime-specific
- policy ranking still puts `tighter policy search` at Rank 1

## Suggested Validation

- current baseline reduced smoke / benchmark pointer refresh
- tighter policy candidate artifact reread
- if needed, dry-run revision gate on the shortlisted next candidate
