# Next Issue: Post-Runtime Benchmark Refresh And Experiment Reentry

## Summary

runtime queue で primary tail cache まで default mainline に昇格したので、次は速度改善後の土台で benchmark refresh と experiment queue を再開する段階に戻る。

## Objective

faster default data path を前提に current benchmark / smoke を再確認し、次の ROI-improving experiment line を再開する。

## In Scope

- current mainline smoke / benchmark refresh
- promoted operational baseline の artifact refresh
- next experiment family の再優先付け
- issue queue を runtime から experiment へ戻す

## Non-Goals

- new runtime optimization track
- NAR work
- broad docs reorganization

## Success Criteria

- runtime improvements 後の baseline artifact が新しい default path で確認される
- next experiment issue が 1 本に絞られる
- queue が experiment-first に戻る

## Starting Context

current accepted state:

- `configs/data_2025_latest.yaml` now includes freshness-guarded primary tail cache
- default smoke read is `loading training table 0m02s`, total `0m15s`
- summary equivalence against explicit alias differs only by `run_context.data_config`
- runtime operations issues `#36`, `#37`, `#38` are complete

## Suggested Validation

- reduced smoke on current mainline
- benchmark / evaluation artifact refresh where needed
- issue shortlist refresh for next experiment family
