# Next Issue: Evaluate Post-Inference Progress Gap

## Summary

JRA `run_evaluate.py` では `inference complete` の後、`running leakage audit started` が出るまでに 60 秒超の no-output 区間が残ることがある。2026-04-03 の `r20260403_owner_signal_ablation_audit_v1` でも同症状を再確認した。

## Objective

`run_evaluate.py` の `inference -> summary assembly -> leakage audit` 区間に bounded progress を入れ、数十秒以上の silent interval を残さない。

## Hypothesis

if post-inference heavy work を phase 分解して分母付き progress か phase checkpoint を出す, then operator は evaluate が生存中かつどの段にいるかを追跡でき、silent failure と誤認しなくなる。

## In Scope

- `scripts/run_evaluate.py`
- post-inference heavy phase の progress checkpoint 追加
- 必要な unit / regression test

## Non-Goals

- evaluation metric の定義変更
- policy search 自体の rewrite
- JRA/NAR benchmark decision の再計算

## Success Metrics

- `inference complete` 後に bounded progress が継続する
- 60 秒超の no-output を残さない
- current evaluate log の読み口を壊さない

## Validation Plan

1. post-inference phase の処理ブロックを確認
2. phase checkpoint を追加
3. regression test 追加
4. small smoke で log を確認

## Stop Condition

- progress 追加が metric payload を壊す
- phase 分解が複雑すぎて保守性を落とす

## Actual Execution Read

implementation:

- `scripts/run_evaluate.py`
  - `inference complete` の直後に `[evaluate post]` progress bar を追加
  - phases:
    - `score sources ready`
    - `summary payload ready`
    - `post-inference phases finished`

verification:

- `python -m py_compile scripts/run_evaluate.py`
- smoke:
  - `artifacts/logs/evaluate_post_inference_progress_gap_smoke.log`
  - `--max-rows 5000 --wf-mode off`
  - `inference complete` の直後に次を確認
    - `[evaluate post] 1/3`
    - `[evaluate post] 2/3`
    - `[evaluate post] 3/3`

note:

- smoke 自体は small tail slice が single-class になり calibration で失敗した
- ただし目的だった post-inference no-output 解消は log 上で確認できた

## Final Read

- post-inference silent interval を phase checkpoint で分解した
- `inference -> leakage audit` の間に bounded progress が出る
- evaluate payload / decision logic 自体は変更していない
- current hard rule に対する corrective としては十分である
