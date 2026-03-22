# Evaluation と Promotion Gate ガイド

## 1. この文書の役割

この文書は、`nr-learn` で正式な改善判断を行うときの evaluation 導線をまとめたガイドである。

対象は次の 4 つである。

1. `run_evaluate.py` の位置づけ
2. `stability_assessment` の見方
3. `run_promotion_gate.py` の役割
4. `revision` 単位での正式判断フロー

## 2. 基本方針

- 短い smoke / probe は候補の絞り込みに使う。
- 正式な採用判断は `run_evaluate.py` と `run_promotion_gate.py` を通した revision 単位で行う。
- 単発の高 ROI や短い比較結果を、そのまま benchmark の代表値や昇格根拠には使わない。

## 3. `run_evaluate.py` の役割

本命候補の判定は、`run_evaluate.py` による nested walk-forward を基準に行う。

理由は次のとおりである。

1. 同じ期間で最適化と評価を兼ねないため。
2. fold ごとの no-bet や gating failure まで見られるため。
3. `stability_assessment` により、短窓 run を formal judgment から分離できるため。

基本例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py \
  --profile current_best_eval \
  --max-rows 120000
```

より厳密な評価例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py \
  --profile current_best_eval \
  --max-rows 200000 \
  --wf-mode full
```

## 4. `stability_assessment` の見方

`run_evaluate.py` の summary / manifest には `stability_assessment` と `stability_guardrail` が保存される。

ここで重要なのは次の区別である。

- `representative`
  - 正式判断の基盤として扱える run
- `caution`
  - 情報としては見るが、昇格判断は慎重に扱う run
- `probe_only`
  - 方向確認専用であり、正式判断の根拠には使わない run

つまり、正式な改善判断で最低限必要なのは `stability_assessment=representative` である。

## 5. 見るべき主要 artifact

正式判断でまず見るものは次の 2 つである。

- `artifacts/reports/evaluation_summary.json`
- `artifacts/reports/evaluation_manifest.json`

ここで見る項目:

1. `stability_assessment`
2. `stability_guardrail`
3. ROI / bets / total net 系の主要指標
4. versioned artifact の有無

artifact の全体像は [artifact_guide.md](artifact_guide.md) を参照する。

## 6. `run_promotion_gate.py` の役割

`run_promotion_gate.py` は、evaluation 単体ではなく、次をまとめて昇格可否として判定する。

1. evaluation manifest の整合性
2. evaluation の `stability_assessment=representative`
3. matching な walk-forward feasibility summary の存在
4. feasibility summary 側の `stability_assessment=representative`
5. feasible fold 数

基本例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_promotion_gate.py \
  --evaluation-manifest artifacts/reports/evaluation_manifest.json
```

必要なら `--wf-summary` を明示できるが、通常は config / data / feature tuple から自動解決される。

## 7. promotion gate report の見方

promotion gate の report では、次を見る。

1. `status`
2. `decision`
3. `checks`
4. `blocking_reasons`
5. `warnings`
6. `wf_diagnostics`

特に重要なのは次である。

- `status=pass` か `block` か
- `decision=promote` か `hold` か
- feasible fold 数が足りているか
- dominant failure reason が何か

## 8. walk-forward feasibility の扱い

promotion gate は matching な `wf_feasibility_diag_*.json` を参照する。

ここでのポイント:

- feasible fold が少なすぎる候補は hold に寄る
- fold の valid/test が短窓になりやすい場合、warning が出ても即 fail とは限らない
- ただし全体として `representative` を満たさない run は昇格根拠にしない

2026-03-22 の `current_bankroll_candidate` は、この節の典型例になった。

- evaluation summary 自体は `representative`
- matching な `wf_feasibility_diag` も `representative`
- それでも promotion gate は `wf_feasible_fold_count=0` で `block/hold`

つまり、`evaluation_representative=true` だけでは足りず、matching WF の fold support まで満たして初めて昇格候補になる。特に `dominant_failure_reason=min_bets` で `binding_min_bets_source=absolute` が揃っている場合は、「方向性はあっても support が足りない」状態として読む。

## 9. `revision` 単位での正式判断

正式な判断は、個別コマンドをばらばらに解釈するより、revision 単位でまとめて扱うほうがよい。

標準フロー:

1. train
2. evaluate
3. promotion gate
4. pass したものだけ revision として扱う

この流れは `run_revision_gate.py` でまとめて実行できる。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --profile current_best_eval \
  --revision r20260321a \
  --evaluate-max-rows 200000 \
  --evaluate-wf-mode full
```

重い実行の前に orchestration だけ確認したいときは `--dry-run` を付ける。

主な出力:

- `artifacts/reports/promotion_gate_<revision>.json`
- `artifacts/reports/revision_gate_<revision>.json`

`--dry-run` のときは `revision_gate_<revision>.json` に planned command と `status=dry_run` が保存される。

軽量 smoke を real 実行したい場合は、`--train-max-train-rows` / `--train-max-valid-rows` と evaluate 側 row 制限で train/evaluate の負荷を下げられる。

## 10. 何を evaluation で判断し、何を判断しないか

evaluation / promotion gate で判断するもの:

- benchmark 上の正式な改善
- revision の採否
- representative な根拠の有無
- feasible fold の十分性

evaluation だけでは決めないもの:

- runtime の挙動差
- actual calendar 上の細かな policy change の見え方
- rollback 候補の runtime 比較

これらは [serving_validation_guide.md](serving_validation_guide.md) と合わせて見る。

## 11. よくある解釈ミス

避けるべき解釈は次のとおりである。

- 短窓で ROI が良かったので昇格
- serving compare で net が良かったので benchmark 更新
- `probe_only` を正式判断の根拠にする

このプロジェクトでは、短窓の良い結果は次の full evaluation を回す理由にはなっても、昇格の十分条件にはならない。

## 12. 関連 script

- [../scripts/run_evaluate.py](../scripts/run_evaluate.py)
- [../scripts/run_validate_evaluation_manifest.py](../scripts/run_validate_evaluation_manifest.py)
- [../scripts/run_promotion_gate.py](../scripts/run_promotion_gate.py)
- [../scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
- [../scripts/run_wf_feasibility_diag.py](../scripts/run_wf_feasibility_diag.py)