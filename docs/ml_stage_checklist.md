# ML Stage Checklist

## 1. Purpose

このチェックリストは、`docs/ml_model_development_standard.md` を issue 実行時にそのまま使える形へ落とした運用版である。

issue、PR、レビュー、handoff のどこでもこのまま使ってよい。

## 2. Stage Checklist

### Stage 1: Objective Definition

- [ ] 今回の primary objective が 1 文で書かれている
- [ ] non-goals が明記されている
- [ ] primary metric が定義されている
- [ ] guardrail metric が定義されている

### Stage 2: Dataset Freeze

- [ ] universe が固定されている
- [ ] train / validation / holdout window が固定されている
- [ ] data source version が固定されている
- [ ] experiment 中に dataset scope を変えない前提が確認されている

### Stage 3: Baseline Artifact Confirmation

- [ ] current profile が明記されている
- [ ] current revision が明記されている
- [ ] comparison に使う evaluation summary が明記されている
- [ ] comparison に使う promotion gate artifact が明記されている
- [ ] 必要なら compare dashboard が明記されている

### Stage 4: Hypothesis Design

- [ ] hypothesis が `if ..., then ..., while ...` の形で書かれている
- [ ] change type が feature / model / ensemble / policy のどれかに分類されている
- [ ] 期待する upside が書かれている
- [ ] 許容する downside が書かれている

### Stage 5: Implementation

- [ ] in-scope files が明記されている
- [ ] out-of-scope files が暗黙に広がっていない
- [ ] docs / tests / config 更新要否が確認されている
- [ ] 変更理由を後から説明できる粒度に保たれている

### Stage 6: Smoke Validation

- [ ] syntax / unit / smoke のどれを回すか決まっている
- [ ] obvious regression がない
- [ ] output schema が壊れていない
- [ ] exposure / bets / net / bankroll の極端な崩れがない

### Stage 7: Formal Evaluation

- [ ] representative evaluation を実施した
- [ ] walk-forward or nested metrics を確認した
- [ ] feasible fold coverage を確認した
- [ ] drawdown / bankroll を確認した
- [ ] actual-date compare または replay compare を確認した

### Stage 8: Promotion Decision

- [ ] 結論が `promote / keep as candidate / reject` のどれかで書かれている
- [ ] 判断理由が書かれている
- [ ] artifact path が issue または PR に残っている
- [ ] benchmark / roadmap / public snapshot 更新要否が判定されている

## 3. Stop Condition Checklist

- [ ] primary metric が baseline 比で悪化した
- [ ] drawdown / bankroll が許容外になった
- [ ] bets が実運用として薄すぎる
- [ ] leakage or data bug の疑いがある
- [ ] 変更理由を説明できなくなった

いずれかに該当するなら、その issue は打ち切るか分割し直す。
