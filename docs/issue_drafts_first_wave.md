# First-Wave Issue Drafts

## 1. How To Use

この文書は、最初の 3 issue を GitHub にそのまま切るための下書きである。

必要なら各項目を微修正して、そのまま issue template に貼り付けて使う。

## 2. Issue Draft: Formal KPI Definition for ROI>1.20

### Title

`[experiment] Formal KPI definition for ROI>1.20`

### Universe

`JRA`

### Category

`Evaluation`

### Objective

JRA 正本データで `ROI > 1.20` を狙う長期運用開発のために、何を成功とみなすかを formal に定義する。短窓の当たりではなく、formal evaluation と operational compare の両方で読める KPI セットを固定したい。

### Hypothesis

if `ROI > 1.20` の success 条件を primary metric と guardrail metric に分解して明文化する, then experiment の採否と revision gate の判断が一貫し, AI coding の無駄打ちが減る, while exploratory speed is preserved.

### In-Scope Surface

- `docs/roi120_kpi_definition.md`
- `docs/strategy_v2.md`
- `docs/ml_model_development_standard.md`
- `docs/benchmarks.md`
- `docs/evaluation_guide.md`
- 必要なら issue / PR template の評価欄

### Non-Goals

- 新モデルの実装
- NAR benchmark の定義
- baseline profile の切り替え

### Success Metrics

- primary metric, guardrails, minimum support 条件が文章で固定される
- formal ROI, feasible folds, drawdown, bet volume の優先順位が明文化される
- promote / candidate / reject の判断軸が曖昧でなくなる

### Eval Plan

- smoke: docs 間の矛盾がないか確認する
- formal: strategy / benchmark / evaluation guide の参照先を揃え、今後の revision gate issue が同じ KPI を参照できる状態にする

### Validation Commands

- `sed -n '1,220p' docs/strategy_v2.md`
- `sed -n '1,260p' docs/ml_model_development_standard.md`
- `sed -n '1,260p' docs/benchmarks.md`
- `sed -n '1,260p' docs/evaluation_guide.md`

### Expected Artifacts

- docs diff
- `docs/roi120_kpi_definition.md`
- KPI definition を参照する issue / PR links

### Stop Condition

- KPI が多すぎて運用判断に使えない
- JRA の現行 formal artifact と整合しない
- ROI を上げる代わりに stability を無視する定義になってしまう

## 3. Issue Draft: Current JRA Baseline Artifact Inventory and Gap Audit

### Title

`[experiment] Current JRA baseline artifact inventory and gap audit`

### Universe

`JRA`

### Category

`Evaluation`

### Objective

現在の JRA baseline を支える artifact を棚卸しし、ROI>1.20 に向けて足りない比較材料や missing artifact を洗い出す。今後の AI coding が baseline を誤読しない状態を作りたい。

### Hypothesis

if current JRA baseline artifacts と gap を一覧化する, then new experiments can compare against a fixed baseline with less ambiguity, while keeping the current benchmark source of truth intact.

### In-Scope Surface

- `artifacts/reports/`
- `docs/public_benchmark_snapshot.md`
- `docs/benchmarks.md`
- `docs/project_overview.md`
- `docs/roadmap.md`

### Non-Goals

- artifact 再生成
- baseline の昇格判断
- NAR artifact の統合

### Success Metrics

- baseline profile / revision / evaluation / promotion / compare artifacts が一覧化される
- missing or stale artifacts が明示される
- 次の experiment issue が参照すべき baseline セットが固定される

### Eval Plan

- smoke: 現在 docs に記載された artifact path が存在するか確認する
- formal: baseline inventory と gap audit を文書化し、issue / PR から参照できるようにする

### Validation Commands

- `rg "promotion_gate_|serving_compare_dashboard_|current_recommended_serving_2025_latest" docs artifacts/reports -n`
- `rg "current_recommended_serving_2025_latest|r20260325" docs -n`
- `rg --files artifacts/reports | sed -n '1,240p'`

### Expected Artifacts

- `docs/jra_baseline_artifact_inventory.md`
- gap audit markdown
- 必要なら stale / missing artifact list

### Stop Condition

- baseline の source of truth が複数あって 1 issue で収束しない
- docs と artifacts の不整合が大きく、先に definition issue が必要になる

## 4. Issue Draft: ML Model Development Stage Checklist Standardization

### Title

`[experiment] ML model development stage checklist standardization`

### Universe

`JRA`

### Category

`Evaluation`

### Objective

AI と人間が同じ手順でモデル開発を進められるよう、stage-based checklist を運用導線へ組み込む。issue 作成から review まで、どの stage にいるかを迷わない状態にしたい。

### Hypothesis

if ML model development stage checklist is standardized and referenced from the main operating docs, then experiment execution becomes more consistent and easier to automate, while preserving room for fast iteration.

### In-Scope Surface

- `docs/ml_model_development_standard.md`
- `docs/ml_stage_checklist.md`
- `AGENTS.md`
- `.github/copilot-instructions.md`
- 必要なら issue template

### Non-Goals

- 新しい ML アルゴリズム導入
- benchmark 数値の更新
- NAR readiness gate 詳細定義

### Success Metrics

- stage checklist が repo に存在する
- agent instructions から参照できる
- first-wave issues が checklist に沿って記述できる

### Eval Plan

- smoke: docs 間の参照関係が通るか確認する
- formal: first-wave issue drafts が checklist 順で読めるか確認する

### Validation Commands

- `sed -n '1,260p' docs/ml_model_development_standard.md`
- `sed -n '1,260p' docs/ml_stage_checklist.md`
- `sed -n '1,220p' AGENTS.md`
- `sed -n '1,220p' .github/copilot-instructions.md`

### Expected Artifacts

- stage checklist markdown
- docs diff

### Stop Condition

- checklist が長すぎて issue 作成時に使われない
- stage と current workflow が矛盾する

## 5. Next Draft Pointer

次の issue 候補 `Feature family ranking from existing artifacts` は、[feature_family_ranking.md](feature_family_ranking.md) を下書き兼 outcome として使う。

その次の issue 候補 `Policy family shortlist for high-ROI / controlled-drawdown experiments` は、[policy_family_shortlist.md](policy_family_shortlist.md) を下書き兼 outcome として使う。

その直後の実行候補は [next_issue_tighter_policy_frontier.md](issue_library/next_issue_tighter_policy_frontier.md) を使う。
