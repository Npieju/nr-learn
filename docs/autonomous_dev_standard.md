# Autonomous Development Standard

## 1. Purpose

この文書は、`nr-learn` を issue 駆動かつ AI 支援前提で継続改善するための標準手順である。

以後の開発は、既存 docs の自由記述よりも、この標準フローに合わせて進める。

## 2. Recommended Stack

- backlog system: GitHub Issues
- execution board: GitHub Projects
- code review: Pull Request
- merge style: short-lived branch
- quality gates: local tests + revision artifacts + human approval
- source of truth for strategy: issue body + linked artifacts

## 3. Issue Taxonomy

### 3.1 Epic

数週間単位の成果目標を表す。

例:

- `JRA long-run ROI > 1.20`
- `NAR ingestion readiness`
- `formal evaluation hardening`

### 3.2 Experiment Issue

1 つの仮説を検証する最小単位である。原則として 1 issue で 1 つの change surface に絞る。

例:

- pace feature refresh
- value stack retrain
- liquidity guardrail tuning

### 3.3 Revision Gate Issue

候補を formal に審査するための issue である。experiment issue と分離し、artifact を固定する。

### 3.4 Ops / Automation Issue

CI、artifact manifest、crawler、dashboard、template 整備など、開発速度と再現性を上げる変更を扱う。

## 4. Required Fields For Every Issue

すべての issue に最低限必要な項目は次のとおりである。

- objective
- hypothesis
- in-scope files or pipeline surface
- dataset scope
- success metrics
- eval plan
- stop condition
- linked artifacts or expected output paths

## 5. Project Board States

Project の状態は次の 7 列で固定する。

1. `Backlog`
2. `Ready`
3. `In Progress`
4. `Awaiting Eval`
5. `Awaiting Review`
6. `Blocked`
7. `Done`

自動化ルールの推奨:

- issue 作成時に `Backlog`
- branch / PR 紐付け時に `In Progress`
- artifact 出力後に `Awaiting Eval`
- PR open 後に `Awaiting Review`
- merge 後に `Done`

## 6. Definition Of Ready

着手前に次が揃っていること。

- GitHub issue が存在する
- 変更の対象が 1 文で言える
- JRA / NAR / mixed のどれを触るか明示されている
- 成功判定の metric が明示されている
- smoke で見るものと formal gate で見るものが分かれている
- rollback か abandon の条件が書かれている

## 7. Definition Of Done

完了とみなすには次を満たす。

- code or docs change が存在する
- 必要な tests / smoke / scripts を実行している
- artifact path が issue または PR に記録されている
- 未解決リスクが明文化されている
- 次の判断が `promote / keep as candidate / reject` のどれかで書かれている
- issue に decision summary を追記して close できる状態である

## 8. AI Agent Contract

AI に任せる作業でも、最低限の契約を固定する。

- AI は issue を読んでから作業する
- issue がない作業は、まず issue を作るか current queue に下書きを追加する
- issue にない前提変更は PR で明示する
- acceptance criteria がない issue は `Ready` に上げない
- strategy change は必ず artifact とセットでレビューする
- docs only 変更でも、次の operator action が明確であること
- 数秒で終わらない source を追加・更新する場合は progress を必須にする
- 重い task は bounded progress output を持ち、60 秒超の no-output 区間を残さない
- equivalence harness を持つ optimization は、accepted gate mode を docs に固定する

## 9. Branch And PR Policy

- branch 名は `issue-<number>-<slug>` を推奨する
- branch の寿命は数日以内を標準にする
- 1 PR に複数仮説を混ぜない
- 長時間の大型施策は epic 配下の小 issue に分解する
- draft PR を早めに作り、artifact と判断保留点を残す

## 10. Standard Execution Loop

1. epic を定義する
2. measurable な experiment issue に分解する
3. issue に hypothesis / eval plan / stop condition を書く
4. 小さい branch で実装する
5. smoke で壊れていないことを確認する
6. 有望なものだけ revision gate issue を切る
7. formal artifact を出して promote 可否を決める
8. issue に artifacts と decision summary を追記する
9. close 条件を満たした issue を close する
10. benchmark / roadmap / public snapshot を必要箇所だけ更新する

ML 系 issue の実行順は [ml_model_development_standard.md](ml_model_development_standard.md) を優先する。

## 11. Recommended Automation

まず自動化する価値が高いのは次の順である。

1. issue template
2. PR template
3. project status automation
4. revision artifact の自動リンク
5. benchmark refresh checklist
6. NAR readiness checklist

AI coding の標準は [ai_coding_best_practices.md](ai_coding_best_practices.md) を参照する。

## 12. How This Applies To `nr-learn`

現時点の標準運用は次のとおりである。

- ROI 改善の主戦場は JRA
- NAR は ingestion と readiness を別 epic で管理する
- mixed-universe は benchmark 正本から分離する
- exploratory memo は docs に蓄積せず issue / artifact に残す
- 原則として issue を立てずに実装を始めない
- 完了した案件は open のまま放置せず、decision summary を残して close する
- `revision` は formal gate を通した単位でだけ切る
- 基本的に自動 coding を第一選択にする
- ただし issue が AI-ready でない仕事は、先に issue 整形から始める
- 数秒超の long-running source は progress / heartbeat / completion を持たせる

## 13. Source Notes

この標準は、GitHub Projects の自動化、OpenAI の eval best practices、Trunk-Based Development の short-lived branch 原則をこのリポジトリ向けに翻訳したものである。

運用上の注意は [development_operational_cautions.md](development_operational_cautions.md) を参照する。
