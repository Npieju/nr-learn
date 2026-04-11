# AGENTS.md

## Purpose

このリポジトリは、基本的に自動 coding を第一選択として運用する。

人は目標、優先順位、受け入れ条件、昇格判断を担い、AI は実装、反復、定型検証、文書化を担う。

## Read This First

1. `README.md`
2. `docs/strategy_v2.md`
3. `docs/autonomous_dev_standard.md`
4. `docs/ai_coding_best_practices.md`
5. `docs/ml_model_development_standard.md`
6. `docs/development_operational_cautions.md`
7. `docs/initial_issue_backlog.md`
8. `docs/tail_loader_equivalence_gate_standard.md`

既存の `docs/` は reference / legacy として残すが、新しい標準は上の文書を優先する。

## Repo Priorities

- primary objective: JRA 正本データで長期運用 `ROI > 1.20` を狙う
- primary benchmark universe: `JRA`
- parallel track: `NAR ingestion/readiness`
- do not mix `NAR` into core benchmark decisions before readiness is established

## Standard Workflow

1. issue を読む
2. issue がなければ先に issue を作るか current queue に下書きを残す
3. objective / non-goals / success metrics / validation を確認する
4. explore して影響範囲を把握する
5. `docs/ml_model_development_standard.md` の stage を特定する
6. plan を作る
7. code / test / docs を更新する
8. verify を実行する
9. artifact path と residual risk を issue に残す
10. close 条件を満たしたら decision summary を追記して issue を close する

## Hard Rules

- 1 issue = 1 measurable hypothesis
- issue を立てずに実装を始めない
- 短窓の良化だけで promote しない
- formal gate と artifact を優先する
- broad strategy change は human review 前提
- JRA baseline 更新は human judgment を要求する
- 不足する規則は会話ではなく docs / tests / instructions に追加する
- Git の更新を長く滞らせない。同じ変更単位の code / config / script / docs は意味のある小さめの batch でまとめ、issue の進捗に対応する commit / push を早めに行う。
- commit しない一時 doc や作業メモを tracked な `docs/` 配下へ置かない。必要な一時メモは gitignore 管理の scratch/local path に隔離し、current source-of-truth や version/date 付き snapshot と混在させない。
- docs を増やすこと自体は許容するが、index から辿れる単一導線を優先し、同じ変更を複数 docs に平行反映しないと維持できない構造は原則追加しない。
- docs は「current source-of-truth」か「version/date 付き snapshot/reference」に役割を分ける。snapshot/reference は作成時点の記録として凍結し、最新化のたびに平行更新しない。
- docs 整備は本線作業と切り離して長く滞留させない。新しい index / summary を増やす前に、既存 docs へ統合するか、version/date を付けた snapshot として独立させる。
- 重い train / evaluate / gate / compare が running 中なら、その待ち時間では non-blocking な docs 整備、issue 整理、review package の下書きを優先する。artifact 数値や結論に依存する current source-of-truth 更新は、結果確定後に必要最小限だけ行う。
- 数秒で終わらない source に progress がない変更は未完成扱いにする
- progress は possible な限り分母付きで出し、生存確認だけの heartbeat を progress 完了扱いにしない
- 重い task に 60 秒超の no-output 区間が残る変更は未完成扱いにする
- 重い task は bounded interrupt rule を持ち、長時間 silent のまま無制限に走らせない
- Pylance の code snippet execution で重い Python script を回さない。重い Python 実行は workspace Python environment を確認した上で terminal から行う。
- 重い Python 実行では外部ログファイルを必須にする。`--log-file` を優先し、未対応 script は `artifacts/logs/...` へ redirect または `tee` し、ログ path を進捗と最終報告に明記する。
- Python 実行が environment 起因で失敗したときは、安易に別経路へ逃げず、interpreter / package / 実行 prefix を確認して environment を修復してから再実行する。

## Important Entry Points

- training: `scripts/run_train.py`
- evaluation: `scripts/run_evaluate.py`
- revision gate: `scripts/run_revision_gate.py`
- benchmark references: `docs/benchmarks.md`, `docs/public_benchmark_snapshot.md`
- dev process: `docs/autonomous_dev_standard.md`
- AI coding standard: `docs/ai_coding_best_practices.md`
- ML development standard: `docs/ml_model_development_standard.md`
- tail loader gate standard: `docs/tail_loader_equivalence_gate_standard.md`

## Issue Quality Bar

AI に渡す issue には次が必要である。

- objective
- hypothesis
- in-scope surface
- non-goals
- success metrics
- validation plan
- stop condition

足りない場合は、いきなり実装せず issue を具体化する。
