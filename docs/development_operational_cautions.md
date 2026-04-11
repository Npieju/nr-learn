# Development Operational Cautions

## 1. Purpose

この文書は、`nr-learn` の自動開発運用で事故になりやすい注意事項を固定するための正本である。

標準フローの補足ではなく、日常運用で守る hard caution として扱う。

## 2. Mandatory Cautions

### 2.0 Docs Must Remain Navigable Without Grep

docs は検索で掘らないと読めない状態を許容しない。

必須要件:

- current operator が最初に開く入口 doc を明示する
- draft / reference library は raw listing ではなく index から辿れる
- 同じ判断や status を複数 docs へ平行更新しないと整合しない構造を増やさない
- 新しい doc を追加する前に、既存正本へ統合できない理由を確認する
- snapshot / review memo は version または date を file 名に入れ、作成時点の記録だと分かるようにする
- commit しない一時 doc や作業メモは tracked な `docs/` に置かず、gitignore 管理の scratch/local path に隔離する

default rule:

- まず既存 doc へ統合する
- 統合できない場合は新規 doc 本体より先に index / source-of-truth を定義する
- summary doc を増やすときは、何を current source-of-truth にして何を versioned snapshot / 参照専用にするかを同時に固定する
- versioned snapshot は作成時点の情報を保持するためのものとし、current info を追って平行更新しない

review rule:

- grep 前提の docs 導線は reject 候補にする
- parallel update が前提の doc 追加は reject 候補にする
- version/date が無く、current doc と snapshot doc の境界が曖昧な新規文書は reject 候補にする
- commit 予定のない一時 doc を tracked docs に追加する変更は reject 候補にする
- docs 整備だけが長く滞留し、本線変更の commit を止める状態を作らない
- heavy run が無い時間帯に non-blocking docs をまとめて前景化し、結果依存の current doc 更新まで同時に膨らませる進め方は reject 候補にする

timing rule:

- running 中に進めてよい docs は、issue 整理、historical note 付け、index 整備、review package の下書きなど、結果未確定でも無駄になりにくいものに限る
- running 完了後に触る current source-of-truth は、artifact 数値、decision summary、current priority の最小更新に限る
- heavy run 完了後に docs を大きく書き始めるのではなく、下書きは waiting time に進め、確定反映だけを最後に行う

### 2.1 Progress Is Required For Long-Running Sources

数秒で終わらないことが見込まれるすべての source には、progress を入れる。

ここでいう source には次を含む。

- `scripts/` 配下の CLI
- `src/` 配下の long-running function
- crawl / ingest / backfill / training / evaluation / benchmark gate
- large file write や multi-stage wrapper

理由は単純である。progress がないと、実行中なのか無限ループなのか、外から区別できない。自動運用ではこの曖昧さ自体が障害になる。

必須要件:

- 開始時に phase を出す
- 長時間区間では heartbeat か定期進捗を出す
- 完了時に completed message を出す
- 失敗時に停止 phase が分かる
- progress は possible な限り分母付きで出す
- possible なら件数、fold、row、file、cycle などの進捗単位を出す
- 重くなりうる task に unbounded な no-output 区間を残さない

bounded progress rule:

- operator は長時間 task が「遅いだけ」なのか「落ちた / stuck した」のかをログだけで区別できなければならない
- したがって heavy compute でも bounded interval で checkpoint を出す
- checkpoint は `x/y`, `fold a/b`, `rows a/b`, `files a/b` のように、operator が全体に対する位置を読める形を優先する
- 分母が出せない heartbeat 単独の行は、生存確認としては有効でも progress としては不十分とみなす
- heartbeat だけでは phase 内 progress が読めない場合、fold / search_step / row / file / cycle などの中間 checkpoint を追加する
- 目安として 60 秒を超える no-output 区間は未完成扱いにする
- 60 秒以内でも、重い内側ループで数分無音になりうる設計は review で reject 候補にする
- operator が VS Code の local terminal を見られない実行経路でも確認できるよう、長時間 task は repo 内の log file に live 出力されるべきである
- progress の確認先は terminal ではなく stable な log file path として毎回明示する

bounded interrupt rule:

- operator が「まだ健全に進んでいる」と判断できない no-output 区間が続く場合、その task は途中で切れる設計にする
- 具体的には `max_silent_seconds` や `max_elapsed_minutes` のような interrupt guard を持てる形が望ましい
- guard 到達時は silent のまま待ち続けず、`interrupted_for_operator_review` のような明示的な停止理由を残す
- 再実行は narrower search、smaller split、checkpoint 粒度改善のいずれかを伴うことを原則にする

motivating example:

- NAR `wf_feasibility` のように CPU を使い続ける task でも、checkpoint が疎いと operator からは dead/stuck と区別しづらい
- この種の task は「生存 heartbeat」だけでなく「内側探索の進捗 checkpoint」まで必須にする
- それでも bounded interval を超えて silent なら、operator が途中で切って再実行方針へ移れる guard が必要である

推奨実装:

- `racing_ml.common.progress.ProgressBar`
- `racing_ml.common.progress.Heartbeat`
- 既存 logger を通した定期 progress message

最低ライン:

- operator が 30 秒以内に「生きている」と判断できる
- operator が停止位置をログから特定できる
- operator が VS Code から開ける file path を 1 本で把握できる

PR / review rule:

- 数秒超の処理に progress がない新規 source は原則 accept しない
- 既存 source を触るとき progress 不足を見つけたら、同時に是正するか follow-up issue を切る
- 重い task に 60 秒超の no-output 区間が残る変更は原則 accept しない

### 2.2 Long-Running Jobs Must Preserve Readability

progress は単に大量ログを流せばよいわけではない。

- prefix を固定する
- phase 名を安定させる
- 1 行ごとの意味を保つ
- high-frequency spam を避ける

良い progress は「今どこか」「どれくらい進んだか」「何で止まったか」を短時間で伝えられる。

最低でも次のどちらかを満たすべきである。

- 明示的な分母付き progress がある
- 分母をまだ出せない理由と、次に分母付き progress が出る境界が明示されている

### 2.3 Dry-Run Is Preferred Before Formal Execution

formal gate や wrapper 実行前には、possible な限り dry-run で command 解決と output path を先に確認する。

これは誤設定による長時間ジョブ浪費を避けるためである。

### 2.4 Every Long Job Needs An Artifact Or Manifest Exit

長時間ジョブは、成功時だけでなく失敗時や blocked 時も、possible な限り manifest や summary artifact を残す。

operator は terminal scroll ではなく artifact から停止点を読める状態を目指す。

## 3. Default Review Checklist

長時間処理に触る PR では最低限次を確認する。

- progress start がある
- heartbeat または進捗更新がある
- completion がある
- failure phase が読める
- output artifact path が読める

## 4. Scope In This Repository

この caution は特に次に強く適用する。

- `scripts/run_train.py`
- `scripts/run_evaluate.py`
- `scripts/run_revision_gate.py`
- data crawl / backfill wrappers
- local / mixed benchmark gate wrappers

## 5. Relationship To Other Standards

- 開発手順全体は `docs/autonomous_dev_standard.md`
- AI 開発の標準は `docs/ai_coding_best_practices.md`
- ML 開発の stage は `docs/ml_model_development_standard.md`

この文書は、それらすべてに横断して効く運用 caution を定義する。
