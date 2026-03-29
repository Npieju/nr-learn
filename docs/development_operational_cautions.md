# Development Operational Cautions

## 1. Purpose

この文書は、`nr-learn` の自動開発運用で事故になりやすい注意事項を固定するための正本である。

標準フローの補足ではなく、日常運用で守る hard caution として扱う。

## 2. Mandatory Cautions

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
- possible なら件数、fold、row、file、cycle などの進捗単位を出す

推奨実装:

- `racing_ml.common.progress.ProgressBar`
- `racing_ml.common.progress.Heartbeat`
- 既存 logger を通した定期 progress message

最低ライン:

- operator が 30 秒以内に「生きている」と判断できる
- operator が停止位置をログから特定できる

PR / review rule:

- 数秒超の処理に progress がない新規 source は原則 accept しない
- 既存 source を触るとき progress 不足を見つけたら、同時に是正するか follow-up issue を切る

### 2.2 Long-Running Jobs Must Preserve Readability

progress は単に大量ログを流せばよいわけではない。

- prefix を固定する
- phase 名を安定させる
- 1 行ごとの意味を保つ
- high-frequency spam を避ける

良い progress は「今どこか」「どれくらい進んだか」「何で止まったか」を短時間で伝えられる。

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
