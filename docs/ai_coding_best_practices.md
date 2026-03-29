# AI Coding Best Practices

## 1. Positioning

この文書は、`nr-learn` を基本的に自動 coding で前進させるための実務標準である。

前提は単純である。ゴールは人が決める。実装、反復、定型検証、文書化、軽微な修正は AI に寄せる。人は優先順位、受け入れ条件、昇格判断、危険な変更の承認に集中する。

## 2. What Recent Best Practice Converges On

2026-03-29 時点の公式ガイドを読むと、実務上の結論は次の 6 本に収束している。

1. issue は prompt そのものとして設計する
2. 大きい仕事は explore → plan → code → verify に分解する
3. custom instructions / skills / hooks で repo の規律を機械可読にする
4. eval を継続実行し、agent の変更を artifact で比較する
5. agent の権限と対象範囲を狭く保つ
6. 人は review と merge judgment を手放さない

## 3. Standard Allocation Of Work

### 3.1 Human-owned

- north star と KPI の定義
- Epic の優先順位
- acceptance criteria の確定
- JRA / NAR / mixed の境界管理
- baseline 更新判断
- security / legal / production-critical 変更の最終承認

### 3.2 AI-owned by default

- 小中規模の実装
- unit test / smoke test の追加
- docs 更新
- refactor
- artifact path の記録
- PR の初稿作成
- review コメントへの反映

### 3.3 Human-in-the-loop required

- issue が曖昧なままの着手
- broad refactor
- benchmark 正本の更新
- multi-issue をまたぐ設計変更
- 秘密情報、認証、権限、課金、外部公開まわり

## 4. Task Sizing Rule

自動 coding の成功率を上げる最重要ルールは、issue を小さくすることである。

- 1 issue = 1 measurable hypothesis
- 1 PR = 1 reviewable decision
- 1 agent run = 1 branch / 1 outcome

AI に渡してよい仕事:

- bug fix
- test coverage 改善
- documentation 更新
- separate module の refactor
- evaluation script の整理
- artifact manifest の整備

人が先に構造化すべき仕事:

- cross-cutting architecture change
- 曖昧な研究課題
- deep domain judgment が必要な betting strategy change
- incident response

## 5. The Default Autonomous Loop

標準ループは次で固定する。

1. issue を切る
2. acceptance criteria を埋める
3. AI が explore する
4. AI が plan を書く
5. 人が plan を承認する
6. AI が code / test / docs を更新する
7. AI が verify を実行する
8. PR に artifact と residual risk を残す
9. 人が merge / reject / iterate を決める

## 6. Requirements For AI-Ready Issues

AI に渡す issue には最低限次が必要である。

- objective
- hypothesis
- in-scope files or modules
- non-goals
- success metrics
- required validation
- stop condition

GitHub の公式ガイドどおり、issue は prompt として読まれる前提で書く。

## 7. Repository Harness

自動 coding の品質は、個別プロンプトより repo 側の harness に強く依存する。

`nr-learn` では次を repo harness とみなす。

- `.github/copilot-instructions.md`
- issue / PR template
- docs の strategy / standard
- scripts と tests
- artifact naming conventions
- benchmark / revision gate

原則:

- 判断規則は会話ではなく repo に書く
- 再発するレビュー指摘は instruction または test に昇格する
- AI の失敗は「モデルが弱い」ではなく「harness が不足している」とみなす
- 数秒超の source に progress を要求する規則も harness に含める

## 8. Evals For Coding Agents

OpenAI の eval best practices をこの repo に当てると、AI coding の評価は次の 4 層に分かれる。

1. syntax / unit tests
2. task acceptance criteria
3. traceability of changes and artifacts
4. domain outcome metrics

重要なのは、`vibe` ではなく比較可能な評価にすることである。

`nr-learn` では少なくとも次を残す。

- 実行コマンド
- 実行した tests
- 出力 artifact path
- promote / keep / reject の判断

## 9. Security And Control

自動 coding を前提にしても、次は固定で守る。

- default branch へ直接 push しない
- 人間 review なしで strategy change を採用しない
- secret や認証情報を issue / prompt に貼らない
- permissions は最小化する
- logs と commit trace を残す

## 10. Model Strategy

モデル選択は仕事の性質で切り替える。

- 深い設計、微妙な不具合、曖昧な探索は reasoning 強め
- 実装修正、テスト追加、docs 更新は高速モデル優先
- review は別モデルで second opinion を取ると安定しやすい

Inference:

これは GitHub Copilot CLI の推奨と OpenAI の eval / harness の考え方を、この repo の開発運用に当てはめた実務方針である。

## 11. `nr-learn` Default Policy

この repo では、今後の基本方針を次で固定する。

- 基本的に自動 coding を第一選択にする
- ただし issue が AI-ready であることを着手条件にする
- 形式知は docs / instructions / templates に蓄積する
- JRA の benchmark 正本更新は必ず human review を通す
- NAR は ingestion/readiness の自動化を優先する
- ML モデル開発は stage-based standard に従って進める
- 数秒超の処理を追加する AI 実装は progress なしで merge しない

## 12. Immediate Actions

今すぐ効く施策は次の順である。

1. repo instructions を拡張する
2. issue template を AI-ready 化する
3. ML モデル開発 standard を導入する
4. AI coding 用の review checklist を導入する
5. recurring cleanup issue を定例化する
6. coding eval dashboard を作る

運用上の注意は [development_operational_cautions.md](development_operational_cautions.md) を参照する。

## 13. Sources

- GitHub Copilot coding agent best practices
  - <https://docs.github.com/en/copilot/tutorials/coding-agent/get-the-best-results>
- GitHub Copilot CLI best practices
  - <https://docs.github.com/en/copilot/how-tos/copilot-cli/cli-best-practices>
- Responsible use of GitHub Copilot coding agent
  - <https://docs.github.com/en/copilot/responsible-use/copilot-coding-agent>
- OpenAI evaluation best practices
  - <https://developers.openai.com/api/docs/guides/evaluation-best-practices>
- OpenAI trace grading
  - <https://developers.openai.com/api/docs/guides/trace-grading>
- OpenAI harness engineering
  - <https://openai.com/index/harness-engineering/>
