# Strategy V2

## 1. Positioning

この文書は、`nr-learn` の新しい運用基準である。

`docs/` 全体を広く読む前に、まず [README.md](README.md) を入口にして current source-of-truth だけを辿る。tagged snapshot/reference や大きい補助資料は必要になった時だけ開く。

## 2. North Star

このプロジェクトの第 1 目標は、JRA データを使った長期運用モデルで `ROI > 1.20` を狙える再現可能な開発体系を作ることである。

ここでいう `ROI > 1.20` は、単発の短窓当たりではなく、時系列を守った formal evaluation と operational simulation を通した長期運用前提の目標を指す。

## 3. Universe Policy

- 当面の主戦場は `JRA` とする。
- モデル改善、特徴量改善、policy 改善、gate 判定は、原則として取得済みの JRA データセットを正本に進める。
- `NAR` は並行トラックでデータ取得と readiness 整備を進める。
- NAR は readiness gate を通すまでは benchmark の正本に混ぜない。
- mixed-universe の比較は exploratory 扱いとし、主 KPI の判定根拠には使わない。

## 4. Core Success Metrics

優先順位は次の順に固定する。

1. formal forward ROI
2. feasible fold coverage
3. drawdown / bankroll stability
4. bet volume and liquidity sanity
5. predictive quality such as AUC / calibration

補助指標が良くても、`ROI` と `stability` を壊す変更は昇格しない。

`ROI > 1.20` の formal KPI 定義は [roi120_kpi_definition.md](roi120_kpi_definition.md) を正本とする。

## 5. Operating Principles

- `1 issue = 1 measurable hypothesis` を基本にする。
- 短窓 smoke と正式 gate を混同しない。
- 変更前に acceptance criteria と eval plan を決める。
- AI 支援は歓迎するが、昇格は artifact ベースで行う。
- 大きな改善は epic から sub-issue へ分解して小さく流す。
- 長命ブランチを避け、短命 branch / PR を前提にする。

## 6. Standard Workstreams

### 6.1 JRA Alpha

- baseline 改善
- feature store 改善
- model ensemble 改善
- betting policy / bankroll policy 改善

### 6.2 Evaluation Reliability

- walk-forward robustness
- leakage 防止
- benchmark refresh
- replay / actual-date compare の自動化

### 6.3 Data Platform

- JRA dataset quality checks
- schema drift 検知
- source coverage monitoring
- artifact manifest 標準化

### 6.4 NAR Readiness

- crawl completeness
- schema normalization
- JRA との feature parity 点検
- standalone benchmark 準備

## 7. Promotion Rule

新候補を採用候補として扱うには、少なくとも次を満たす。

- JRA formal evaluation が再現可能である
- `stability_assessment` 相当の基準を満たす
- feasible folds が不足していない
- operational compare で catastrophic な悪化がない
- 出力 artifact と issue / PR の紐付けがある

## 8. What We Standardize First

まず標準化するのは次の 4 つである。

1. issue taxonomy
2. issue-to-branch-to-PR の流れ
3. experiment / revision の definition of done
4. AI agent を含む自動開発の acceptance contract

詳細は [autonomous_dev_standard.md](autonomous_dev_standard.md) を正本とする。

ML モデル開発の標準手順は [ml_model_development_standard.md](ml_model_development_standard.md) を正本とする。

## 9. External References

この運用基準は、次の公開ガイドの考え方を参考にしている。

- GitHub Projects automation and planning
  - <https://docs.github.com/issues/planning-and-tracking-with-projects/automating-your-project>
- OpenAI evaluation best practices
  - <https://developers.openai.com/api/docs/guides/evaluation-best-practices>
- Trunk-Based Development
  - <https://trunkbaseddevelopment.com/short-lived-feature-branches/>
