# ML Model Development Standard

## 1. Purpose

この文書は、`nr-learn` における ML モデル開発プロセスの正本である。

狙いは、誰が進めても、あるいは AI が自動で進めても、同じ順番と同じ判断軸でモデル改善を進められるようにすることにある。

## 2. Scope

この標準は次に適用する。

- JRA の feature / model / policy 改善
- formal evaluation 候補の作成
- benchmark 正本の更新判断
- NAR readiness 後の benchmark 化準備

当面の主戦場は `JRA` であり、この文書も JRA を正本として読む。

## 3. Core Principle

モデル開発は、次の 5 レイヤを混ぜずに進める。

1. problem definition
2. dataset definition
3. feature change
4. model / ensemble change
5. betting policy / bankroll change

1 issue で複数レイヤを同時に大きく動かさない。大きい施策は layer ごとに分割する。

## 4. Standard Stage Flow

標準ステージは次の 8 段で固定する。

1. objective 定義
2. dataset freeze
3. baseline artifact 確認
4. hypothesis 設計
5. implementation
6. smoke validation
7. formal evaluation
8. promotion decision

## 5. Stage Details

### 5.1 Objective Definition

最初に次を固定する。

- 何を改善したいか
- 何は今回やらないか
- primary metric は何か
- 許容できる副作用は何か

`nr-learn` の primary metric は原則として `formal forward ROI` である。

`ROI > 1.20` を狙う issue では、[roi120_kpi_definition.md](roi120_kpi_definition.md) を KPI 正本として参照する。

## 5.2 Dataset Freeze

モデル開発では、先に dataset scope を凍結する。

- universe: JRA / NAR / mixed
- train window
- validation window
- holdout / latest split
- data source version

途中で dataset scope を変えたら、同じ experiment とみなさず issue を分ける。

## 5.3 Baseline Artifact Confirmation

変更前に、比較対象の baseline artifact を固定する。

- current profile
- current revision
- latest evaluation summary
- latest promotion gate
- relevant compare dashboard

baseline を曖昧にしたまま改善判定しない。

## 5.4 Hypothesis Design

仮説は次の型で書く。

`if <change>, then <metric> improves, while <guardrails> stay within limits`

例:

`if liquidity guardrail を tightening する, then formal ROI と drawdown が改善し, bets は許容範囲内に残る`

## 5.5 Implementation

実装では、次の 4 区分を明示する。

- feature
- model
- ensemble
- policy

この区分が曖昧だと、後で効果の所在が追えなくなる。

加えて、数秒で終わらない training / evaluation / gate / wrapper 系 source には progress を入れる。operator が実行中と停止を区別できない source は標準未満とみなす。

## 5.6 Smoke Validation

smoke は方向確認用であり、採用根拠には使わない。

smoke で確認するもの:

- code が壊れていない
- output schema が保たれている
- bets / net / bankroll が極端に壊れていない
- feature leakage や obvious bug がない

smoke が良くても formal gate へ進める価値がある、という意味にしかならない。

## 5.7 Formal Evaluation

formal evaluation では少なくとも次を揃える。

- representative evaluation
- nested or walk-forward metrics
- feasible fold coverage
- drawdown / bankroll checks
- actual-date compare または replay compare

短窓 ROI の単発上振れだけで昇格しない。

## 5.8 Promotion Decision

最終判断は次の 3 択で固定する。

- `promote`
- `keep as candidate`
- `reject`

判断時には必ず理由を残す。

## 6. Required Deliverables Per Experiment

1 experiment ごとに最低限必要な成果物は次のとおりである。

- issue
- code or config diff
- validation log or command list
- output artifact paths
- decision summary

## 7. Separation Of Change Types

変更種別ごとの基本ルールは次のとおりである。

### 7.1 Feature Change

- 新特徴量の因果方向を説明する
- 利用時点で参照可能な情報だけを使う
- 既存特徴量との重複を確認する

### 7.2 Model Change

- 学習対象と目的関数を明示する
- baseline より複雑になる理由を説明する
- 追加複雑性に見合う利益があるかを確認する

### 7.3 Ensemble Change

- 各 component の役割を明示する
- component 単体より何が改善するかを書く
- attribution が追えない複雑化を避ける

### 7.4 Policy Change

- exposure をどう変えるかを書く
- ROI だけでなく drawdown と bet volume を併記する
- regime 依存なら regime を明示する

## 8. Standard Decision Gates

各段階での gate は次のとおりである。

### Gate A: Ready To Implement

- objective が明確
- dataset freeze 済み
- baseline artifact が固定済み
- hypothesis が 1 文で書ける

### Gate B: Ready For Formal Eval

- code / config / docs が揃っている
- smoke を通過している
- catastrophic regression が見えていない
- long-running step に progress / heartbeat / completion がある

### Gate C: Ready To Promote

- formal metrics が揃っている
- representative 判定を満たす
- feasible folds が許容範囲
- compare 上の catastrophic loss がない

## 9. Standard Stop Conditions

次のいずれかで、その issue は打ち切る。

- baseline 比で primary metric が明確に悪化
- drawdown / bankroll が許容外
- bets が実運用にならない水準まで減る
- leakage or data bug の疑いが出る
- 変更理由を人間が説明できない

## 10. Documentation Update Rule

formal judgment が出たら、必要に応じて次を更新する。

- `docs/roadmap.md`
- `docs/benchmarks.md`
- `docs/public_benchmark_snapshot.md`
- issue / PR の decision summary

exploratory memo は `docs/` に増やさず issue / artifact に残す。

## 11. Default Weekly Operating Rhythm

- early week: issue slicing と baseline 棚卸し
- mid week: feature / model / policy experiment
- late week: formal gate 候補だけ評価
- week close: benchmark / roadmap 更新判断

## 12. How AI Should Use This Standard

AI は次の順で作業する。

1. issue を読む
2. stage がどこか判定する
3. dataset freeze と baseline artifact を確認する
4. change type を 1 つ選ぶ
5. implement する
6. smoke / formal validation を実行する
7. promote / candidate / reject の材料を残す

つまり AI は、思いつきでコードを書くのではなく、必ず stage-based workflow に従う。
