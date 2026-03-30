# NAR Model Transfer Strategy

## 目的

NAR を JRA と同じ repository / 同じ evaluation discipline で扱えるようにしつつ、最初の導入では JRA と無理に混合しない。

現時点の基本方針は次の 2 行で固定する。

1. NAR は first step では `separate universe` として扱う。
2. JRA で有効だった `development standard / gating / artifact discipline` を NAR にそのまま移植する。

## 位置づけ

NAR データは将来 JRA と mixed-universe compare に進めてもよいが、初手は統合モデルではなく NAR 専用 baseline を作る。

理由は単純である。

- 競馬場構成、頭数分布、開催頻度、騎手・調教師の移動、オッズ形成が JRA とかなり異なる。
- 低い bet-rate や一部 window だけの上振れを、JRA 側の signal と混同しやすい。
- まず NAR 単独で `bets / races / bet-rate / feasible_folds` を読める状態にしたほうが、あとで universe 間比較をしやすい。

したがって、JRA から NAR へ移すべきものは「重み」よりも「進め方」である。

## そのまま移植するもの

- issue-driven execution
- `objective -> dataset freeze -> component train -> formal evaluation -> promotion decision` の stage
- nested walk-forward
- serving smoke / actual-date compare
- `min_bets_abs`, `min_bet_ratio`, `feasible_folds`, `drawdown`, `final_bankroll` を含む gate
- promoted と operational default を分ける考え方
- progress 必須、artifact first、same-summary equivalence の運用

## そのまま移植しないもの

- JRA で効いた feature family の優先順位そのもの
- JRA の policy threshold をそのまま使うこと
- JRA と NAR を早い段階で同一 train table に混ぜること
- JRA の low-frequency candidate を NAR で根拠なく再利用すること

## JRA から転用しやすい知見

- class / rest / surface は NAR でも重要候補になりやすい
- jockey / trainer / combo family は NAR のほうがむしろ効く可能性がある
- low bet-rate candidate は NAR でも過学習疑いを強く持つべき
- formal `pass / promote` と operational 採用は分けるべき
- actual-date compare で regime ごとの差を見るべき

## NAR 専用に再検証するもの

- race / bet opportunity の分母
- track / distance / season ごとの support 偏り
- jockey / trainer / stable 固着の強さ
- horse key / pedigree key の coverage
- odds quality と popularity / expected value の歪み

## First Wave

NAR の first wave は次の順で進める。

1. local Nankan primary dataset readiness を固定する。
2. NAR baseline model を current standard で formalize する。
3. NAR の bet denominator standard を明文化する。
4. JRA Tier A family を NAR 用に narrow replay する。
5. 最後にだけ JRA vs NAR の compare artifacts を作る。

## Success Criteria

NAR line の最初の成功条件は、JRA と同じで「高い一発 ROI」ではない。

最初に満たしたい条件は次である。

- NAR baseline が end-to-end で再現可能
- formal gate を回せる
- `bets / races / bet-rate` を常に読める
- low-support / low-frequency candidate を昇格させない
- promoted と operational default を分けて判断できる

## Current Reading

2026-03-30 時点では、local Nankan raw は揃っている。

- [data_local_nankan.yaml](/workspaces/nr-learn/configs/data_local_nankan.yaml)
- [model_local_baseline.yaml](/workspaces/nr-learn/configs/model_local_baseline.yaml)
- [local_nankan_primary.py](/workspaces/nr-learn/src/racing_ml/data/local_nankan_primary.py)

したがって、「NAR は後で考える」段階はもう過ぎている。正しい next step は、JRA の知見を参考にしながら、NAR を separate universe baseline として formalize することである。
