# ROI 1.20 KPI Definition

## 1. Purpose

この文書は、`nr-learn` において `ROI > 1.20` を狙うときの formal KPI 定義を固定するための正本である。

ここでの目的は、単に高い ROI を出すことではない。JRA 正本データで、再現可能な formal support を伴った長期運用候補を育てることである。

## 2. Scope

当面この定義は `JRA` の正式 benchmark と revision gate に適用する。

`NAR` と mixed-universe は、この KPI を参照してよいが、ready になるまでは正本判定には使わない。

## 3. North-Star Reading

`ROI > 1.20` は、単一の数字だけで達成判定しない。

この目標は次の 3 層が揃って初めて meaningful と読む。

1. formal forward ROI が高い
2. feasible fold support が十分にある
3. operational compare で catastrophic でない

## 4. KPI Hierarchy

優先順位は次で固定する。

### 4.1 Primary KPI

- formal forward ROI

### 4.2 Support KPI

- feasible fold coverage
- `stability_assessment=representative`

### 4.3 Guardrail KPI

- drawdown / bankroll stability
- bet volume sanity
- total net

### 4.4 Secondary KPI

- AUC
- calibration
- EV top1 ROI

secondary 指標は方向確認には使うが、primary と support を上書きしない。

## 5. Target Bands

`ROI > 1.20` の north star は、次の 3 段で読む。

### 5.1 Exploration Band

- formal forward ROI が baseline を明確に上回る
- support / guardrail はまだ候補段階

### 5.2 Candidate Band

- formal forward ROI が高い
- feasible folds と representative support が揃う
- operational compare で catastrophic regression がない

### 5.3 North-Star Band

- formal forward ROI が `1.20` 超を安定して示す
- support KPI が十分ある
- guardrail を壊していない
- actual-date role を説明できる

## 6. Minimum Conditions For Formal Success

`ROI > 1.20` を目指す experiment が formal success と呼べる最低条件は次のとおりである。

- `stability_assessment=representative`
- matching な walk-forward support がある
- feasible fold が不足していない
- drawdown / bankroll に catastrophic な悪化がない
- bet volume が実運用として薄すぎない
- baseline より良い理由を文章で説明できる

## 7. What Does Not Count As Success

次は success とみなさない。

- 短窓だけで `ROI > 1.20`
- bets が極端に少ない `ROI > 1.20`
- fold support が弱い `ROI > 1.20`
- drawdown を悪化させた `ROI > 1.20`
- actual-date compare で catastrophic loss を出す候補

## 8. Default Decision Rules

### Promote

- primary KPI が改善
- support KPI が基準を満たす
- guardrail KPI が許容内

### Keep As Candidate

- primary KPI は強い
- ただし support または operational role がまだ弱い

### Reject

- support 不足
- guardrail 悪化
- baseline を上回る理由が曖昧

## 9. Usage

以後、`ROI > 1.20` を目指す issue / PR / revision gate では、この文書を KPI 正本として参照する。

関連文書:

- [strategy_v2.md](strategy_v2.md)
- [ml_model_development_standard.md](ml_model_development_standard.md)
- [benchmarks.md](benchmarks.md)
- [evaluation_guide.md](evaluation_guide.md)
