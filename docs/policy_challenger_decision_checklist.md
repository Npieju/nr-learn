# Policy Challenger Decision Checklist

## 1. Purpose

このチェックリストは、promoted anchor に対して challenger candidate を読むときの標準判定シートである。

`#5` のように同一 family 内で B/C を比較するとき、毎回同じ観点で読むために使う。

## 2. Required Inputs

- promoted anchor revision gate artifact
- promoted anchor promotion gate artifact
- promoted anchor evaluation summary
- challenger revision gate artifact
- challenger promotion gate artifact
- relevant evaluation summary / compare outputs

## 3. Primary Read

- [ ] challenger equivalence check が `equivalent / different / unavailable` のどれか確認した
- [ ] `equivalent` の場合は、full gate を続ける理由があるか確認した

- [ ] challenger の formal benchmark weighted ROI を確認した
- [ ] promoted anchor の formal benchmark weighted ROI と比較した
- [ ] `ROI > 1.20` north star との距離を確認した

## 4. Support Read

- [ ] feasible fold count を anchor と比較した
- [ ] `stability_assessment=representative` を確認した
- [ ] bets total / bets density を anchor と比較した

## 5. Guardrail Read

- [ ] drawdown が悪化していないか確認した
- [ ] bankroll が catastrophic でないか確認した
- [ ] support を増やすだけで exposure が荒れていないか確認した

## 6. Role Read

- [ ] challenger が anchor と違う役割を持つか確認した
- [ ] 役割差が `min_prob widening` なのか `odds widening` なのか説明できる
- [ ] actual-date compare で誤読しやすい broad replacement になっていない

## 7. Decision Rule

### Promote Over Anchor

- [ ] anchor より primary KPI が改善している
- [ ] support を維持または改善している
- [ ] guardrail を壊していない

### Keep As Secondary Candidate

- [ ] anchor は超えないが distinct role がある
- [ ] support または regime-fit に説明価値がある

### Reject

- [ ] anchor を上回る理由が弱い
- [ ] support / guardrail / role のどれかが悪い

## 8. Output Requirement

issue close 時には最低限次を残す。

- final decision: `promote / keep as candidate / reject`
- equivalence read: `equivalent / different / unavailable`
- anchor comparison sentence
- weighted ROI
- feasible folds
- bets total
- artifact paths
