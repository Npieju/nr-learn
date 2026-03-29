# Revision Gate Candidate Checklist

## 1. Purpose

このチェックリストは、experiment candidate を formal revision gate に載せる前の標準確認項目である。

`run_revision_gate.py` を回す前に、最低限これを満たしていることを確認する。

## 2. Candidate Readiness

- [ ] issue が存在する
- [ ] objective と hypothesis が明記されている
- [ ] universe が JRA / NAR / mixed のどれかで固定されている
- [ ] dataset freeze が済んでいる
- [ ] baseline profile / revision / artifact が明記されている

## 3. Artifact Readiness

- [ ] versioned config がある
- [ ] feature config が固定されている
- [ ] compare に使う baseline artifact が固定されている
- [ ] latest pointer ではなく versioned artifact を主参照にしている

## 4. Evaluation Readiness

- [ ] smoke で catastrophic regression が見えていない
- [ ] expected primary KPI が書かれている
- [ ] support KPI が書かれている
- [ ] drawdown / bankroll / bet volume の guardrail が書かれている

## 5. Revision Gate Command Readiness

- [ ] revision ID が決まっている
- [ ] `--dry-run` command を先に確認した
- [ ] full 実行 command が issue に残っている
- [ ] train を skip するなら reused artifact suffix が明記されている

## 6. Decision Readiness

- [ ] promote の条件が書かれている
- [ ] keep as candidate の条件が書かれている
- [ ] reject の条件が書かれている
- [ ] public snapshot / benchmark 更新要否が判定できるようになっている

## 7. Do Not Gate Yet If

- [ ] issue がまだ曖昧
- [ ] baseline が固定されていない
- [ ] compare role が不明
- [ ] smoke の時点で drawdown が壊れている
- [ ] 何をもって success とするか説明できない

いずれかに該当するなら、revision gate に上げる前に issue を分割する。
