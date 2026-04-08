# Next Issue: NAR JRA-Equivalent Trust Completion Gate

## Summary

NAR side の個別 corrective は進んでいるが、`解決した` の意味が top-level で明文化されていなかった。

ここでの完了条件は明示的に次とする。

- NAR が解決したと言えるのは、`JRA相当の信頼度で運用判断できるモデル line` が構築されたときだけである
- local Nankan future-only readiness は、その completion に至るための Stage 0 blocker resolution であり、完了そのものではない

したがって次の重大 issue は、個別 readiness task の upstream に `NAR completion gate` を固定し、以後の NAR issue がこの gate に対してどの段を解いているのかを明文化することである。

## Objective

NAR の top-level 完了条件を `JRA相当の信頼度モデル構築` として固定し、current open issue (`#101`, `#103`, `#122` など) をその completion gate に対する blocker / prerequisite として再配置する。

## Hypothesis

if NAR の completion を `JRA相当の信頼度` として docs / queue / issue thread で明示し、readiness issue と completion issue を分離する, then operators will stop reading temporary readiness progress as solution completion and will advance NAR in the same gate discipline used for JRA.

## In Scope

- NAR completion definition の明文化
- `JRA相当の信頼度` を構成する trust surface の明文化
- `#101`, `#103`, `#122` の role を completion gate に対して再配置
- `result arrival` / `result-ready support` の用語定義
- queue / issue thread / ladder の同期

## Non-Goals

- いまこの issue だけで NAR model quality を引き上げること
- readiness blocker を completion と言い換えること
- local Nankan only line を full NAR representative と主張すること
- artifact のない broad success claim

## Completion Criteria

この重大 issue 自体の close 条件は次である。

1. docs 上で `NAR solved = JRA-equivalent trust model constructed` が明文化されている
2. NAR の current queue が readiness / architecture / parity / serving を completion gate に紐づけて読める
3. `result arrival` が何を意味するかが readiness docs で曖昧でない
4. GitHub issue thread に decision summary が残っている

NAR 全体の実質完了条件は次である。

1. strict `pre_race_only` result-ready benchmark が完走し、JRA と同じ formal read discipline で評価できる
2. architecture parity line (`#103`) が strict benchmark 上で end-to-end 実行される
3. feature / policy / serving が JRA と同じ issue ladder と gate で比較可能になる
4. promoted / hold / reject の運用判断を JRA と同水準の artifact で説明できる

## Validation Plan

1. parity ladder に top-level completion definition を追記する
2. future-only readiness doc に `result arrival` の定義を追記する
3. current queue に `#122` は completion ではなく blocker resolution と書く
4. GitHub に重大 issue を起票し、以後の NAR issue をこの gate に従属させる

## Stop Condition

- docs / queue / issue を同期しても completion definition がなお曖昧に読める
- その場合は wording をさらに narrowed に修正し、`trust`, `readiness`, `completion` を混ぜない

## Current Reading

- current NAR は未解決である
- historical local Nankan line は diagnostic only である
- current executable path は future-only readiness track である
- `result arrival` とは、future-only strict `pre_race_only` subset に対応する race result が実データへ反映され、artifact 上で `result_ready_races>0` になることを指す
- したがって current issue `#122` は completion issue ではなく `Stage 0 benchmark trust readiness` issue である
