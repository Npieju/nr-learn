# Next Issue: JRA Market Deviation Market-Aware Probability Path

## Summary

Stage 2 と Stage 3 により、JRA `market_deviation` track では race-normalized residual alpha が current best signal-source になった。

- `alpha_pred_corr=0.2232080870`
- `positive_signal_rate=0.2132800000`
- `ev_threshold_1_0_roi=1.1144751854`

一方、Stage 4 first pass の bounded sidecar reintegration は current read を次で閉じている。

- representative compare では calibration / auc はほぼ同等だったが、`top1_roi` と `ev_threshold_1_0_roi` は baseline をわずかに下回った
- September difficult window では `32 bets / -27.3 / 0.2959` に対して `35 bets / -4.7458333333 / 0.4352668690` で損失圧縮は見えた
- ただし support drift は残り、`alpha_weight: 0.05 -> 0.02` follow-up も `35 bets / -14.6 / 0.3314214738` で `reject` だった

meaning は明確である。

- signal-source は改善した
- しかし current late-fusion sidecar path は、その signal を probability layer の improvement として mainline compare できる形へ戻せていない

したがって next issue は policy tuning ではなく、「market signal を probability path へどう戻すか」という architecture / score-composition branch を 1 本に固定する。

## Objective

JRA mainline について、race-normalized residual alpha を post-hoc sidecar ではなく market-aware probability path の一部として扱う最小 candidate surface を 1 本定義し、late-fusion sidecar より構造的に良い compare surface を作れるかを判定する。

## Hypothesis

if race-normalized residual alpha を `compose_value_blend_probabilities` 上の後段 additive sidecar ではなく、baseline win probability と market information を受ける dedicated probability-path candidate に移す, then Stage 4 sidecar で残った support drift を増やさずに、representative compare と actual-date compare を同じ probability surface 上で読めるようになる。

## In Scope

- current late-fusion sidecar path の structural limitation の固定
- market-aware probability path の first candidate を 1 本だけ定義する
- compare で使う primary metrics と rollback boundary の固定
- policy family / serving default を動かさない evaluation-only candidate surface の用意

## Non-Goals

- Stage 4 bounded sidecar issue に別 parameter probe を追加すること
- baseline win model をこの issue 単体で置換すること
- broad policy rewrite や family search redesign を同時に入れること
- target redesign と probability path redesign を同時に再実装すること

## Why This Issue

current scoring path は次の形で late-fusion している。

1. `win_prob` を logit 化する
2. `alpha_raw`, `roi_raw`, `time_raw` を tanh 圧縮して加算する
3. 最後に `market_blend_weight` で market probability に anchor する

この構造は bounded compare には向くが、signal-source が良化しても final support / exposure の読みが後段 policy に吸われやすい。

特に current read では次が確認されている。

1. `alpha_weight` を下げても final `policy_bets` は減らなかった
2. representative top line の改善は weak で、actual-date の損失圧縮だけでは mainline promotion judgment を支えられなかった
3. `market_deviation` signal の責任範囲が probability path ではなく sidecar composition に残り、構造比較がしにくい

したがって next question は「どの parameter が良いか」ではなく、「market signal の入り口が late-fusion sidecar のままでよいのか」である。

## First Candidate Boundary

first candidate の boundary は次で固定する。

1. baseline win model artifact は固定する
2. race-normalized residual alpha artifact も固定する
3. candidate 差分は probability path のみとし、policy search surface と serving policy は baseline を維持する
4. compare は representative evaluate と actual-date compare の両方で行う
5. `advance / hold / reject` をこの issue 単体で閉じる

## Candidate Theme

first candidate theme は次で固定する。

- baseline win probability
- race-normalized residual alpha
- market probability

の 3 source を受ける dedicated probability-path candidate を 1 本作り、current late-fusion sidecar と同じ compare surface で読む。

ここで重要なのは、candidate を「policy tweak」としてではなく「probability path の別構造」として扱うことである。

## Primary Metrics

1. representative compare での `auc` / `logloss` / EV threshold read
2. actual-date compare での `total net` と bankroll deterioration の抑制
3. support drift を伴わない `policy_bets` / selection rate の維持

guardrail:

- candidate の改善が actual-date だけに偏り、representative support を壊した場合は採用しない

## Current Baseline References

- operational baseline profile:
  - `current_recommended_serving_2025_latest`
- Stage 3 signal-source candidate:
  - `r20260418_jra_lightgbm_alpha_race_norm_v1`
- Stage 4 bounded reintegration issue:
  - [next_issue_jra_market_deviation_policy_reintegration.md](next_issue_jra_market_deviation_policy_reintegration.md)
- current score composition implementation:
  - [../../src/racing_ml/evaluation/scoring.py](../../src/racing_ml/evaluation/scoring.py)

## Success Metrics

- probability-path candidate を 1 本だけ定義できる
- compare rule が policy issue ではなく probability issue として明文化される
- representative / actual-date の両 read で `advance / hold / reject` を閉じられる
- Stage 4 sidecar issue に戻らずに next action を決められる

## Validation Plan

1. current late-fusion sidecar path の compare rule を固定する
2. candidate path を 1 本だけ追加する
3. representative evaluate を実行し、baseline / sidecar / new probability-path candidate を compare する
4. actual-date compare を実行し、support drift と損失圧縮を読む
5. `advance / hold / reject` のいずれかで閉じる

## Stop Condition

- probability path の候補定義に policy family rewrite が必要になる
- target redesign と probability path redesign を同時にやらないと仮説が書けない
- representative compare を壊してまで actual-date の局所改善しか取れない

## Exit Rule

- `advance`: market-aware probability path が late-fusion sidecar より良い compare surface を作る
- `hold`: signal はあるが probability-path replacement を主張するには弱い
- `reject`: probability path ではなく prediction foundation か broader architecture branch へ戻る

## Follow-Up Boundary

- この issue が終わるまで、Stage 4 bounded reintegration issue に新しい single-lever probe は追加しない
- execution policy track に戻るのは、この issue が `advance` で閉じた後だけにする