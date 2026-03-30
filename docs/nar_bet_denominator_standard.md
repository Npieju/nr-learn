# NAR Bet Denominator Standard

## 目的

NAR evaluation を読むときは、`bets` の絶対数だけで判断しない。

最初に読むべきなのは、必ず次の分母つき指標である。

1. `bets / races`
2. `bets / rows`
3. `bets / feasible_races`

NAR は JRA よりも開催構成や race shape の偏りが強い可能性があるため、low bet-rate の候補を高 ROI だけで誤読しやすい。したがって NAR line では、bet denominator を formal read の必須項目として固定する。

## 必須表示

NAR の formal read、actual-date compare、serving compare では少なくとも次を明示する。

- `policy_bets`
- `race_count`
- `row_count`
- `feasible_race_count`
- `bet_rate_races = policy_bets / race_count`
- `bet_rate_rows = policy_bets / row_count`
- `bet_rate_feasible_races = policy_bets / feasible_race_count`

可能なら追加で次も残す。

- `zero_bet_dates / total_dates`
- `feasible_folds / total_folds`
- `bets_by_fold`
- `bets_by_date`

## 読み方

### 1. `bets / races`

最も基本の分母である。

- 実際に何レース中どれだけ bet が出たかを見る。
- low-frequency candidate の過学習疑いを最初に見る入口にする。

### 2. `bets / rows`

頭数差や scratch を含む row volume に対する露出を見る。

- 同じ race count でも row density が違うと、bet の sparse さが見えにくい。
- NAR では field size や card shape の差が出やすいので補助指標として残す。

### 3. `bets / feasible_races`

policy gate を通過した race 母集団の中で、どれだけ bet が出たかを見る。

- support が弱いのか
- そもそも feasible race 自体が少ないのか

を切り分けるために使う。

## Warning 条件

次のどれかに当たる場合は、`pass / promote` でも operational 採用を保留する。

- `bet_rate_races` が著しく低い
- `zero_bet_dates` が多い
- `feasible_folds` がぎりぎり
- top line の大半が少数日・少数 race に集中している

ここで重要なのは、「損失が小さい」ことと「良い operational candidate」であることを混同しないことである。NAR でも、bet しなくなっただけの suppressive path は improvement と見なさない。

## Standard Conclusion Template

NAR の issue / artifact read では、次の形で結論を書く。

1. `policy_bets` と 3 つの denominator を書く。
2. `weighted ROI`, `total net`, `feasible_folds` を書く。
3. low bet-rate / concentration の懸念があるかを書く。
4. `promote / hold / reject` ではなく、必要なら `formal promoted but operational hold` まで分けて書く。

## Current Reading

2026-03-30 時点の `#52` では、local Nankan は feasibility-only ではなく baseline formalization に進める段階に入っている。

したがって次の NAR baseline run からは、`bets / races / bet-rate` を optional note ではなく first-class output として扱う。
