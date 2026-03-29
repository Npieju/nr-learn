# Tail Loader Equivalence Gate Standard

## 1. Purpose

この文書は、tail loader の aggressive optimization を比較するときに、`exact` / `canonical` / `value` のどの gate をどう使うかを固定する標準である。

`#19` で harness 自体は整備できたため、以後は「どこまで同値なら次へ進めるか」をこの文書で判断する。

## 2. Gate Modes

`scripts/run_tail_loader_equivalence.py` の `--fail-gate` は次の 3 種で使い分ける。

### 2.1 `exact`

- raw frame が完全一致のときだけ pass
- `comparison.raw.exact_equal=true` が必要
- formal promotion-ready と見なせるのはこの mode を通った候補だけ

### 2.2 `canonical`

- raw frame が完全一致、または canonical dtype drift のみなら pass
- `comparison.raw.exact_equal=true` または `comparison.raw.canonical_dtype_only_difference=true`
- loader optimization の exploration / profiling / follow-up design に使ってよい
- ただしこれだけでは default replacement の根拠にしない

### 2.3 `value`

- 値差分がなければ dtype drift を問わず pass
- `comparison.raw.value_equal=true`
- 診断や drift 読みには使ってよい
- operator-facing な acceptance gate には使わない

## 3. Canonical Dtype Drift

現時点で harness が canonical と分類するのは次である。

- `all_null`
  - 両 side とも非欠損値が存在しない列
- `numeric_integral_equivalent`
  - 値としては同じ整数列で、`int64` / `float64` のような dtype 差だけがある列
- `numeric_dtype_only`
  - 数値値自体は同じで、dtype だけが異なる列

この分類は「値差分がない」ことを前提にしている。value drift がある場合は canonical 扱いしない。

## 4. Decision Rule

tail loader candidate の採否は次の順で決める。

1. まず `exact` で比較する。
2. `exact` が fail なら `canonical` と dtype category を確認する。
3. `canonical` まで pass でも、その候補は analysis-first として扱う。
4. default path へ昇格させる前には、少なくとも次のどちらかが必要である。
   - exact equality
   - 明示的に「この canonical dtype drift は production-safe」と docs / tests / issue で承認されていること

## 5. Current Read For `deque_trim`

2026-03-29 時点の `current` vs `deque_trim` は次のとおりである。

- `exact`: fail
- `canonical`: pass
- `value`: pass

現在の dtype drift は、`レース記号/*` の `all_null` と `斤量` の `numeric_integral_equivalent` に限られる。

つまり `deque_trim` は「value-stable かつ canonical-dtype-only」な candidate だが、まだ exact-equivalent ではない。したがって現時点では optimization candidate であり、default replacement ではない。

## 6. Standard Operator Flow

tail optimization を試すときは次の順で読む。

1. `scripts/run_tail_loader_equivalence.py --fail-gate exact`
2. fail したら manifest の `comparison.raw` と `comparison.normalized`
3. 次に `--fail-gate canonical` で pass/fail を確認する
4. canonical pass の場合だけ downstream smoke / summary compare に進む
5. default replacement 判断の前に、この文書の rule に照らして exact 必須か、canonical 承認で足りるかを明示する

## 7. Current Default

現在の default は次のとおりである。

- promotion-ready gate: `exact`
- analysis / optimization exploration gate: `canonical`
- diagnostic-only gate: `value`

この default を変えるときは、`docs/command_reference.md`、この文書、関連 issue を同時に更新する。
