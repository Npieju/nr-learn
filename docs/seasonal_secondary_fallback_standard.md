# Seasonal Secondary Fallback Standard

## 1. Purpose

この文書は、`current_long_horizon_serving_2025_latest` を first seasonal alias とした後の、secondary fallback ordering を固定するための正本である。

## 2. Current Ordering

2026-03-29 時点の seasonal fallback ordering は次で固定する。

1. first seasonal alias
   - `current_long_horizon_serving_2025_latest`
2. second seasonal fallback
   - `current_sep_guard_candidate`
3. third defensive option
   - `current_tighter_policy_search_candidate_2025_latest`
4. analysis-first retrain fallback
   - `current_recommended_serving_2025_recent_2018`

## 3. Why Sep Guard Is Second

`current_sep_guard_candidate` は次の性質を持つ。

- September 10 日 window で baseline `-21.6` に対して `+12.7067`
- May-Sep 45 日でも baseline `-42.2080` に対して `-7.9014`
- `differing_policy_dates` は September 10 日に局在し、non-September baseline を広く壊していない

したがって、September sparse-selection regime に対する secondary fallback としては十分に強い。

一方で first alias にしない理由も明確である。

- long-horizon は latest 2025 actual-date compare で September `-27.3 -> -4.3`、December `+14.9 -> +14.9` と、より単純な controlled override として読める
- sep-guard family は selected-rows / fallback path の説明が長く、運用 alias としては一段重い

## 4. Why Tighter Policy Stays Third

`current_tighter_policy_search_candidate_2025_latest` は formal support が強いが、seasonal serving ordering では third に留める。

理由:

- current reading の主軸は serving-side controlled override である
- tighter policy は analysis-first defensive candidate としての性質が強い
- seasonal runtime alias としては long-horizon / sep-guard のほうが説明が単純

## 5. Family-Level Rule

selected-rows September guard family は次の 2 層で扱う。

- stable secondary line
  - `current_sep_guard_candidate`
- analysis-only family variants
  - count-only selected-rows family
  - ev-only / lower-blend reinsert variants
  - other selected-rows probes

つまり、family 内 frontier は `current_sep_guard_candidate` で代表させ、その他は reference に留める。

## 6. Standard Conclusion Template

issue / docs には次の文面を使う。

```md
Conclusion
- first seasonal alias remains `current_long_horizon_serving_2025_latest`
- second seasonal fallback remains `current_sep_guard_candidate`
- `current_tighter_policy_search_candidate_2025_latest` stays third defensive option
- other selected-rows September guard variants remain analysis-only
```
