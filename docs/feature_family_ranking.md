# Feature Family Ranking

## 1. Purpose

この文書は、既存 artifact から見える feature family の優先順位をまとめた初版 ranking である。

狙いは、新しい feature 実験を思いつき順ではなく evidence 順で進めることである。

## 2. Reading Rule

ここでの ranking は、次を総合して決める。

- formal evaluation の AUC / top1 ROI
- walk-forward support
- coverage の安定性
- 現行 baseline family への採用有無

単一の単発 run ではなく、現行 baseline に近い family を優先する。

## 3. Feature Families

この repo の主要 family は、ひとまず次の 7 つに分けて扱う。

1. base race context
2. recent form / history
3. class / rest / surface change
4. jockey / trainer / combo
5. gate / frame / course bucket bias
6. lineage / breeder pedigree
7. pace / corner / closing-fit

## 4. Current Ranking

### Tier A: Keep And Extend

#### 4.1 Class / Rest / Surface Change

優先度: 最上位

理由:

- `features_catboost_rich_high_coverage_diag.yaml` に残っている
- current serving 系でも使われている
- `horse_days_since_last_race`, `horse_weight_change`, `horse_distance_change`, `horse_surface_switch`, `race_class_score`, `horse_class_change` などは、high-coverage 側の中核を構成している

読み:

この family は baseline に近く、coverage も安定している。次の feature 改善はこの family の interaction や regime 条件を掘る価値が高い。

#### 4.2 Jockey / Trainer / Combo

優先度: 最上位

理由:

- `jockey_last_30_win_rate`, `trainer_last_30_win_rate`, `jockey_trainer_combo_last_50_win_rate`, `...avg_rank` が current strong family に残っている
- fundamental から rich / high-coverage まで一貫して採用されている

読み:

安定して残る family であり、feature importance と operational stability の両面で重要とみなせる。

#### 4.3 Gate / Frame / Course Bucket Bias

優先度: 高

理由:

- `gate_ratio`, `frame_ratio`, `course_gate_bucket_last_100_*` が high-coverage rich family に残っている
- serving 系の現行 family にも入っている

読み:

構造的に解釈しやすく、coverage も比較的安定している。course-conditioned bias の細分化余地がある。

### Tier B: Keep But Validate More

#### 4.4 Recent Form / History

優先度: 高

理由:

- すべての family で共通に残る
- `horse_last_3_avg_rank`, `horse_last_5_win_rate` は baseline 的な土台になっている

読み:

外す理由は薄いが、差別化の源泉としては飽和している可能性がある。interaction と decay 設計の改善余地を見るべきである。

#### 4.5 Base Race Context

優先度: 中

理由:

- track, weather, ground, race calendar などは広く使われている
- 単独での alpha というより他 family の条件付けに効いている

読み:

主役ではないが必須土台であり、削るより conditioning 用 feature として扱うのが妥当である。

### Tier C: Selective Or Conditional

#### 4.6 Owner Signal

優先度: 中

理由:

- `owner_last_50_win_rate` は high-coverage rich family に残っている
- ただし lineage 全体ほど heavy ではない

読み:

pedigree よりは実務的で、high coverage family に残っているため再評価価値がある。owner 単独と lineage の切り分けが次の論点になる。

### Tier D: Deprioritize Until Better Evidence

#### 4.7 Lineage / Breeder Pedigree

優先度: 低

理由:

- `rich` family では `breeder_last_50_win_rate`, `sire_last_100_*`, `damsire_last_100_win_rate`, `sire_track_distance_last_80_win_rate` が low coverage に出ている
- `rich_high_coverage_diag` ではこれらが意図的に外されている
- それでも `evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_diag_20240101_20240930_wf_full_nested.json` は `evaluation_summary_catboost_value_stack_20240101_20240930_wf_full_nested.json` より AUC / top1 ROI / support が改善している

読み:

現時点の evidence では、pedigree 系は current JRA baseline に対する first-priority alpha source ではない。coverage 改善や source quality が進むまでは主戦場にしない。

#### 4.8 Pace / Corner / Closing-Fit

優先度: 低

理由:

- `fundamental_enriched_no_lineage` と `rich` の両方で pace / corner 系が low coverage に出ている
- `rich_high_coverage_diag` では `horse_closing_pace_fit`, `horse_front_pace_fit`, `horse_closing_vs_course` を含めて外している
- それでも high-coverage side の評価は strong で、low-coverage pace family を無理に残す必要性が薄い

読み:

この family は「全否定」ではなく、「coverage と definition を改善するまで優先順位を下げる」が妥当である。

## 5. Key Evidence

### 5.1 Rich vs High-Coverage Rich

- `evaluation_summary_catboost_value_stack_20240101_20240930_wf_full_nested.json`
  - feature count `117`
  - selected low coverage features に lineage / pace-fit が多数残る
  - fold support が弱く `no_bet` fold が多い
- `evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_diag_20240101_20240930_wf_full_nested.json`
  - feature count `103`
  - selected low coverage features `0`
  - AUC `0.8423`
  - top1 ROI `0.7941`

読み:

低 coverage の lineage / pace-fit を抱えた rich family より、high-coverage rich family のほうが current path に合っている。

### 5.2 Fundamental Enriched No-Lineage

- `evaluation_summary_catboost_fundamental_enriched_no_lineage_win_20240101_20240930_wf_full_nested.json`
  - AUC `0.7184`
  - top1 ROI `0.7026`
  - pace / corner 系が low coverage

読み:

fundamental を広げる方向は悪くないが、current serving family の主力に比べるとまだ弱い。baseline replacement というより feeder experiment 扱いが妥当である。

## 6. Default Next Bets

次に試すべき feature issue は次の順を推奨する。

1. class / rest / surface interaction 強化
2. jockey-trainer-combo family の regime-aware 拡張
3. gate / frame / course bucket family の細分化
4. owner signal を pedigree から切り離した単独評価

## 7. Not Recommended Next

当面、次を primary な新規 feature issue にしない。

- pedigree feature の大量追加
- low coverage な pace-fit feature の追加
- current baseline に乗っていない lineage-heavy family への大きな投資

## 8. Operating Rule

新しい feature issue には、最低限次を入れる。

- どの feature family を触るか
- この ranking 上の tier
- baseline family に対して何を上積みしたいか
- coverage risk があるか
