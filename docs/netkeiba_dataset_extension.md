# netkeiba データ拡張設計

最終更新: 2026-03-09

## 1. 目的
- netkeiba のような外部ソースを後から追加しても、loader 本体を毎回書き換えない。
- 行追加と列追加を config に分離し、将来の crawler 実装と学習系を疎結合に保つ。
- 公開 benchmark と比較するときに不足している pre-race 情報を段階的に増やせるようにする。

## 2. 基本方針
- 主表は引き続き Kaggle 側の JRA データを使う。
- 外部ソースは 2 種類に分ける。
  - `append_tables`: 履歴行を足す。主に過去レース結果の増補用。
  - `supplemental_tables`: 主表に列を足す。主に racecard、pedigree、owner / breeder などの付帯情報用。
- 列名差分は table ごとの `column_aliases` で吸収する。

## 3. 先に集めるべき表
- race_result
  - 目的: Kaggle 主表にない履歴や欠損年を埋める。
  - merge 方針: `append_tables`
  - 主キー: `race_id + horse_id`
- race_card
  - 目的: 出走時点で利用可能な owner / breeder / 枠番 / 馬番を足す。
  - merge 方針: `supplemental_tables`
  - 主キー: `race_id + horse_id`
- pedigree
  - 目的: 血統系特徴量の基礎を追加する。
  - merge 方針: `supplemental_tables`
  - 主キー: `horse_id`

## 4. 実装済み受け皿
- [src/racing_ml/data/dataset_loader.py](/workspaces/nr-learn/src/racing_ml/data/dataset_loader.py)
  - `append_tables`
  - `supplemental_tables`
  - table 単位の `column_aliases`
- [src/racing_ml/features/builder.py](/workspaces/nr-learn/src/racing_ml/features/builder.py)
  - `gate_ratio`
  - `frame_ratio`
  - `owner_last_50_win_rate`
  - `breeder_last_50_win_rate`
  - `sire_last_100_win_rate`
  - `sire_last_100_avg_rank`
  - `damsire_last_100_win_rate`
  - `sire_track_distance_last_80_win_rate`
- テンプレート config: [configs/data_netkeiba_template.yaml](/workspaces/nr-learn/configs/data_netkeiba_template.yaml)

## 5. column_aliases の書き方
- key は最終的に揃えたい canonical 名。
- value は source 側に現れうる列名候補の配列。
- 例:

```yaml
column_aliases:
  race_id:
    - "race_key"
    - "レースID"
  horse_id:
    - "horse_key"
    - "出走馬ID"
  sire_name:
    - "父"
```

## 6. 次段階
- crawler 実装前に、保存先 CSV の列名をこの template に合わせる。
- pedigree や owner のような高カーディナリティ列は、まず raw のまま保持し、採用は feature selection 側で制御する。
- 追加データ導入後の採用判定は raw ROI ではなく `benter_delta_pseudo_r2` を優先する。

## 7. 運用チェック
- validation CLI: [scripts/run_validate_data_sources.py](../scripts/run_validate_data_sources.py)
- feature gap CLI: [scripts/run_feature_gap_report.py](../scripts/run_feature_gap_report.py)
- 出力: `artifacts/reports/data_source_validation.json`
- 出力: `artifacts/reports/feature_gap_summary_<feature_config>.json`
- train / evaluate artifact 内の `feature_coverage`
- 確認内容:
  - primary dataset が存在するか
  - append / supplemental table が見つかるか
  - required columns / join keys が足りているか
  - dedupe key 上の重複がどの程度あるか
  - netkeiba template 上の canonical raw columns が今の主表でどこまで埋まっているか
  - benchmark feature profile の force include 特徴が missing / empty / low coverage になっていないか

## 8. 直近の gap 結果
- `features_catboost_fundamental_enriched` の gap report では、優先 missing raw columns は `breeder_name`, `sire_name`, `dam_name`, `damsire_name` だった。
- つまり次に効果が見込める外部ソースは pedigree / breeder 系で、owner や gate/frame はすでに主表だけでも利用可能。
- low coverage 側では `horse_last_3_avg_corner_2_position`, `horse_last_3_avg_corner_2_ratio`, `horse_last_3_avg_corner_gain_2_to_4` が残っており、corner 前半列の品質改善も有効候補。
- 既存 JRA raw の `corner_passing_order.csv` を supplemental として有効化しても coverage 改善は小さく、benchmark 指標はほぼ不変だったため、次の投資先は pedigree 系が優先。