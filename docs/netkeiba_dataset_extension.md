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