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
  - merge 方針: `append_tables` + `supplemental_tables`
  - 主キー: `race_id + horse_id`
- race_card
  - 目的: 出走時点で利用可能な `horse_key` / 枠番 / 馬番 / 性齢 / 斤量 / 騎手 / 調教師を足す。
  - merge 方針: `supplemental_tables`
  - 主キー: `race_id + horse_id`
- 現行の shutuba page では owner / breeder は直接出ないため、これらは引き続き horse profile / pedigree 側から補う。
- pedigree
  - 目的: 血統系特徴量の基礎を追加する。
  - merge 方針: `supplemental_tables`
  - 主キー: `horse_key`

## 3.1 `horse_id` と `horse_key` の分離
- 既存 Kaggle 主表の `horse_id` は row-level join key として扱う。
- stable な競走馬 ID は `horse_key` として別列で保持する。
- netkeiba crawler の `race_result` は `horse_id = race_id + 馬番(2桁)` を生成し、同時に馬リンクから `horse_key` を抽出する。
- まず `race_id + horse_id` で主表へ `horse_key` を補完し、その後 `horse_key` で pedigree を merge する。

## 4. 実装済み受け皿
- [src/racing_ml/data/dataset_loader.py](/workspaces/nr-learn/src/racing_ml/data/dataset_loader.py)
  - `append_tables`
  - `supplemental_tables`
  - table 単位の `column_aliases`
  - `horse_key` canonical column
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
    - "出走馬ID"
  horse_key:
    - "horse_key"
    - "競走馬ID"
  sire_name:
    - "父"
```

## 6. 次段階
- crawler 出力を template に合わせる。
- `race_result` は append 用だけでなく、`horse_key` 補完用の supplemental merge にも使う。
- pedigree や owner のような高カーディナリティ列は、まず raw のまま保持し、採用は feature selection 側で制御する。
- 追加データ導入後の採用判定は raw ROI ではなく `benter_delta_pseudo_r2` を優先する。

## 6.1 crawler 初期実装
- 設定テンプレート: [configs/crawl_netkeiba_template.yaml](/workspaces/nr-learn/configs/crawl_netkeiba_template.yaml)
- ID 生成 CLI: [scripts/run_prepare_netkeiba_ids.py](/workspaces/nr-learn/scripts/run_prepare_netkeiba_ids.py)
- 実行 CLI: [scripts/run_collect_netkeiba.py](/workspaces/nr-learn/scripts/run_collect_netkeiba.py)
- 年単位 backfill CLI: [scripts/run_backfill_netkeiba.py](/workspaces/nr-learn/scripts/run_backfill_netkeiba.py)
- 初期 target:
  - `race_result`: race page を収集し、`horse_id`, `horse_key`, `owner_name` を含む canonical CSV を出す。
  - `race_card`: shutuba page を収集し、`horse_id`, `horse_key`, `frame_no`, `gate_no`, `sex`, `age`, `weight`, `jockey_id`, `trainer_id` を含む canonical CSV を出す。
  - `pedigree`: horse top page と AJAX pedigree endpoint を収集し、`horse_key`, `sire_name`, `dam_name`, `damsire_name`, `owner_name`, `breeder_name` を出す。
- raw HTML は `data/external/netkeiba/raw_html/` に保存し、既存 cache があれば再 fetch せず再利用する。
- canonical CSV は batch ごとに累積更新し、同一 dedupe key の重複行は最新 batch を優先する。
- `run_prepare_netkeiba_ids.py` は主表から `race_ids.csv` を作り、主表と既存の `race_result` / `race_card` 出力から `horse_keys.csv` を作る。既に出力済みの ID はデフォルトでは除外する。
- `run_prepare_netkeiba_ids.py` / `run_backfill_netkeiba.py` は `--date-order desc` で最新レース優先に切り替えられる。benchmark 改善の確認を急ぐときは descending を使う。
- default の [configs/data.yaml](/workspaces/nr-learn/configs/data.yaml) では `netkeiba_race_card` を `optional: true` で組み込み済みとし、まだ CSV が無い段階では validation 上 `optional_missing` として扱う。
- `run_backfill_netkeiba.py` は training table を 1 回だけ読み込み、指定期間で pending race IDs と horse_keys を batch 単位に自動生成しながら `race_result` / `race_card` / `pedigree` を繰り返し回す。cycle ごとの要約は `artifacts/reports/netkeiba_backfill_manifest.json` に保存する。
- `run_backfill_netkeiba.py` は `--post-cycle-command` を受けられる。cycle 完了直後の安定タイミングで snapshot / benchmark を差し込む用途に使う。
- すでに別の collect/backfill が lock を保持している場合は `scripts/run_netkeiba_wait_then_cycle.py` を使うと、lock 解放を監視したあとに `--max-cycles 1` backfill と benchmark gate を自動で接続できる。待機 manifest は `artifacts/reports/netkeiba_wait_then_cycle_manifest.json`、follow-up cycle の要約は `artifacts/reports/netkeiba_backfill_handoff_manifest.json` に保存する。
- `run_collect_netkeiba.py` / `run_backfill_netkeiba.py` は共通 lock (`artifacts/reports/netkeiba_crawl_manifest.json.lock`) を使い、shared な ID CSV・output CSV・manifest の同時更新を防ぐ。
- target manifest (`artifacts/reports/netkeiba_crawl_manifest_<target>.json`) は batch 完了待ちではなく実行途中にも更新され、`status=running` と `processed_ids` で進捗を確認できる。
- `race_card` parser は `race_id + horse_id` で重複行を落とす。layout 差で同一馬が 2 回並ぶページがあるため、row count は canonical 側で正規化して扱う。
- 推奨順序:
  - `run_prepare_netkeiba_ids.py --target race_result`
  - `run_collect_netkeiba.py --target race_result` または `race_card`
  - `run_prepare_netkeiba_ids.py --target pedigree`
  - `run_collect_netkeiba.py --target pedigree`

## 7. 運用チェック
- validation CLI: [scripts/run_validate_data_sources.py](../scripts/run_validate_data_sources.py)
- feature gap CLI: [scripts/run_feature_gap_report.py](../scripts/run_feature_gap_report.py)
- snapshot CLI: [scripts/run_netkeiba_coverage_snapshot.py](../scripts/run_netkeiba_coverage_snapshot.py)
- benchmark gate CLI: [scripts/run_netkeiba_benchmark_gate.py](../scripts/run_netkeiba_benchmark_gate.py)
- snapshot JSON には target manifest 状態と readiness が含まれるため、cycle 途中の一時的不整合と benchmark rerun 可否を分離して確認できる
- benchmark gate は snapshot を 1 回更新し、`benchmark_rerun_ready=true` のときだけ train/evaluate を実行する。結果 manifest は `artifacts/reports/netkeiba_benchmark_gate_manifest.json` に保存する
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
- `sire_track_distance_last_80_win_rate` も現在は force-include missing として出る。これは `sire_name` が無い状態で擬似的な `track+distance` 集計を作らないように補正したためで、pedigree raw が入るまでこの特徴は正しく立たない。
- low coverage 側では `horse_last_3_avg_corner_2_position`, `horse_last_3_avg_corner_2_ratio`, `horse_last_3_avg_corner_gain_2_to_4` が残っており、corner 前半列の品質改善も有効候補。
- 既存 JRA raw の `corner_passing_order.csv` を supplemental として有効化しても coverage 改善は小さく、benchmark 指標はほぼ不変だったため、次の投資先は pedigree 系が優先。