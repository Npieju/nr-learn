# 開発フロー

## 1. この文書の役割

この文書は、`nr-learn` で日々の試行と正式な改善判断をどう分けるかを定義する正本である。

`project_overview.md` が「このプロジェクトは何か」を説明する資料だとすると、この文書は「このプロジェクトをどう進めるか」を説明する資料である。

進行中の優先順位、直近の判断済み事項、次の作業順は [roadmap.md](roadmap.md) を正本として別管理する。

ここで決めるのは次の 4 点である。

1. 短い smoke / probe を何のために使うか。
2. 正式な改善判断を何で行うか。
3. `revision` をどの単位で切るか。
4. 長時間ジョブにどの程度の進捗可視化を求めるか。

## 2. 基本原則

- 100〜200 レース程度の短窓結果は、方向確認には使っても昇格根拠には使わない。
- 改善判断は、一定区切りごとの full train / evaluate / promotion gate を通した結果で行う。
- 実験の速さと、採用判断の厳しさは分けて運用する。
- 長時間ジョブは、途中経過が見えることを標準とする。
- この workspace / container では full train や full evaluate の並列実行で OOM kill が起きうるため、正式 checkpoint は直列実行を基本とする。
- operator-facing CLI は、設定ミス・入力不足・output path の取り違えを fail-fast で検出し、想定内の失敗では traceback を出さない。
- まとまった変更を終えたら `git status` と diff を確認し、意味のある単位で commit する。共有 remote が使える作業では push までを完了条件に含め、push できない場合は理由を明示して残す。

## 3. 2 段階の評価運用

### 3.1 smoke / probe

短い期間、少ないレース数、限定 window の評価は smoke / probe とみなす。

用途は次のとおりである。

- 設定変更で処理が壊れていないか確認する。
- policy の挙動差を素早く比較する。
- 次に full evaluation を回す価値があるかを絞り込む。

この段階では、次のようなスクリプトを使う。

- [../scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- [../scripts/run_serving_smoke_compare.py](../scripts/run_serving_smoke_compare.py)
- [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py)

serving 側の具体的な導線は [serving_validation_guide.md](serving_validation_guide.md) にまとめてある。

ただし、この段階の改善は正式採用とみなさない。

### 3.2 revision gate

正式な改善判断は、full train / evaluate / promotion gate をまとめて通した revision 単位で行う。

判定の中心は次の 2 つである。

- `run_evaluate.py` の `stability_assessment` が `representative` であること
- `run_promotion_gate.py` が `pass` すること

この 2 条件を満たさない限り、短窓で結果が良くても採用しない。

## 4. revision の切り方

`revision` は、単なる思いつきの設定変更ごとではなく、一定区切りの full judgment ごとに切る。

推奨する考え方は次のとおりである。

- 複数の軽い probe で候補を絞る。
- 有望候補だけ full evaluation に進める。
- full evaluation と promotion gate を通した単位を 1 revision とする。
- コミットや比較メモも、その revision 名を軸に揃える。

この運用にすると、後から見返したときに「どの変更が正式に評価済みか」が分かりやすい。

## 5. 標準フロー

### 5.1 日常の探索

1. serving smoke や replay compare で方向性を見る。
2. 短窓では bets、total net、bankroll の悪化有無を見る。
3. 良さそうな候補だけを full evaluation 候補に残す。

### 5.2 正式評価

1. train を実行する。
2. representative な条件で evaluate を実行する。
3. promotion gate を実行する。
4. pass したものだけを revision として扱う。

上の 1〜3 は、少なくともこの dev container では並列に投げず直列で回す。`run_revision_gate.py` を使う場合も、別の full train / evaluate を同時に走らせないことを前提にする。

evaluation と promotion gate の具体的な読み方は [evaluation_guide.md](evaluation_guide.md) にまとめてある。

この流れは [../scripts/run_revision_gate.py](../scripts/run_revision_gate.py) で 1 コマンドにまとめられる。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --profile current_best_eval \
  --revision r20260321a \
  --evaluate-max-rows 200000 \
  --evaluate-wf-mode full
```

重い run に入る前に command 解決と manifest 出力だけ確認したいときは `--dry-run` を付ける。

実際に 1 本通す smoke では、`--train-max-train-rows` / `--train-max-valid-rows` と evaluate 側の row 制限を併用して負荷を下げてよい。

2025 まで backfill した最新データを正式判断へ載せるときは、通常の `run_revision_gate.py` を手で組み立てる代わりに、readiness 確認込みの wrapper を使ってよい。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_latest_revision_gate.py \
  --revision r20260323_2025latest \
  --evaluate-pre-feature-max-rows 300000 \
  --evaluate-max-rows 200000 \
  --evaluate-wf-mode full
```

この wrapper は次を直列で行う。

- 2025 backfill 用 manifest を明示指定した coverage snapshot を実行する。
- `benchmark_rerun_ready=true` を確認する。
- `current_best_eval_2025_latest` を使って revision gate を起動する。

まず command 解決だけ確認したいときは `--dry-run` を付ける。pedigree crawl の完了待ちも同じ入口で行いたいときは `--wait-timeout-seconds` を付ける。

2025 latest の evaluate は full table feature build だと OOM になりやすいため、この wrapper は `--evaluate-pre-feature-max-rows` を既定で `300000` にしている。必要なら明示的に上書きする。

主な出力:

- `artifacts/reports/promotion_gate_<revision>.json`
- `artifacts/reports/revision_gate_<revision>.json`

## 6. 採用判断で見るもの

正式な採用判断では、短窓の単発 ROI ではなく次をまとめて確認する。

1. nested / walk-forward で改善しているか。
2. `stability_assessment=representative` を満たしているか。
3. feasible fold が十分あるか。
4. actual calendar の比較で total net や bankroll を壊していないか。
5. 同等挙動を出せるより単純な候補がないか。

## 7. 進捗表示の方針

長時間ジョブは、処理が進んでいるか見えることを標準とする。

原則は次のとおりである。

- CLI には `ProgressBar` または `Heartbeat` を入れる。
- train / evaluate / backtest / gate のような長い処理は、開始と完了だけでなく途中経過も出す。
- 新しく追加する orchestration script でも progress を省略しない。
- operator-facing CLI では、`output file` と `output dir` を明確に区別して早い段階で検証する。
- 想定内の operator error は concise な `failed: ...` で返し、unexpected exception のときだけ traceback を維持する。
- `src/` 側の実装でも、JSON / text / CSV / PNG / model artifact の書き出しは `racing_ml.common.artifacts` の helper を優先して使い、file path 検証を call site ごとに重複させない。
- `scripts/` 側でも path 表示、output file / output dir 検証、JSON / text / CSV / PNG / model artifact の書き出しは `racing_ml.common.artifacts` を直接呼び、薄い wrapper や直書きを script ごとに持ち込まない。
- ただし、`os.O_EXCL` で取得した lock file のように atomic create と同じ file descriptor へ即時に書く必要があるケースは例外とし、shared helper より atomicity を優先する。

実務上は次のように解釈する。

- 長時間 script を起動したとき、無言で数分以上止まる状態を標準にしない。
- `run_serving_profile_compare.py`、`run_backfill_netkeiba.py`、`run_revision_gate.py`、`run_wf_feasibility_diag.py` のような重い入口では、段階ごとの進捗が見えることを前提に運用する。
- 補助的な集計 script でも、入力読込、集計生成、artifact 書き出しのどこにいるか分かるようにする。
- 作業を人へ引き渡す前には、作業対象の diff を確認してから commit し、通常運用では push まで済ませる。remote 権限や認証で push できない場合は、その時点で止めずに失敗理由を報告する。
- ただし、`check_meeting_run_status.sh` のような短命な status / inspection script は、progress bar を持たなくてもよい。重要なのは即時に状態を返すことである。

## 8. 実務上の解釈

要するに、このプロジェクトでは次の分担で運用する。

- 短窓評価: 速く回して候補を絞る
- full revision gate: 正式な強化判断をする

この 2 つを混ぜないことが、改善の積み上げを壊さないための基本ルールである。