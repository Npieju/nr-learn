この workspace では、ユーザーとの対話、進捗共有、質問への回答、最終報告は常に日本語で行う。

ユーザーが明示的に別の言語を要求した場合のみ、その言語へ切り替える。

コード、設定キー、ライブラリ名、識別子、CLI コマンドは、必要に応じて原文のまま扱ってよい。

このリポジトリでは、基本的に自動 coding を前提とする。ただし、次の原則を守ること。

- issue は prompt そのものとして扱う。曖昧なら、まず issue を具体化する。
- 1 task で 1 measurable hypothesis に絞る。
- `JRA` を benchmark 正本として扱う。`NAR` は readiness track として扱い、主 KPI 判定に混ぜない。
- broad な strategy change や baseline 更新は human review 前提で進める。
- 変更前に、目的、非目的、成功条件、検証方法を明確にする。
- explore -> plan -> code -> verify の順を守る。
- ML 変更では `objective -> dataset freeze -> baseline artifact confirmation -> hypothesis -> implementation -> smoke -> formal evaluation -> promotion decision` の順を守る。
- 実装だけでなく、必要な tests、docs、artifact path 更新まで含めて完了とする。
- review で繰り返し出る指摘は、その場限りで済ませず docs / instructions / tests に昇格する。
- 短窓の良い結果だけで promote しない。formal gate と artifact を優先する。
- デフォルトブランチへ直接反映する前に、必ず差分とリスクを明文化する。
- Git の更新を長く滞らせない。同じ変更単位の code / config / script / docs は意味のある小さめの batch にまとめ、issue の進捗と乖離しないタイミングで commit / push する。
- Pylance の code snippet execution は軽い probe / 集計 / 単発確認に限る。train / evaluate / revision gate / serving compare / backfill / multi-date replay などの重い Python script を Pylance で回さない。
- 重い Python script を実行するときは、先に workspace の Python environment を整えたうえで terminal 実行に切り替え、必ず外部ログファイルを残す。script 自身に `--log-file` があればそれを使い、無ければ shell redirect や `tee` で `artifacts/logs/...` に保存し、その path をユーザーへ明示する。
- Python 実行が import / interpreter / package / environment 起因で落ちた場合は、すぐ別手段へ迂回せず、まず Python environment を修復・確認する。少なくとも interpreter、主要 package、実行コマンドを確認してから再実行する。
