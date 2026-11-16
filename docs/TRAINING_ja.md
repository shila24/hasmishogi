# 学習と自律ループ

English summary: Install the optional learning dependencies, run CPU smoke training, resume checkpoints, and monitor the WSL2 loop. Sleep continuation and process-kill restart were verified on WSL2 on 2026-11-15.

## インストールとデバイス

Linux / WSL2 / macOS ではリポジトリのルートで実行します。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[learn]"
.venv/bin/python -m hasami_learn.cli devices
.venv/bin/python -m hasami_learn.cli list
python3 scripts/build_native.py
```

`learn` と `ai` はいずれも直接依存として `torch>=2.2`、`numpy>=1.26`、`jsonschema>=4.18` を導入します。PyTorch のデバイスライブラリなどの間接依存は選んだ wheel と環境で変わります。`yaml` は `pyyaml>=6` だけを直接追加します（JSON 設定では不要）。本体は Python 標準ライブラリで動作します。新しい clone の学習コマンドは `learn` で足ります。依存パッケージは利用環境に合わせて導入します。

`--device auto` は CUDA → MPS → CPU の順です。CPU は tiny の検証用です。NVIDIA は WSL2 側で利用し、CUDA 対応の PyTorch を選びます。WSL 内へ NVIDIA ドライバーを重ねて入れません。`nvidia-smi` と devices 出力で確認します。デバイス別の性能は各実測の装置欄を参照してください。Windows で Python コマンドを試す場合は `python3` を `python`、`.venv/bin/python` を `.venv\Scripts\python.exe` に読み替えます。

## CPU 接続確認

```sh
.venv/bin/python -m hasami_learn.cli train --algorithm alphazero --tiny --device cpu --output-dir runs/release-smoke
.venv/bin/python -m hasami_learn.cli train --config configs/tiny/alphazero-shogi.json --device cpu --output-dir runs/release-shogi
```

tiny は学習から推論までの接続確認用です。ポーカーと人狼もエンジン・学習コード・Python API と学習アダプターを提供します。設定は `configs/tiny/`、入口は `hasami_learn.cli` です。

<a id="resume"></a>
## 保存・停止・再開

```sh
.venv/bin/python -m hasami_learn.cli train --algorithm alphazero --tiny --device cpu --output-dir runs/release-smoke --resume auto
```

`--resume auto` は最新の正常な保存から再開し、なければ新規開始します。明示パスも指定できます。保存済み config が基準で、変更可能なのはデバイス、延長する反復数、quiet です。Ctrl+C / SIGTERM は反復の更新後に保存します。強制終了や電源断では保存以後の作業を失う場合があります。replay とモデルはそれぞれの保存時点を持ちます。

長時間学習には `bash scripts/train_supervisor.sh configs/alphazero.json --device cuda` を使えます。同じ run を二つのプロセスで書かないでください。

## WSL2 の自律ループ

```sh
.venv/bin/python -m hasami_learn.autoloop validate --config configs/autoloop/tiny-hasami.json
bash scripts/autoloop/install_wsl.sh --dry-run configs/autoloop/tiny-hasami.json
.venv/bin/python -m hasami_learn.autoloop run --config configs/autoloop/tiny-hasami.json
.venv/bin/python -m hasami_learn.autoloop status --loop runs/autoloop-tiny/tiny-hasami --json
.venv/bin/python -m hasami_learn.autoloop stop --loop runs/autoloop-tiny/tiny-hasami
.venv/bin/python -m hasami_learn.autoloop resume-check --loop runs/autoloop-tiny/tiny-hasami
```

tiny は CPU 学習 → 評価 → gate → ローカル書出しを 1 サイクル実行します。閾値 0 は接続確認用です。完了後の再起動は完了済み工程を繰り返しません。出力は WSL の Linux ファイルシステムに置きます。`/mnt/` 配下は拒否します。

systemd 有効の WSL2 で、所有者がサービスを導入するときのコマンドです。この配布検証ではサービス変更を実行しません。

```sh
bash scripts/autoloop/install_wsl.sh configs/autoloop/tiny-hasami.json
unit=hasami-autoloop@tiny-hasami
systemctl --user start "$unit.service"
systemctl --user status "$unit.service"
systemctl --user stop "$unit.service"
```

2026-11-15 の WSL2 確認ではサービス起動、PC スリープ後の自動継続、プロセス強制終了後の再起動を確認しました。この確認は上記の停止・復帰条件で行いました。ループは復帰検知後に別プロセスでヘルス検査し、失敗が続くと `needs_wsl_restart` で止まります。ディスク不足は `paused_disk`、再試行上限は `blocked` です。Windows 用タスクと PowerShell スクリプトも収録しています。`scripts/windows/` の既定はスリープ許可で、WSL 強制再起動を有効にするかは所有者が判断します。

本番設定の `require_validation` を無効化して回避しないでください。`bash scripts/validate_training.sh --expect-commit <公開コミットSHA>` が native / sanitizer / 両 Python / CUDA smoke / 再開等を検査して記録します。PASS と現在の正確な commit、clean な tracked tree が必要です。この長時間 CUDA ゲートは WSL2/CUDA ホストで実行し、その環境の検査結果を記録します。

<a id="evaluation"></a>
## 評価と診断

評価・モデル書出しは [進捗と診断](PROGRESS_ja.md)、学習方法と測定条件は [手法](METHOD_ja.md)、公開済みモデルの結果は [RESULTS](RESULTS.md) を参照してください。学習出力やローカル export は自動公開されません。
