# 将棋所・ShogiGUI への接続

English summary: Installation, usage and verification commands for this distribution.

## 準備

[配布リポジトリ](https://github.com/shila24/hasmishogi) の Code → Download ZIP、または Releases から取得し展開します。将棋所 / ShogiGUI 本体は各配布元から別途導入してください。


Python 3.10 以上とこの配布物が必要です。USI は標準入出力で動き、Tk・画面・GPU は不要です。
最初は組込みの古典探索、Python バックエンドで確認してください。C++ を使う場合は
[引き継ぎガイド](HANDOFF_ja.md) のビルドを先に行います。

## Windows の Python で登録

1. Windows 用 Python を導入し、コマンドプロンプトで `python --version` を確認します。
2. 配布物のフォルダで `python scripts/usi_smoke.py --classical-only` を実行し、`"ok": true` を確認します。
3. 将棋所または ShogiGUI のエンジン管理画面を開き、「追加」から `launch_shogi.py` と同じ場所にある
   `shogi-usi.bat` を選びます。ファイル選択が実行ファイルに限定されていれば全ファイル表示に切り替えます。
   バッチの登録を受け付けない版では、そのアプリの起動コマンド設定で
   `cmd.exe /c "C:\games\hasmishogi\shogi-usi.bat"` を指定します。
   起動引数を設定できない版については、このバッチを登録できることを確認してから利用してください。
4. エンジン名 `Hasami Lab Shogi` が表示されたら登録し、対局の先手または後手に選びます。
5. エンジン設定はまず `Model` を空、`Mode` を `strong`、`Depth` を `3` にします。
   秒読みを設定して対局を開始し、指し手が返ることを確認します。

バッチは自分のフォルダを作業ディレクトリにし、`.venv\Scripts\python.exe` があれば優先します。
なければ PATH の `python` を使います。バッチをダブルクリックすると入力待ちになります。
手動確認は `usi`、`isready`、`quit` を一行ずつ入力してください。

## Windows 画面と WSL 内のエンジンで登録

1. WSL の既定ディストリビューション内に Python と配布物を用意します。
   WSL の端末で配布物へ移動し、`pwd` で Linux 側の絶対パスを確認します。
2. 同じ端末で `python3 scripts/usi_smoke.py --classical-only` を実行します。
   モデルを使う場合の仮想環境・依存ライブラリも WSL 側に導入します。
3. `shogi-usi-wsl.bat` 冒頭の `WSL_REPO` を、手順 1 で確認したパスに変更します。
   この変数には Windows のドライブ表記を入れません。既定のディストリビューション以外を使う場合は
   `wsl.exe` に `--distribution` とその名前を追加します。
4. Windows 側の将棋所 / ShogiGUI に `shogi-usi-wsl.bat` を登録します。
   ファイルの選択・起動コマンドの扱いは上の Windows 手順と同様です。
5. エンジン名を確認して対局者に選びます。`Model` に直接パスを指定する場合は WSL 側のパスにします。

このバッチは `wsl.exe --cd` で作業ディレクトリを指定し、WSL 側の `.venv/bin/python`、
なければ `python3` を実行します。WSLg と Tk は USI 接続には不要です。

## 実装されている USI オプション

以下の宣言は配布物のエンジンから生成します。ランチャーはオプションを固定せず、登録先の設定画面から指定します。

| 名前 | USI 宣言（種類・既定値・選択肢） | 内容 |
| --- | --- | --- |
| `Model` | `option name Model type string default <empty>` | モデル名またはチェックポイントフォルダ。空なら古典探索 |
| `EvalFile` | `option name EvalFile type string default <empty>` | `Engine=alphabeta-cpp` 専用の `hlnnw1` NNUE 重みファイルのパスです。空なら手書き評価を使います。将棋所 / ShogiGUI のエンジン設定で `Engine` を `alphabeta-cpp`、`Model` を空、`Mode` を `strong` にして、この欄に重みファイルのパスを指定します（WSL ランチャーでは WSL 側のパス）。チェックポイントフォルダを指定する `Model` とは別です。対応するネイティブビルドが必要で、ファイル欠落・形式や特徴量、構造、検証値の不一致は `isready` / `go` で `info string error` となり、別の評価へ自動変更しません。公開 NNUE 重みは `models/index.json` に記録します。取得後は `models/<モデル名>/weights.hlnnw1`、またはインストール済みモデル名を指定します |
| `Engine` | `option name Engine type combo default python var python var alphabeta-cpp` | `python` は Python の古典探索またはモデル推論。`alphabeta-cpp` は C++ の古典 αβ 探索で、`Model` は空、`Mode=strong` が必須です。`python3 scripts/build_native.py` によるネイティブ探索ライブラリのビルドが必要です。ライブラリがない、または ABI が不一致の場合は `isready` / `go` で `info string error` を返し、Python 探索へ自動変更しません。`--backend`（局面エンジンの選択）とは別の設定です |
| `Mode` | `option name Mode type combo default strong var strong var practical var level` | `strong` / `practical` / `level` |
| `Method` | `option name Method type combo default search var auto var search var learned` | `auto` / `search` / `learned`。助言器の方式 |
| `Level` | `option name Level type spin default 1 min 1 max 1000000` | `practical` / `level` の代理レベル。実際に使える値はモデルの収録レベルに従う |
| `MaxNodes` | `option name MaxNodes type spin default 200000 min 1 max 2147483647` | 探索ノード上限 |
| `Depth` | `option name Depth type spin default 3 min 1 max 64` | 古典探索の深さ |

- 古典探索: `Model=<empty>`、`Mode=strong`。追加モデルは不要です。
- モデル推論: `Model=models/<モデル名>`、`Mode=strong`。将棋用モデルが必要です。
- 実戦的助言: 対応するモデルを指定し、`Mode=practical`、`Method=search`、収録された `Level` を指定します。
  `Method=learned` は対応する学習済み助言モデルが必要です。`Mode=level` は代理プレイヤーで指します。
  表示値は人間代理プログラム上の期待得点の推定です。

モデルは [取得と検証](../models/README.md) に従い `models/<モデル名>/` に配置します。
`models/index.json` に登録された、インストール済みの将棋モデルは `Model` に名前で指定できます。
組込みの古典探索も選択できます。モデル推論には torch と numpy が必要です。

## 時間設定と接続確認

`go` は `btime/wtime/byoyomi/binc/winc/movetime/nodes/depth/infinite` を受け付けます。
秒読みがあれば 200 ms の余裕を引き、なければ残り時間の約 1/30 と加算時間を使います。
短い秒読みでも最低 1 ms です。`stop` は現在の合法な最善候補を返します。ponder は無効です。
まず秒読み 2 秒以上で接続を確認し、装置とモデルの応答時間に合わせて設定してください。

```sh
python3 scripts/usi_smoke.py
python3 scripts/usi_smoke.py --model models/<モデル名> --timeout 60
python3 scripts/usi_smoke.py --eval-file <NNUE重みファイル> --timeout 60
```

最初のコマンドは古典探索と、インストール済みの索引内の将棋チェックポイント・NNUE アセットを検査します。
`usi` とオプション、`isready`、新規対局、手順付き局面、秒読み探索、合法な `bestmove`、
`go infinite` / `stop`、`quit` の終了を確認します。NNUE の検査はネイティブ探索を使うため、先に `python3 scripts/build_native.py` でビルドします。
`--model` はチェックポイントフォルダ、`--eval-file` は NNUE 重みファイルを選びます。`scripts/inference_smoke.py <モデル名>` でも native / USI を検査できます。

## 困ったとき

- 応答がない: バッチをコマンドプロンプトから起動し、Python のパス・作業フォルダ・WSL の変数を確認します。
  `usi` に `usiok` が返るか確認し、スモークテストのタイムアウトも調整します。
- モデルが見つからない: 名前、ゲーム種別、`manifest.json` と `weights.pt`、実行中の Python を確認します。
  `isready` は初期化エラーの後にも `readyok` を返すため、`info string error` を見落とさないでください。
- 依存ライブラリのエラー: モデルを実行する側の仮想環境を確認します。
  `python -m pip install -e ".[ai]"` は torch / numpy / jsonschema を追加します。推論スモークで確認してください。
- Tk のエラー: GUI には Tk が必要ですが、USI は `python -m shogi.usi` で起動し、Tk を使いません。
- C++ ライブラリがない: ビルドするか、既定の Python バックエンドに戻します。

## 検証範囲 / English summary

この配布準備ではバッチを生成し、静的な構文・引用符の検査を行いました。
標準入出力の USI 接続は同梱スモークで確認できます。
アプリの版によるバッチ登録方法は利用環境で確認が必要です。

Use the native Windows or WSL launcher to register the USI engine. Classical play needs only Python;
models are optional. Run the subprocess smoke test before registration. The launchers were generated and
statically checked. The connection checks above exercise the installed engine.
