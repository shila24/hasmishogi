# 配布版の引き継ぎ

English summary: Installation, usage and verification commands for this distribution.

## 配置と収録内容

[README のゲーム別一覧](../README.md) は、実際に収録したモジュール・起動点・モデル索引から生成しています。
`hasami/` と `shogi/` は規則・GUI、`riichi/` は麻雀エンジンと点数計算 CLI です。
`poker/` と `werewolf/` は Python の規則エンジンで、学習アダプターも収録します。
Python API から利用します。
`hasami_learn/` は学習・自己対戦・評価・推論・助言・自律ループを含むパッケージ全体です。
`cpp/` はネイティブエンジンと探索、`scripts/` はビルド・モデル取得・動作確認、
`tests/` は実行時の回帰テスト、`models/` は索引と追加モデルの配置先です。
出所とファイル一覧は `SOURCE_COMMIT` と `RELEASE_MANIFEST.json` に記録します。

## ビルドとテスト

Python 3.10 以上を使います。CMake 3.16 以上と C++20 コンパイラーがあれば、
はさみ将棋・将棋・麻雀のネイティブエンジンとバッチ探索を一緒にビルドできます。

```sh
python3 scripts/build_native.py
HASAMI_REQUIRE_NATIVE=1 python3 -m unittest discover -s tests -v
python3 scripts/usi_smoke.py --classical-only
python3 launch_shogi.py --backend cpp
python3 launch_hasami.py --backend cpp
```

最初のコマンドには CTest も含まれます。はさみ将棋のコンソール版は `build/hasami_cli`
（Windows の複数構成ビルドでは `build/Release/hasami_cli.exe`）で起動できます。
Python のみなら環境変数を付けずにテストを実行します。
ネイティブや画面のテストが skip になった場合、その部分を検証済みと数えません。
Windows のコマンドプロンプトでは環境変数は別行の `set HASAMI_REQUIRE_NATIVE=1` で指定し、
`python3` を `python` に読み替えます。画面のない Linux では、導入済みなら
`xvfb-run -a python3 scripts/gui_smoke.py --backend both` で GUI を確認できます。

## モデルの取得・検証・接続

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[ai]"
.venv/bin/python scripts/download_models.py --model <モデル名>
.venv/bin/python scripts/inference_smoke.py <モデル名>
.venv/bin/python scripts/usi_smoke.py --model models/<将棋モデル名>
```

Windows の仮想環境は `.venv\Scripts\python.exe` です。モデル名は `models/index.json` から選びます。
モデルごとの取得コマンドと実測をモデルカードに掲載しています。
取得スクリプトはアーカイブの SHA256 と各ファイルの SHA256、ファイル一覧を検証して配置します。
手動でコピーする場合も [モデル案内](../models/README.md) の検証を行ってください。
`ai` extra は torch / numpy / jsonschema を導入します。取得後は推論スモークで接続を確認できます。

GUI はモデル用の Python で起動し、「モデル読込」からチェックポイントフォルダを選びます。
将棋では索引内のインストール済みモデルも利用できます。USI は `Model` にモデル名またはフォルダを指定します。
`strong`、`practical`、`level` の使い分けと Windows / WSL の登録手順は
[将棋所・ShogiGUI 接続](SHOGIDOKORO_ja.md) を参照してください。
学習コードはソース配布に含みます。重みは任意の別配布です。学習・再開・監視は [学習ガイド](TRAINING_ja.md) を参照してください。

## 麻雀

麻雀は Python API と C++ バックエンドを収録します。画面対局用 Python API と学習アダプターを提供します。
点数計算 CLI は和了前の 13 枚と和了牌を指定します。

```sh
python3 -m riichi.cli score 123m123p123s45s77p --win 6s --tsumo --riichi
```

JSON で役・符・点数を返します。点数計算の入口として利用できます。

## 実測結果と確認範囲

[評価結果](RESULTS.md) と `docs/MODEL_CARDS/` に、評価の日時・装置・対戦相手・対局数・区間を記録します。
各表の数値は記載された評価条件での測定です。
Windows 用バッチは生成・静的検査を行っています。ホストアプリへの接続手順は専用ガイドに記載しています。

## English summary

The public tree contains engines, training code, two Tk GUIs, shogi USI, a riichi scoring CLI and optional model inference.
Poker and werewolf provide Python engines and training adapters. Build native libraries with `scripts/build_native.py`,
run the runtime tests, then verify installed model assets and USI replies. Measured results live in `docs/RESULTS.md`
and the model cards. The host-integration guide includes connection checks to run after registration.
