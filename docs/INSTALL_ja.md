# インストール

English summary: Installation, usage and verification commands for this distribution.

Python 3.10 以上と Tk を使います。配布物を展開して、そのフォルダでコマンドを実行してください。古典的 AI で遊ぶ場合、torch は不要です。

## Windows

python.org の Python インストーラーで Tcl/Tk を含めて導入します。ターミナルで次を実行し、小さい確認ウィンドウが開くことを確認します。

```powershell
python -m tkinter
python launch_hasami.py --backend python
```

## macOS

Homebrew の Python と、同じ Python バージョンに対応する `python-tk` を導入します。Tk の版と実行する Python の版を合わせてください。

```sh
brew install python python-tk
python3 -m tkinter
python3 launch_hasami.py --backend python
```

## Linux / WSL2

Ubuntu 系では `python3`、`python3-venv`、`python3-tk` が必要です。WSL2 は WSLg など、ウィンドウを表示できる環境で使います。

```sh
sudo apt install python3 python3-venv python3-tk
python3 -m tkinter
python3 launch_hasami.py --backend python
```

GUI は表示環境を使用します。Python エンジンのテストは `python3 -m unittest discover -s tests -v` で実行できます。

## C++ エンジン（任意）

CMake 3.16 以上と C++20 対応コンパイラーを導入します。Windows では Visual Studio の C++ 開発ツール、macOS では Command Line Tools、Linux では GCC または Clang を使えます。

```sh
python3 scripts/build_native.py
python3 launch_hasami.py --backend cpp
```

`--backend cpp` でライブラリが見つからない場合はエラーになります。Python 版を使うときは明示的に `--backend python` を指定します。

## 学習済みモデル（任意）

評価済みモデルが公開されている場合に利用できます。仮想環境を作り、torch と numpy を導入してください。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[ai]"
.venv/bin/python scripts/download_models.py
.venv/bin/python launch_hasami.py --backend python
```

Windows では `.venv/bin/python` を `.venv\Scripts\python.exe` に読み替えます。CPU だけでも推論できます。CUDA を使う場合は、利用する OS と NVIDIA ドライバーに対応する PyTorch を公式のインストール案内に従って導入してください。ダウンロードスクリプトは公開版のチェックサムと照合します。GUI 自体はネットワークにアクセスしません。

## 将棋

将棋 GUI も同じ Python と Tk で動きます。古典的 AI（ランダム・貪欲・αβ探索）だけなら追加の導入は不要です。

```sh
python3 launch_shogi.py --backend python
```

C++ エンジンをビルドすると（上の「C++ エンジン」）`--backend cpp` で高速なエンジンを使えます。
学習済みモデル・レベル別 AI・「間違いにくい手」の検討モードを使うには、上の「学習済みモデル」の手順で torch を
入れた `.venv` の Python から起動してください。

```sh
.venv/bin/python launch_shogi.py --backend cpp
```

ShogiGUI や将棋所から使う場合は、USI エンジンとして次のコマンドを登録します（Windows では `.venv\Scripts\python.exe`）。

```sh
.venv/bin/python -m shogi.usi --backend cpp
```

## ソースの取得

```sh
git clone https://github.com/shila24/hasmishogi.git
cd hasmishogi
```

各 OS の導入手順を上に示します。配布検証の実行環境とモデルの実測は [評価結果](RESULTS.md) とモデルカードを参照してください。
