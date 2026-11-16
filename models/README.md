# モデル

English summary: Download indexed checkpoint and NNUE assets, verify their hashes, and use the recorded model cards.

重みは Git に含めません。公開済みのモデルは `index.json` に、ゲーム、サイズ、SHA256、MIT ライセンス、公開タグ、評価証拠とともに記録されます。各モデルの用途と学習記録は下の一覧から確認できます。

配布フォルダで次を実行すると、その版の公開モデルを取得します。

```sh
python3 scripts/download_models.py
python3 scripts/download_models.py --model MODEL_NAME
```

スクリプトは公開タグの `SHA256SUMS` とローカルの一覧を照合し、アーカイブ全体の SHA256 を確認してから展開します。一覧に記録された全ファイルの有無と個別の SHA256 も確認します。`levels.json` や `engine/` を含むモデルは、フォルダ構成を保ってまとめて配置します。絶対パス、親への移動、リンク、予定外のファイルは拒否します。すでに同名のフォルダがある場合は上書きせず停止します。

展開先は `models/MODEL_NAME/` です。GUI の「モデル読込」でこのフォルダを指定してください。`ai` extra（torch / numpy / jsonschema）が必要です。GUI から自動ダウンロードは行いません。

学習データは自己対戦です。公開モデルごとの条件と実測は [評価結果とモデルカード](../docs/RESULTS.md) を参照してください。

NNUE アセットは将棋 GUI の対局者から選べます（GUI のモデル選択には `ai` extra を導入します）。将棋所 / ShogiGUI では `Engine=alphabeta-cpp`、`Mode=strong`、`Model=<empty>` とし、`EvalFile=models/<モデル名>/weights.hlnnw1` を指定します。USI の NNUE 推論は Python 標準ライブラリとネイティブエンジンで動作します。

## この版のモデル

| モデル | ゲーム | 種類 | 学習記録 |
| --- | --- | --- | --- |
| [shogi-alphazero-iter30](../docs/MODEL_CARDS/shogi-alphazero-iter30.md) | shogi | checkpoint | 30 |
| [hasami-alphazero-iter44](../docs/MODEL_CARDS/hasami-alphazero-iter44.md) | hasami | checkpoint | 44 |
| [nnue-factorised](../docs/MODEL_CARDS/nnue-factorised.md) | shogi | nnue | 3 |

# models

English summary: Distributed source and entry points; see the generated code guide for the full module inventory.

公開モデルの索引と検証付き取得方法。重みは Releases のアセットから取得します。

[全ファイルの役割と接続](../docs/CODE_GUIDE_ja.md#models)

