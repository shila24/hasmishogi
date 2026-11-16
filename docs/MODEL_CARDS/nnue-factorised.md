# nnue-factorised

English summary: Model identity, training record, usage and recorded evaluations.

NNUE: `hlnn-kp-v1:w256:32:32:f32:cp600`、特徴量 `hlnn-kp-v1`。ネイティブ αβ 探索で評価値を計算します。

ライセンス: MIT。学習データ: 自己対戦（候補の申告）。



## 取得と利用

```sh
python3 scripts/download_models.py --model nnue-factorised
```

将棋 GUI の対局者一覧で `nnue-factorised` を選択します。「モデル読込」から `models/nnue-factorised` も指定できます。
GUI のモデル選択には `python3 -m pip install -e '.[ai]'` の依存を導入します。ネイティブエンジンは `python3 scripts/build_native.py` でビルドします。
将棋所 / ShogiGUI のエンジン設定は `Engine=alphabeta-cpp`、`Mode=strong`、`Model=<empty>` とします。

```text
setoption name Engine value alphabeta-cpp
setoption name Mode value strong
setoption name Model value <empty>
setoption name EvalFile value models/nnue-factorised/weights.hlnnw1
isready
```

## 学習記録

iteration は完了反復数、completed_epochs は実行 epoch 数です。created_at は checkpoint 作成日時、completed_by は評価時点までに学習を完了した日時の上限です。

```json
{
  "completed_epochs": 3,
  "selected_epoch": 1,
  "device": "cuda",
  "architecture": "hlnn-kp-v1:w256:32:32:f32:cp600",
  "factorised": true,
  "batch_size": 1024,
  "best_validation_loss": 0.5757099675982699,
  "seed": 5151,
  "train_records": 3262475,
  "validation_records": 357220,
  "train_loss": 0.5533698927935875,
  "validation_loss": 0.5757099675982699,
  "teacher_correlation": 0.8754297275342816,
  "completed_by": "2026-11-15T16:03:00.513890+00:00",
  "source_config_sha256": "7797f1d7c7f0b6eadb1d5bb6821281e9bb52b861442f50b0a30814e96f55968f",
  "source_metrics_sha256": "1aab7b4c222ecfbc12ba9cf54fa5bf8c328b7d054aab0b00bc2d757dc4670532"
}
```

## 学習設定

```json
{
  "amp": "off",
  "architecture": "hlnn-kp-v1:w256:32:32:f32:cp600",
  "batch_size": 1024,
  "best_validation_loss": 0.5757099675982699,
  "completed_epochs": 3,
  "correlation_clip": 3000.0,
  "device": "cuda",
  "early_stopped": true,
  "factorised": true,
  "feature_backend": "native",
  "forward": "vectorised-v1",
  "lr": 0.001,
  "lr_schedule": "none",
  "mate_target": "sigmoid saturation, unclipped scores",
  "min_delta": 0.0,
  "mix": 0.8,
  "mu": 0.0,
  "patience": 2,
  "scale": 600.0,
  "schedule_epochs": 10,
  "seed": 5151,
  "selected_epoch": 1,
  "shuffle_buffer": 65536,
  "threads": 1,
  "validation_fraction": 0.1,
  "weight_decay": 0.01,
  "workers": 2
}
```

設定 SHA256: `458206bf4f0d7374ead4c5f38ab3885c306dc5d2a01a4c52351205f79b87c823`

## 評価

評価 SHA256: `7d2566d0c78caa35acf3300314d8810a6e92e36838a3a2710f5de655bfb52a27`

| 日付 | 装置 | 対戦相手 | 探索予算 | 対局数 | 勝–敗–分 | 得点率 | 有効標本数 | 有効得点率 | Wilson 95% 区間 | 打切り数 |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 2026-11-15T16:03:00.513890+00:00 | cpu | hand-written evaluator | 1000 nodes | 400 | 129–205–66 | 0.4050 | 400 | 0.4050 | 0.358017–0.453790 | 0 |
| 2026-11-15T16:09:20.590887+00:00 | cpu | hand-written evaluator | 10000 nodes | 400 | 72–326–2 | 0.1825 | 400 | 0.1825 | 0.147727–0.223314 | 0 |
| 2026-11-15T16:10:27.835662+00:00 | cpu | hand-written evaluator | 10 ms | 400 | 105–246–49 | 0.3237 | 400 | 0.3237 | 0.279760–0.371094 | 0 |

区間は開始局面・席順のグループ平均を等しく重み付けします。旧形式の区間は記録された実対局数を用います。
