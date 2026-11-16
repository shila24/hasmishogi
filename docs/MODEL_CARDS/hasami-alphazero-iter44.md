# hasami-alphazero-iter44

English summary: Model identity, training record, usage and recorded evaluations.

alphazero の hasami 推論モデル。方策と局面評価を探索に使います。

ライセンス: MIT。学習データ: 自己対戦（候補の申告）。

学習時の search semantics: legacy（バージョン指定前）。推論用アセットとして利用します。

## 取得と利用

```sh
python3 scripts/download_models.py --model hasami-alphazero-iter44
```

`python3 launch_hasami.py` で GUI を開き、
「モデル読込」で `models/hasami-alphazero-iter44` を指定します。推論依存は `python3 -m pip install -e '.[ai]'` で導入します。

## 学習記録

iteration は完了反復数、completed_epochs は実行 epoch 数です。created_at は checkpoint 作成日時、completed_by は評価時点までに学習を完了した日時の上限です。

```json
{
  "iteration": 44,
  "created_at": "2026-11-15T04:57:34+00:00",
  "train_device": "cuda",
  "algorithm": "alphazero",
  "configured_iterations": 60,
  "source_config_sha256": "59c3af5f9bde95512bb4d8066a6818169588e2da0f716d26c921ed4ff0971881"
}
```

## 学習設定

```json
{
  "algorithm": {
    "name": "alphazero",
    "params": {
      "simulations": 64,
      "batch_games": 256,
      "search_backend": "cpp"
    }
  },
  "network": {
    "name": "resnet",
    "params": {
      "channels": 64,
      "blocks": 4
    }
  },
  "encoder": {
    "name": "hasami-enc-v1",
    "params": {}
  },
  "optimizer": {
    "name": "adam",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "momentum": 0.9,
    "grad_clip": 1.0,
    "updates_per_iteration": 400,
    "lr_schedule": "constant"
  },
  "replay": {
    "capacity": 200000,
    "batch_size": 256,
    "min_size": 256,
    "persist": true
  },
  "selfplay": {
    "games_per_iteration": 256,
    "max_plies": 300,
    "workers": 1,
    "temperature": 1.0,
    "temperature_plies": 20,
    "symmetry_augmentation": true,
    "truncation_value": 0.0,
    "draw_value_start": -0.2,
    "draw_value_end": 0.0,
    "draw_value_iterations": 100,
    "draw_reasons": [
      "repetition"
    ]
  },
  "device": {
    "device": "cuda",
    "amp": "auto",
    "threads": null,
    "deterministic": false,
    "inference_batch": 256,
    "fixed_batch_shape": true
  },
  "game": {
    "name": "hasami",
    "params": {}
  },
  "run": {
    "iterations": 60,
    "seed": 1,
    "backend": "cpp"
  }
}
```

設定 SHA256: `1041ebed78b7868870f25a2cd675729a2e24a6db63092a6eb5ceb956ca5cb0ad`

## 評価

評価 SHA256: `4858be9f4552c26c98c80c512ba38044a6e0ce91af098888cbea9876494fc41d`

| 日付 | 装置 | 対戦相手 | 探索予算 | 対局数 | 勝–敗–分 | 得点率 | 有効標本数 | 有効得点率 | Wilson 95% 区間 | 打切り数 |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 2026-11-16T05:36:02.900727+00:00 | cuda:0 | random | 64 simulations, 200 ms | 40 | 14–2–24 | 0.6500 | 40 | 0.6500 | 0.495056–0.778657 | 24 |
| 2026-11-16T05:40:22.844028+00:00 | cuda:0 | greedy | 64 simulations, 200 ms | 40 | 0–15–25 | 0.3125 | 40 | 0.3125 | 0.190743–0.467116 | 1 |
| 2026-11-16T05:42:04.933564+00:00 | cuda:0 | alphabeta:1 | 64 simulations, 200 ms | 40 | 0–39–1 | 0.0125 | 40 | 0.0125 | 0.001306–0.109128 | 0 |
| 2026-11-16T05:43:21.957498+00:00 | cuda:0 | alphabeta:2 | 64 simulations, 200 ms | 40 | 0–40–0 | 0.0000 | 40 | 0.0000 | 0.000000–0.087625 | 0 |

区間は開始局面・席順のグループ平均を等しく重み付けします。旧形式の区間は記録された実対局数を用います。
