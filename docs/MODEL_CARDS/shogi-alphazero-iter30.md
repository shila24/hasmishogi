# shogi-alphazero-iter30

English summary: Model identity, training record, usage and recorded evaluations.

alphazero の shogi 推論モデル。方策と局面評価を探索に使います。

ライセンス: MIT。学習データ: 自己対戦（候補の申告）。

学習時の search semantics version: 2。

## 取得と利用

```sh
python3 scripts/download_models.py --model shogi-alphazero-iter30
```

`python3 launch_shogi.py` で GUI を開き、
「モデル読込」で `models/shogi-alphazero-iter30` を指定します。推論依存は `python3 -m pip install -e '.[ai]'` で導入します。
USI では `Engine=python`、`Mode=strong` を選び、
`setoption name Model value models/shogi-alphazero-iter30` を設定します。

## 学習記録

iteration は完了反復数、completed_epochs は実行 epoch 数です。created_at は checkpoint 作成日時、completed_by は評価時点までに学習を完了した日時の上限です。

```json
{
  "iteration": 30,
  "created_at": "2026-11-16T03:35:38+00:00",
  "train_device": "cuda",
  "algorithm": "alphazero",
  "configured_iterations": 2000,
  "source_config_sha256": "959f45b896fdcea4cefb6b6b5bfe6040ea6176771559876bd142d0aed20d5689"
}
```

## 学習設定

```json
{
  "algorithm": {
    "name": "alphazero",
    "params": {
      "simulations": 128,
      "batch_games": 256,
      "temperature_plies": 30,
      "search_backend": "cpp"
    }
  },
  "network": {
    "name": "resnet",
    "params": {
      "channels": 128,
      "blocks": 10
    }
  },
  "encoder": {
    "name": "shogi-enc-v1",
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
    "max_plies": 400,
    "workers": 1,
    "temperature": 1.0,
    "temperature_plies": 20,
    "symmetry_augmentation": false,
    "truncation_value": 0.0,
    "draw_value_start": -0.2,
    "draw_value_end": 0.0,
    "draw_value_iterations": 100,
    "draw_reasons": [
      "repetition",
      "max_moves"
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
    "name": "shogi",
    "params": {
      "max_moves": 320,
      "declaration": "csa27"
    }
  },
  "run": {
    "iterations": 2000,
    "seed": 1,
    "backend": "cpp"
  }
}
```

設定 SHA256: `7dfc51e68d377dd385ba0328f440b8dda142d9f5f0e584725b41997f41159d9d`

## 評価

評価 SHA256: `5afe4f2fb04fc32ae6ecbe06c7f8aaa3ce9a41b76f04cf8db0bd0f80eb3a2431`

| 日付 | 装置 | 対戦相手 | 探索予算 | 対局数 | 勝–敗–分 | 得点率 | 有効標本数 | 有効得点率 | Wilson 95% 区間 | 打切り数 |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: |
| 2026-11-16T04:06:14.281379+00:00 | cuda:0 | random | 64 simulations, 200 ms | 40 | 24–2–14 | 0.7750 | 40 | 0.7750 | 0.624966–0.876840 | 0 |
| 2026-11-16T04:21:46.457835+00:00 | cuda:0 | greedy | 64 simulations, 200 ms | 40 | 3–1–36 | 0.5250 | 40 | 0.5250 | 0.374971–0.670648 | 0 |
| 2026-11-16T04:31:43.193821+00:00 | cuda:0 | alphabeta:1 | 64 simulations, 200 ms | 40 | 0–29–11 | 0.1375 | 40 | 0.1375 | 0.062490–0.276038 | 0 |
| 2026-11-16T04:38:17.283022+00:00 | cuda:0 | alphabeta:2 | 64 simulations, 200 ms | 40 | 0–40–0 | 0.0000 | 40 | 0.0000 | 0.000000–0.087625 | 0 |

区間は開始局面・席順のグループ平均を等しく重み付けします。旧形式の区間は記録された実対局数を用います。
