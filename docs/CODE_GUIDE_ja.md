# コード案内

English summary: This inventory is generated from the distributed files. Engines feed game adapters, training and inference; GUIs and USI consume agents.

規則エンジン → `hasami_learn/games/` のアダプター → 自己対戦・学習 → チェックポイント → 推論エージェント → GUI / USI の順につながります。
GUI は規則を複製せずエンジンに委譲します。学習を使わない規則エンジンは標準ライブラリで動きます。

| ディレクトリ | 役割・読む順番 |
| --- | --- |
| [configs/](../configs/README.md) | 学習・NNUE・自律ループの JSON 設定。tiny/ は接続確認用です。 |
| [cpp/](../cpp/README.md) | C++20 の規則、C ABI、MCTS、将棋探索と NNUE。CMakeLists.txt と各 include/ が入口です。 |
| [docs/](../docs/README.md) | 利用者向けの操作・学習・手法・実測範囲の説明。 |
| [examples/](../examples/README.md) | 信頼済みプラグインと外部エージェント設定の例。 |
| [hasami/](../hasami/README.md) | はさみ将棋の規則・履歴・Python/C++ 接続・Tk GUI。engine.py から読みます。 |
| [hasami_learn/](../hasami_learn/README.md) | ゲームアダプター、自己対戦、学習、評価、推論、運用。cli.py と games/ から読みます。 |
| [models/](../models/README.md) | 公開モデルの索引と検証付き取得方法。重みはソースに同梱しません。 |
| [poker/](../poker/README.md) | ポーカーの規則・観測・評価。GUI はありません。 |
| [riichi/](../riichi/README.md) | リーチ麻雀の規則・得点・C++ 接続。game.py と cli.py が入口です。 |
| [scripts/](../scripts/README.md) | ビルド、計測、GUI/USI の確認、学習の起動と監視。下のファイル一覧を参照。 |
| [shogi/](../shogi/README.md) | 本将棋の規則・棋譜・探索接続・GUI・USI。engine.py、native_search.py、usi.py が入口です。 |
| [tests/](../tests/README.md) | 公開ファイルだけで実行する規則・接続・学習テスト。unittest discover で実行します。 |
| [werewolf/](../werewolf/README.md) | 5 人の人狼の規則・観測・基準エージェント。GUI はありません。 |

## 起動ファイル

| ファイル | 役割 |
| --- | --- |
| [launch_hasami.py](../launch_hasami.py) | Launch the Hasami Shogi GUI. |
| [launch_shogi.py](../launch_shogi.py) | Launch the Shogi GUI when the runtime module is available. |
| [shogi-usi-wsl.bat](../shogi-usi-wsl.bat) | Windows から WSL 内の USI エンジンを起動（Windows 未実行）。 |
| [shogi-usi.bat](../shogi-usi.bat) | Windows Python から USI エンジンを起動（Windows 未実行）。 |

## configs

学習・NNUE・自律ループの JSON 設定。tiny/ は接続確認用です。

| ファイル | 役割 |
| --- | --- |
| [configs/adversarial.json](../configs/adversarial.json) | 設定またはスキーマ（JSON）。 |
| [configs/alphazero.json](../configs/alphazero.json) | 設定またはスキーマ（JSON）。 |
| [configs/autoloop/hasami-debug.json](../configs/autoloop/hasami-debug.json) | 設定またはスキーマ（JSON）。 |
| [configs/autoloop/shogi-engine.json](../configs/autoloop/shogi-engine.json) | 設定またはスキーマ（JSON）。 |
| [configs/autoloop/shogi-nnue.json](../configs/autoloop/shogi-nnue.json) | 設定またはスキーマ（JSON）。 |
| [configs/autoloop/shogi.json](../configs/autoloop/shogi.json) | 設定またはスキーマ（JSON）。 |
| [configs/autoloop/tiny-hasami.json](../configs/autoloop/tiny-hasami.json) | 設定またはスキーマ（JSON）。 |
| [configs/autoloop/tiny-shogi-nnue.json](../configs/autoloop/tiny-shogi-nnue.json) | 設定またはスキーマ（JSON）。 |
| [configs/dqn.json](../configs/dqn.json) | 設定またはスキーマ（JSON）。 |
| [configs/nnue/shogi.json](../configs/nnue/shogi.json) | 設定またはスキーマ（JSON）。 |
| [configs/nnue/tiny-shogi.json](../configs/nnue/tiny-shogi.json) | 設定またはスキーマ（JSON）。 |
| [configs/poker-ppo-league.json](../configs/poker-ppo-league.json) | 設定またはスキーマ（JSON）。 |
| [configs/ppo.json](../configs/ppo.json) | 設定またはスキーマ（JSON）。 |
| [configs/reinforce.json](../configs/reinforce.json) | 設定またはスキーマ（JSON）。 |
| [configs/riichi-ppo-league-tonpuu.json](../configs/riichi-ppo-league-tonpuu.json) | 設定またはスキーマ（JSON）。 |
| [configs/riichi-ppo-league.json](../configs/riichi-ppo-league.json) | 設定またはスキーマ（JSON）。 |
| [configs/shogi-alphazero.json](../configs/shogi-alphazero.json) | 設定またはスキーマ（JSON）。 |
| [configs/supervised.json](../configs/supervised.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/adversarial.json](../configs/tiny/adversarial.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/alphazero-shogi.json](../configs/tiny/alphazero-shogi.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/alphazero.json](../configs/tiny/alphazero.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/dqn.json](../configs/tiny/dqn.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/leduc-ppo-league.json](../configs/tiny/leduc-ppo-league.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/poker-ppo-league.json](../configs/tiny/poker-ppo-league.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/ppo-riichi.json](../configs/tiny/ppo-riichi.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/ppo.json](../configs/tiny/ppo.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/practical-hasami.json](../configs/tiny/practical-hasami.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/practical-joint-hasami.json](../configs/tiny/practical-joint-hasami.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/reinforce.json](../configs/tiny/reinforce.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/riichi-ppo-league.json](../configs/tiny/riichi-ppo-league.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/supervised.json](../configs/tiny/supervised.json) | 設定またはスキーマ（JSON）。 |
| [configs/tiny/werewolf-ppo-league.json](../configs/tiny/werewolf-ppo-league.json) | 設定またはスキーマ（JSON）。 |
| [configs/werewolf-ppo-league.json](../configs/werewolf-ppo-league.json) | 設定またはスキーマ（JSON）。 |

## cpp

C++20 の規則、C ABI、MCTS、将棋探索と NNUE。CMakeLists.txt と各 include/ が入口です。

| ファイル | 役割 |
| --- | --- |
| [cpp/include/hasami/c_api.h](../cpp/include/hasami/c_api.h) | C++ 宣言 / C ABI。 |
| [cpp/include/hasami/game.hpp](../cpp/include/hasami/game.hpp) | C++ 宣言 / C ABI。 |
| [cpp/include/hasami/search.hpp](../cpp/include/hasami/search.hpp) | C++ 宣言 / C ABI。 |
| [cpp/mcts/include/mcts/c_api.h](../cpp/mcts/include/mcts/c_api.h) | C++ 宣言 / C ABI。 |
| [cpp/mcts/include/mcts/pool.hpp](../cpp/mcts/include/mcts/pool.hpp) | C++ 宣言 / C ABI。 |
| [cpp/mcts/include/mcts/search.hpp](../cpp/mcts/include/mcts/search.hpp) | C++ 宣言 / C ABI。 |
| [cpp/mcts/src/c_api.cpp](../cpp/mcts/src/c_api.cpp) | C++ 実装またはテスト。 |
| [cpp/mcts/tests/test_search.cpp](../cpp/mcts/tests/test_search.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/include/riichi/agari.hpp](../cpp/riichi/include/riichi/agari.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/c_api.h](../cpp/riichi/include/riichi/c_api.h) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/core.hpp](../cpp/riichi/include/riichi/core.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/encoding.hpp](../cpp/riichi/include/riichi/encoding.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/game.hpp](../cpp/riichi/include/riichi/game.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/rng.hpp](../cpp/riichi/include/riichi/rng.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/rules.hpp](../cpp/riichi/include/riichi/rules.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/scoring.hpp](../cpp/riichi/include/riichi/scoring.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/shanten.hpp](../cpp/riichi/include/riichi/shanten.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/tiles.hpp](../cpp/riichi/include/riichi/tiles.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/include/riichi/yaku.hpp](../cpp/riichi/include/riichi/yaku.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/src/agari.cpp](../cpp/riichi/src/agari.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/c_api_core.cpp](../cpp/riichi/src/c_api_core.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/c_api_game.cpp](../cpp/riichi/src/c_api_game.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/encoding.cpp](../cpp/riichi/src/encoding.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/game.cpp](../cpp/riichi/src/game.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/game_json.hpp](../cpp/riichi/src/game_json.hpp) | C++ 宣言 / C ABI。 |
| [cpp/riichi/src/rng.cpp](../cpp/riichi/src/rng.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/rules.cpp](../cpp/riichi/src/rules.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/scoring.cpp](../cpp/riichi/src/scoring.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/shanten.cpp](../cpp/riichi/src/shanten.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/tiles.cpp](../cpp/riichi/src/tiles.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/src/yaku.cpp](../cpp/riichi/src/yaku.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/tests/test_core.cpp](../cpp/riichi/tests/test_core.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/tests/test_game.cpp](../cpp/riichi/tests/test_game.cpp) | C++ 実装またはテスト。 |
| [cpp/riichi/tools/selfplay_bench.cpp](../cpp/riichi/tools/selfplay_bench.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/include/shogi/c_api.h](../cpp/shogi/include/shogi/c_api.h) | C++ 宣言 / C ABI。 |
| [cpp/shogi/include/shogi/game.hpp](../cpp/shogi/include/shogi/game.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi/src/c_api.cpp](../cpp/shogi/src/c_api.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/src/game.cpp](../cpp/shogi/src/game.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/src/mechanics.cpp](../cpp/shogi/src/mechanics.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/src/notation.cpp](../cpp/shogi/src/notation.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/tests/test_history.cpp](../cpp/shogi/tests/test_history.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/tests/test_perft.cpp](../cpp/shogi/tests/test_perft.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/tests/test_rules.cpp](../cpp/shogi/tests/test_rules.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi/tools/bench_shogi.cpp](../cpp/shogi/tools/bench_shogi.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/include/shogi_search/binary.hpp](../cpp/shogi_search/include/shogi_search/binary.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/include/shogi_search/c_api.h](../cpp/shogi_search/include/shogi_search/c_api.h) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/include/shogi_search/data.hpp](../cpp/shogi_search/include/shogi_search/data.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/include/shogi_search/nnue.hpp](../cpp/shogi_search/include/shogi_search/nnue.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/include/shogi_search/nnue_layout.hpp](../cpp/shogi_search/include/shogi_search/nnue_layout.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/include/shogi_search/position.hpp](../cpp/shogi_search/include/shogi_search/position.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/include/shogi_search/search.hpp](../cpp/shogi_search/include/shogi_search/search.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/src/binary.cpp](../cpp/shogi_search/src/binary.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/src/c_api.cpp](../cpp/shogi_search/src/c_api.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/src/data.cpp](../cpp/shogi_search/src/data.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/src/nnue.cpp](../cpp/shogi_search/src/nnue.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/src/position.cpp](../cpp/shogi_search/src/position.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/src/search.cpp](../cpp/shogi_search/src/search.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tests/data/mates-v1.tsv](../cpp/shogi_search/tests/data/mates-v1.tsv) | 補助データまたは利用者向け文書。 |
| [cpp/shogi_search/tests/data/parity-v1.tsv](../cpp/shogi_search/tests/data/parity-v1.tsv) | 補助データまたは利用者向け文書。 |
| [cpp/shogi_search/tests/mate_vectors.hpp](../cpp/shogi_search/tests/mate_vectors.hpp) | C++ 宣言 / C ABI。 |
| [cpp/shogi_search/tests/test_mates.cpp](../cpp/shogi_search/tests/test_mates.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tests/test_nnue.cpp](../cpp/shogi_search/tests/test_nnue.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tests/test_position.cpp](../cpp/shogi_search/tests/test_position.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tests/test_search.cpp](../cpp/shogi_search/tests/test_search.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tools/bench.cpp](../cpp/shogi_search/tools/bench.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tools/generate_mates.cpp](../cpp/shogi_search/tools/generate_mates.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tools/mates.cpp](../cpp/shogi_search/tools/mates.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tools/profile_nnue.cpp](../cpp/shogi_search/tools/profile_nnue.cpp) | C++ 実装またはテスト。 |
| [cpp/shogi_search/tools/selfplay.cpp](../cpp/shogi_search/tools/selfplay.cpp) | C++ 実装またはテスト。 |
| [cpp/src/c_api.cpp](../cpp/src/c_api.cpp) | C++ 実装またはテスト。 |
| [cpp/src/game.cpp](../cpp/src/game.cpp) | C++ 実装またはテスト。 |
| [cpp/src/main.cpp](../cpp/src/main.cpp) | C++ 実装またはテスト。 |
| [cpp/src/search.cpp](../cpp/src/search.cpp) | C++ 実装またはテスト。 |
| [cpp/tests/test_core.cpp](../cpp/tests/test_core.cpp) | C++ 実装またはテスト。 |

## docs

利用者向けの操作・学習・手法・実測範囲の説明。

| ファイル | 役割 |
| --- | --- |
| [docs/CUDA_WSL2_ja.md](../docs/CUDA_WSL2_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/HANDOFF_ja.md](../docs/HANDOFF_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/INSTALL_ja.md](../docs/INSTALL_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/METHOD_ja.md](../docs/METHOD_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/MODEL_CARDS/README.md](../docs/MODEL_CARDS/README.md) | 補助データまたは利用者向け文書。 |
| [docs/PROGRESS_ja.md](../docs/PROGRESS_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/RESULTS.md](../docs/RESULTS.md) | 補助データまたは利用者向け文書。 |
| [docs/SHOGIDOKORO_ja.md](../docs/SHOGIDOKORO_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/TRAINING_ja.md](../docs/TRAINING_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/USAGE_ja.md](../docs/USAGE_ja.md) | 補助データまたは利用者向け文書。 |
| [docs/images/hasami_01_start.png](../docs/images/hasami_01_start.png) | 補助データまたは利用者向け文書。 |
| [docs/images/hasami_02_hint.png](../docs/images/hasami_02_hint.png) | 補助データまたは利用者向け文書。 |
| [docs/images/hasami_03_trained_model.png](../docs/images/hasami_03_trained_model.png) | 補助データまたは利用者向け文書。 |
| [docs/images/hasami_training_progress.png](../docs/images/hasami_training_progress.png) | 補助データまたは利用者向け文書。 |
| [docs/images/nnue_vs_handwritten.png](../docs/images/nnue_vs_handwritten.png) | 補助データまたは利用者向け文書。 |
| [docs/images/shogi_01_start.png](../docs/images/shogi_01_start.png) | 補助データまたは利用者向け文書。 |
| [docs/images/shogi_02_select.png](../docs/images/shogi_02_select.png) | 補助データまたは利用者向け文書。 |
| [docs/images/shogi_04_engine_game.png](../docs/images/shogi_04_engine_game.png) | 補助データまたは利用者向け文書。 |
| [docs/images/shogi_05_eval_graph.png](../docs/images/shogi_05_eval_graph.png) | 補助データまたは利用者向け文書。 |

## examples

信頼済みプラグインと外部エージェント設定の例。

| ファイル | 役割 |
| --- | --- |
| [examples/__init__.py](../examples/__init__.py) | Runnable non-learning integration examples. |
| [examples/agents.example.json](../examples/agents.example.json) | 設定またはスキーマ（JSON）。 |
| [examples/agents.learned.json](../examples/agents.learned.json) | 設定またはスキーマ（JSON）。 |
| [examples/external_random.py](../examples/external_random.py) | A complete external agent example; no project imports are required. |
| [examples/plugin_agent.py](../examples/plugin_agent.py) | A trusted Python agent plugin demonstrating the factory contract. |

## hasami

はさみ将棋の規則・履歴・Python/C++ 接続・Tk GUI。engine.py から読みます。

| ファイル | 役割 |
| --- | --- |
| [hasami/__init__.py](../hasami/__init__.py) | Hasami Lab: deterministic game rules and pluggable agents. |
| [hasami/agents.py](../hasami/agents.py) | Classical baselines, trusted plugins, and a language-neutral process adapter. |
| [hasami/arena.py](../hasami/arena.py) | Run reproducible evaluation games without training any model. |
| [hasami/engine.py](../hasami/engine.py) | Independent Python rules engine with bitboards and exact history. |
| [hasami/env.py](../hasami/env.py) | A dependency-free turn-based environment, not a Gymnasium API claim. |
| [hasami/gui.py](../hasami/gui.py) | Playable Tk GUI. Only the main thread may access Tk objects. |
| [hasami/gui_support.py](../hasami/gui_support.py) | Display-free helpers used by the Tk GUI (importable without tkinter). |
| [hasami/model.py](../hasami/model.py) | Shared value objects, validation, and stable wire conventions. |
| [hasami/native.py](../hasami/native.py) | ctypes bridge to the C++ engine. No native rules are reimplemented here. |
| [hasami/observation.py](../hasami/observation.py) | Public observations for agents; no learning framework is imported. |
| [hasami/records.py](../hasami/records.py) | Versioned JSON records, replay validation, and atomic saves. |

## hasami_learn

ゲームアダプター、自己対戦、学習、評価、推論、運用。cli.py と games/ から読みます。

| ファイル | 役割 |
| --- | --- |
| [hasami_learn/__init__.py](../hasami_learn/__init__.py) | Optional learning package for Hasami Lab. |
| [hasami_learn/algorithms/__init__.py](../hasami_learn/algorithms/__init__.py) | Import every algorithm module so they register with ALGORITHMS. |
| [hasami_learn/algorithms/adversarial.py](../hasami_learn/algorithms/adversarial.py) | Legal-policy league training around a learner's actor and ingestion hooks. |
| [hasami_learn/algorithms/alphazero.py](../hasami_learn/algorithms/alphazero.py) | AlphaZero self-play learner using batched PUCT and policy/value replay. |
| [hasami_learn/algorithms/base.py](../hasami_learn/algorithms/base.py) | Learner contract shared by every algorithm (see docs/TRAINING_ja.md). |
| [hasami_learn/algorithms/dqn.py](../hasami_learn/algorithms/dqn.py) | One-step Double DQN over learner decisions with opponent replies folded in. |
| [hasami_learn/algorithms/ismcts.py](../hasami_learn/algorithms/ismcts.py) | Single-observer information-set UCT with fresh belief samples per simulation. |
| [hasami_learn/algorithms/mcts.py](../hasami_learn/algorithms/mcts.py) | Compatibility alias: the PUCT search moved to ``hasami_learn.search.mcts``. |
| [hasami_learn/algorithms/ppo.py](../hasami_learn/algorithms/ppo.py) | On-policy PPO with terminal-reward GAE over opponent-pool games. |
| [hasami_learn/algorithms/ppo_league.py](../hasami_learn/algorithms/ppo_league.py) | Shared N-seat PPO, duplicate-deal PFSP and isolated archetype exploiters. |
| [hasami_learn/algorithms/practical.py](../hasami_learn/algorithms/practical.py) | Masked practical-score and slip learning from calibrated proxy continuations. |
| [hasami_learn/algorithms/reinforce.py](../hasami_learn/algorithms/reinforce.py) | Reference learner: REINFORCE with a value baseline over self-play games. |
| [hasami_learn/algorithms/supervised.py](../hasami_learn/algorithms/supervised.py) | Imitation learning from validated records with legal policy and value targets. |
| [hasami_learn/autoloop/__init__.py](../hasami_learn/autoloop/__init__.py) | Standard-library-only supervision of resumable learning subprocesses. |
| [hasami_learn/autoloop/__main__.py](../hasami_learn/autoloop/__main__.py) | CLI for local, resumable subprocess plans. |
| [hasami_learn/autoloop/evaluate.py](../hasami_learn/autoloop/evaluate.py) | Resolve the evaluation device inside the stage and run the existing CLI. |
| [hasami_learn/autoloop/evidence.py](../hasami_learn/autoloop/evidence.py) | Standard-library validation of the shared release evidence contracts. |
| [hasami_learn/autoloop/health.py](../hasami_learn/autoloop/health.py) | Injectable clocks, suspend detection, disk guard, and isolated device probes. |
| [hasami_learn/autoloop/local_stage.py](../hasami_learn/autoloop/local_stage.py) | Run bounded gate/export work in a supervised standard-library subprocess. |
| [hasami_learn/autoloop/plan.py](../hasami_learn/autoloop/plan.py) | Plan validation and production evidence gate, without learning imports. |
| [hasami_learn/autoloop/runner.py](../hasami_learn/autoloop/runner.py) | Persistent training plan controller with injectable external effects. |
| [hasami_learn/autoloop/stages.py](../hasami_learn/autoloop/stages.py) | Stage commands and verified, idempotent local artifacts. |
| [hasami_learn/autoloop/state.py](../hasami_learn/autoloop/state.py) | Durable JSON, packaged schema checks, and kernel-owned instance locks. |
| [hasami_learn/autoloop/worker.py](../hasami_learn/autoloop/worker.py) | Keep the loop lock until its stage stops, including after parent SIGKILL. |
| [hasami_learn/auxiliary.py](../hasami_learn/auxiliary.py) | Hand-end supervision; unrevealed labels are absent, never negative examples. |
| [hasami_learn/batched_selfplay.py](../hasami_learn/batched_selfplay.py) | Multi-table self-play with shared, fixed-shape batched evaluation (docs/METHOD_ja.md#methods). |
| [hasami_learn/catalog.py](../hasami_learn/catalog.py) | Catalogue of every registry, for the training CLI. |
| [hasami_learn/checkpoint_bundle.py](../hasami_learn/checkpoint_bundle.py) | Self-contained copies of checkpoint directories for promotion, export and release assets. |
| [hasami_learn/cli.py](../hasami_learn/cli.py) | Command line entry: python -m hasami_learn.cli {train,eval,devices,list,dataset,export-agents-config,config-template}. |
| [hasami_learn/config.py](../hasami_learn/config.py) | Training configuration dataclasses validated against hasami_learn/schemas/training-config.schema.json. |
| [hasami_learn/datasets.py](../hasami_learn/datasets.py) | Validated game records as dense policy/value imitation datasets. |
| [hasami_learn/datastore.py](../hasami_learn/datastore.py) | Single-writer, capped self-play archives with an append-only recovery journal. |
| [hasami_learn/devices.py](../hasami_learn/devices.py) | Device selection for CPU, Apple Silicon (MPS) and NVIDIA (CUDA). |
| [hasami_learn/diagnostics.py](../hasami_learn/diagnostics.py) | Read-only, bounded-memory diagnostics for two-player AlphaZero checkpoints. |
| [hasami_learn/encoding.py](../hasami_learn/encoding.py) | Generic policy helpers and legacy Hasami encoder exports. |
| [hasami_learn/evaluation.py](../hasami_learn/evaluation.py) | Evaluation matches against classical baselines and pool members. |
| [hasami_learn/games/README.md](../hasami_learn/games/README.md) | 補助データまたは利用者向け文書。 |
| [hasami_learn/games/__init__.py](../hasami_learn/games/__init__.py) | Explicitly registered game adapters; never discover or import external plugins. |
| [hasami_learn/games/base.py](../hasami_learn/games/base.py) | Contracts for player observations, joint decisions, and game adapters. |
| [hasami_learn/games/hasami.py](../hasami_learn/games/hasami.py) | Hasami adapter: engine, encoding, symmetries, and classical baselines. |
| [hasami_learn/games/poker.py](../hasami_learn/games/poker.py) | GameSpec v2 for heads-up limit Hold'em and Leduc verification. |
| [hasami_learn/games/records.py](../hasami_learn/games/records.py) | Full-history records for game adapters, with optional engine format hooks. |
| [hasami_learn/games/riichi.py](../hasami_learn/games/riichi.py) | GameSpec v2 adapter for seeded four-player Riichi Mahjong. |
| [hasami_learn/games/shogi.py](../hasami_learn/games/shogi.py) | GameSpec adapter for standard shogi (hasami_learn/games/shogi.py). |
| [hasami_learn/games/werewolf.py](../hasami_learn/games/werewolf.py) | GameSpec v2 adapter and observation-only werewolf-enc-v1 planes. |
| [hasami_learn/history.py](../hasami_learn/history.py) | Stateless history planes stacked on a base encoder (for example ``riichi-history``). |
| [hasami_learn/inference.py](../hasami_learn/inference.py) | Inference agents that satisfy hasami.agents.Agent, plus the agents-config factory. |
| [hasami_learn/ladder.py](../hasami_learn/ladder.py) | Side-swapped round robin with Wilson score intervals and Bradley-Terry Elo. |
| [hasami_learn/manifest.py](../hasami_learn/manifest.py) | Model manifest and checkpoint layout (see hasami_learn/schemas/model-manifest.schema.json). |
| [hasami_learn/metrics.py](../hasami_learn/metrics.py) | Append-only JSONL metrics following hasami_learn/metrics.py. |
| [hasami_learn/networks/__init__.py](../hasami_learn/networks/__init__.py) | Network factories registered under hasami_learn.registry.NETWORKS. |
| [hasami_learn/networks/base.py](../hasami_learn/networks/base.py) | Shared head modules and conventions for policy-value and Q networks. |
| [hasami_learn/networks/mlp.py](../hasami_learn/networks/mlp.py) | Small fully-connected network; fast enough for CPU smoke tests. |
| [hasami_learn/networks/resnet.py](../hasami_learn/networks/resnet.py) | AlphaZero-style residual tower for the game board. |
| [hasami_learn/networks/tilenet.py](../hasami_learn/networks/tilenet.py) | Conv1d features over tiles, retaining observer-relative rows and tile identity. |
| [hasami_learn/nnue/__init__.py](../hasami_learn/nnue/__init__.py) | Original MIT NNUE features, portable formats and optional PyTorch training. |
| [hasami_learn/nnue/control.py](../hasami_learn/nnue/control.py) | Standard-library configuration and durable NNUE control records. |
| [hasami_learn/nnue/data.py](../hasami_learn/nnue/data.py) | Streaming hlnn1 shards; fixed little-endian records and verified JSON sidecars. |
| [hasami_learn/nnue/distill.py](../hasami_learn/nnue/distill.py) | Add family-Z values in batches, replaying the generator's complete histories. |
| [hasami_learn/nnue/export.py](../hasami_learn/nnue/export.py) | hlnnw1 float32 weights; no pickle, executable code or external dependencies. |
| [hasami_learn/nnue/features.py](../hasami_learn/nnue/features.py) | hlnn-kp-v1; layout.py is the shared source for the generated C++ table. |
| [hasami_learn/nnue/generate.py](../hasami_learn/nnue/generate.py) | Run and verify complete-game native self-play without importing torch. |
| [hasami_learn/nnue/layout.py](../hasami_learn/nnue/layout.py) | Canonical hlnn-kp-v1 description shared with the generated C++ table. |
| [hasami_learn/nnue/loader.py](../hasami_learn/nnue/loader.py) | Bounded shuffled hlnn1 streaming with native batch feature extraction. |
| [hasami_learn/nnue/loop.py](../hasami_learn/nnue/loop.py) | One resumable family-N generation, run as an existing autoloop module stage. |
| [hasami_learn/nnue/match.py](../hasami_learn/nnue/match.py) | Seeded, colour-paired native evaluator matches with effective-sample evidence. |
| [hasami_learn/nnue/model.py](../hasami_learn/nnue/model.py) | Sparse shared first layer and deterministic float32 NNUE reference forward. |
| [hasami_learn/nnue/train.py](../hasami_learn/nnue/train.py) | Deterministic streaming NNUE training on hlnn1 files, with atomic CPU-safe resume. |
| [hasami_learn/nnue/window.py](../hasami_learn/nnue/window.py) | Immutable cumulative training selections; source shards are never deleted here. |
| [hasami_learn/opponents.py](../hasami_learn/opponents.py) | Opponent pool for adversarial / league style training. |
| [hasami_learn/parallel.py](../hasami_learn/parallel.py) | Spawned, deterministic MCTS self-play with numpy-only trajectory results. |
| [hasami_learn/practical.py](../hasami_learn/practical.py) | Level-conditioned proxies and practical advice, with no training dependencies. |
| [hasami_learn/practical_calibrate.py](../hasami_learn/practical_calibrate.py) | Calibrate internal proxy levels with side-swapped games and Elo intervals. |
| [hasami_learn/practical_eval.py](../hasami_learn/practical_eval.py) | Reproducible E1-E8 proxy experiments; tiny reports are explicitly smoke-only. |
| [hasami_learn/practical_general.py](../hasami_learn/practical_general.py) | Sampled expectimax for GameSpec v2, without training-time dependencies. |
| [hasami_learn/practical_search.py](../hasami_learn/practical_search.py) | Method A: batched, history-preserving expectimax over a frozen engine. |
| [hasami_learn/progress.py](../hasami_learn/progress.py) | Read-only, standard-library training progress reports. |
| [hasami_learn/registry.py](../hasami_learn/registry.py) | Named registries so every agent-related component can be swapped by config. |
| [hasami_learn/replay.py](../hasami_learn/replay.py) | Fixed-capacity numpy ring buffer with npz persistence. |
| [hasami_learn/runlock.py](../hasami_learn/runlock.py) | Standard-library writer locks and narrowly scoped staging-file cleanup. |
| [hasami_learn/schema_validation.py](../hasami_learn/schema_validation.py) | Validate packaged JSON contracts, with a dependency-free subset fallback. |
| [hasami_learn/schemas/autoloop-config.schema.json](../hasami_learn/schemas/autoloop-config.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/autoloop-state.schema.json](../hasami_learn/schemas/autoloop-state.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/evaluation-evidence.schema.json](../hasami_learn/schemas/evaluation-evidence.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/model-manifest.schema.json](../hasami_learn/schemas/model-manifest.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/nnue-config.schema.json](../hasami_learn/schemas/nnue-config.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/practical-eval.schema.json](../hasami_learn/schemas/practical-eval.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/practical-levels.schema.json](../hasami_learn/schemas/practical-levels.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/release-candidate.schema.json](../hasami_learn/schemas/release-candidate.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/training-config.schema.json](../hasami_learn/schemas/training-config.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/schemas/training-metrics.schema.json](../hasami_learn/schemas/training-metrics.schema.json) | 設定またはスキーマ（JSON）。 |
| [hasami_learn/search/__init__.py](../hasami_learn/search/__init__.py) | Search used at play time (PUCT MCTS). Lives outside ``hasami_learn.algorithms`` so that runtime code |
| [hasami_learn/search/draw.py](../hasami_learn/search/draw.py) | Shared draw utility for search leaves and training targets; no rule changes. |
| [hasami_learn/search/mcts.py](../hasami_learn/search/mcts.py) | History-preserving PUCT/Gumbel search with batched inference across games. |
| [hasami_learn/search/native_mcts.py](../hasami_learn/search/native_mcts.py) | C++ PUCT with one Python evaluator and NumPy-owned Dirichlet draws. |
| [hasami_learn/selfplay.py](../hasami_learn/selfplay.py) | Episode runners shared by all learners. Rules stay inside the selected game adapter. |
| [hasami_learn/signature_cache.py](../hasami_learn/signature_cache.py) | Cached ``inspect.signature(...).bind`` checks for callables invoked once per decision (~30 us each uncached). |
| [hasami_learn/trainer.py](../hasami_learn/trainer.py) | Training loop: config -> learner -> iterations -> checkpoints -> evaluation -> resume. |

## models

公開モデルの索引と検証付き取得方法。重みはソースに同梱しません。

| ファイル | 役割 |
| --- | --- |
| [models/index.json](../models/index.json) | 設定またはスキーマ（JSON）。 |

## poker

ポーカーの規則・観測・評価。GUI はありません。

| ファイル | 役割 |
| --- | --- |
| [poker/__init__.py](../poker/__init__.py) | Dependency-free heads-up limit Hold'em and the Leduc verification variant. |
| [poker/baselines.py](../poker/baselines.py) | Observation-only baselines and deterministic equity estimates. |
| [poker/encoding.py](../poker/encoding.py) | Information-set planes as plain Python floats (no array dependency). |
| [poker/evaluation.py](../poker/evaluation.py) | Paired duplicate-deal head-to-head evaluation in milli-big-blinds per hand. |
| [poker/evaluator.py](../poker/evaluator.py) | Seven-card evaluation without tables or third-party dependencies. |
| [poker/game.py](../poker/game.py) | Seeded heads-up limit Hold'em and Leduc with immutable undo snapshots. |
| [poker/leduc.py](../poker/leduc.py) | Exact Leduc best responses constrained to the responder's information sets. |
| [poker/records.py](../poker/records.py) | Versioned, seed-replayed JSON records; sampled worlds are not seed-replayable. |

## riichi

リーチ麻雀の規則・得点・C++ 接続。game.py と cli.py が入口です。

| ファイル | 役割 |
| --- | --- |
| [riichi/__init__.py](../riichi/__init__.py) | Dependency-free four-player Riichi hand analysis and scoring. |
| [riichi/actions.py](../riichi/actions.py) | The fixed 111-action table for riichi-4p-v1. |
| [riichi/agari.py](../riichi/agari.py) | Enumerate all structural decompositions and winning-tile assignments. |
| [riichi/baselines.py](../riichi/baselines.py) | Information-set-only random, tsumogiri, and shanten-greedy baselines. |
| [riichi/cli.py](../riichi/cli.py) | Small command-line score calculator for a concealed hand before its win. |
| [riichi/encoding.py](../riichi/encoding.py) | Information-set encoding riichi-enc-v1: 120 planes of shape (4, 34). |
| [riichi/game.py](../riichi/game.py) | Seeded four-player hand and match state machine for riichi-4p-v1. |
| [riichi/native.py](../riichi/native.py) | Dependency-free ctypes bridge to the independent native riichi engine. |
| [riichi/rules.py](../riichi/rules.py) | Versioned rule configuration; game-phase flags are consumed by the later engine. |
| [riichi/scoring.py](../riichi/scoring.py) | Fu, limit points, four-player payments, and placement settlement. |
| [riichi/shanten.py](../riichi/shanten.py) | Suit-table dynamic programming for standard, seven-pairs, and orphan shanten. |
| [riichi/tiles.py](../riichi/tiles.py) | Physical tile IDs, type counts, notation, and reproducible wall layout. |
| [riichi/yaku.py](../riichi/yaku.py) | Pattern yaku, contextual yaku, yakuman variants, and bonus-tile counting. |

## scripts

ビルド、計測、GUI/USI の確認、学習の起動と監視。下のファイル一覧を参照。

| ファイル | 役割 |
| --- | --- |
| [scripts/autoloop/hasami-autoloop@.service](../scripts/autoloop/hasami-autoloop@.service) | 自律ループを systemd user service で監督するテンプレート。 |
| [scripts/autoloop/install.py](../scripts/autoloop/install.py) | Render user service launchers and optionally install them on WSL2. |
| [scripts/autoloop/install_wsl.sh](../scripts/autoloop/install_wsl.sh) | WSL の user service 導入用ラッパー（dry-run 対応）。 |
| [scripts/autoloop/uninstall_wsl.sh](../scripts/autoloop/uninstall_wsl.sh) | 該当する user service を止めて解除します。学習データは残します。 |
| [scripts/benchmark.py](../scripts/benchmark.py) | Measure a fixed perft workload, not playing strength or bitboard-only speedup. |
| [scripts/benchmark_collection.py](../scripts/benchmark_collection.py) | End-to-end league collection throughput (docs/METHOD_ja.md#methods section 4). |
| [scripts/benchmark_decisions.py](../scripts/benchmark_decisions.py) | Isolated decision latency and evaluator throughput per device (docs/METHOD_ja.md#methods section 4). |
| [scripts/benchmark_mcts.py](../scripts/benchmark_mcts.py) | Compare Python/native PUCT with identical CPU/CUDA evaluation batches. |
| [scripts/benchmark_nnue.py](../scripts/benchmark_nnue.py) | Bounded single-thread NNUE learning sanity experiment; not the NNUE generation generator. |
| [scripts/benchmark_riichi.py](../scripts/benchmark_riichi.py) | Measure complete random matches through both adapters and the C++ tool. |
| [scripts/benchmark_shogi.py](../scripts/benchmark_shogi.py) | Compare identical random playouts through Python, ctypes, and the C++ core. |
| [scripts/benchmark_shogi_search.py](../scripts/benchmark_shogi_search.py) | CPU-only Family N measurements and optional paired-opening baseline matches. |
| [scripts/build_native.py](../scripts/build_native.py) | Configure, build, and test the portable C++ backend. |
| [scripts/download_models.py](../scripts/download_models.py) | Explicit model download command using only the standard library. |
| [scripts/gui_smoke.py](../scripts/gui_smoke.py) | Exercise real Tk events. On headless Linux, run this under xvfb-run. |
| [scripts/gui_smoke_shogi.py](../scripts/gui_smoke_shogi.py) | Exercise real Tk events on macOS or under xvfb-run on Linux. |
| [scripts/inference_smoke.py](../scripts/inference_smoke.py) | Check one installed model on its initial game without mutating the game. |
| [scripts/kill_resume_smoke.py](../scripts/kill_resume_smoke.py) | Process-kill resume test, and the verification step of a manual power-loss test. |
| [scripts/pre-push](../scripts/pre-push) | 所有者が承認した SHA と main の更新だけを許すローカルガード。 |
| [scripts/train_supervisor.sh](../scripts/train_supervisor.sh) | 学習を起動し、未完了なら保存済み状態から再起動します。 |
| [scripts/usi_smoke.py](../scripts/usi_smoke.py) | Exercise a real USI subprocess and check replies against the Python shogi engine. |
| [scripts/validate_training.sh](../scripts/validate_training.sh) | 本番学習前の native / 両 Python / CUDA / 再開の検証と記録。 |
| [scripts/validation_env.py](../scripts/validation_env.py) | Print a sanitized description of the validation machine as JSON (no host, user, IP or absolute paths). |
| [scripts/validation_summary.py](../scripts/validation_summary.py) | Sanitize a validate_training.sh result directory, extract key results, decide the gate, and package it. |
| [scripts/windows/install_tasks.ps1](../scripts/windows/install_tasks.ps1) | Windows ログオン・復帰タスクの登録（未実行）。 |
| [scripts/windows/keep_awake.ps1](../scripts/windows/keep_awake.ps1) | 指定時に稼働中だけ Windows の自動スリープを抑制（未実行）。 |
| [scripts/windows/resume.ps1](../scripts/windows/resume.ps1) | WSL の復帰マーカーと再開監視（未実行）。 |
| [scripts/windows/uninstall_tasks.ps1](../scripts/windows/uninstall_tasks.ps1) | Windows タスクの解除（未実行）。 |

## shogi

本将棋の規則・棋譜・探索接続・GUI・USI。engine.py、native_search.py、usi.py が入口です。

| ファイル | 役割 |
| --- | --- |
| [shogi/__init__.py](../shogi/__init__.py) | Standard-library shogi reference engine (shogi-std-v1). |
| [shogi/__main__.py](../shogi/__main__.py) | Run the standard shogi command-line tools. |
| [shogi/agents.py](../shogi/agents.py) | Classical legal baselines using the common Agent decision/limit protocol. |
| [shogi/cli.py](../shogi/cli.py) | Command-line inspection, perft and seeded random-play tools. |
| [shogi/encoding.py](../shogi/encoding.py) | Side-to-move shogi-enc-v1 planes and injective legal policy labels. |
| [shogi/engine.py](../shogi/engine.py) | History-complete deterministic shogi reference engine. |
| [shogi/gui.py](../shogi/gui.py) | Japanese shogi workbench; Tk stays on the main thread and agents receive clones. |
| [shogi/gui_support.py](../shogi/gui_support.py) | Tk-free input transitions and presentation; legality belongs to the engine. |
| [shogi/mechanics.py](../shogi/mechanics.py) | Movement tables and king-safe legal generation, independent of game history. |
| [shogi/model.py](../shogi/model.py) | Immutable values and absolute move codes for shogi-std-v1. |
| [shogi/native.py](../shogi/native.py) | Dependency-free ctypes bridge to the independent C++ shogi engine. |
| [shogi/native_search.py](../shogi/native_search.py) | Standard-library bridge to the project's MIT C++ alpha-beta searcher. |
| [shogi/notation.py](../shogi/notation.py) | Validated SFEN, USI, CSA and Japanese move text. |
| [shogi/perft.py](../shogi/perft.py) | Conventional move-tree counts, independent of repetition and move-limit draws. |
| [shogi/records.py](../shogi/records.py) | Validated JSON, KIF and CSA records, replayed exclusively by the engine. |
| [shogi/usi.py](../shogi/usi.py) | Line-flushed USI adapter with cancellable snapshot workers and deadline replies. |

## tests

公開ファイルだけで実行する規則・接続・学習テスト。unittest discover で実行します。

| ファイル | 役割 |
| --- | --- |
| [tests/__init__.py](../tests/__init__.py) | Rules, engine parity, protocol, and integration tests. |
| [tests/conformance/__init__.py](../tests/conformance/__init__.py) | Python entry point or supporting module. |
| [tests/conformance/_agents_support.py](../tests/conformance/_agents_support.py) | Independent fixtures for agent conformance, with no shared worker helpers. |
| [tests/conformance/_support.py](../tests/conformance/_support.py) | Shared deterministic fixtures and thin adapters to the independent oracle. |
| [tests/conformance/test_agents.py](../tests/conformance/test_agents.py) | A1-A8 conformance for classical, configured external, and learned agents. |
| [tests/conformance/test_engine_rules.py](../tests/conformance/test_engine_rules.py) | E1-E12 conformance checks for the hasami-lab-v1 engine contract. |
| [tests/conformance/test_gui_rules.py](../tests/conformance/test_gui_rules.py) | G1-G11 conformance through real Tk events; G9 needs no display. |
| [tests/conformance/test_riichi_rules.py](../tests/conformance/test_riichi_rules.py) | Seeded state-machine properties and GameSpec v2 information boundaries. |
| [tests/learn/__init__.py](../tests/learn/__init__.py) | Python entry point or supporting module. |
| [tests/learn/_common.py](../tests/learn/_common.py) | Shared helpers for hasami_learn tests; everything skips when torch is missing. |
| [tests/learn/autoloop_fakes/controller.py](../tests/learn/autoloop_fakes/controller.py) | Real controller process with a fake command adapter; no production hooks. |
| [tests/learn/autoloop_fakes/stage.py](../tests/learn/autoloop_fakes/stage.py) | Small idempotent subprocess used by the autoloop crash tests. |
| [tests/learn/test_adversarial.py](../tests/learn/test_adversarial.py) | League collection, exploiter isolation, and checkpoint regression tests. |
| [tests/learn/test_alphazero.py](../tests/learn/test_alphazero.py) | AlphaZero training, replay targets, and checkpoint inference regressions. |
| [tests/learn/test_autoloop.py](../tests/learn/test_autoloop.py) | Autoloop section 7: deterministic failures and real process recovery. |
| [tests/learn/test_batched_selfplay.py](../tests/learn/test_batched_selfplay.py) | batched collection: batched multi-table runner, fixed-shape evaluator and seats (docs/METHOD_ja.md#methods). |
| [tests/learn/test_checkpoint_bundle.py](../tests/learn/test_checkpoint_bundle.py) | Self-contained checkpoint bundles (standard library only; no torch needed). |
| [tests/learn/test_datastore.py](../tests/learn/test_datastore.py) | Python entry point or supporting module. |
| [tests/learn/test_decision_contracts.py](../tests/learn/test_decision_contracts.py) | Contracts prepared for batched collection: seat-signature caching, the shared decision rule, self-play agents. |
| [tests/learn/test_diagnostics.py](../tests/learn/test_diagnostics.py) | Independent diagnostics, corruption detection, and read-only CPU integration. |
| [tests/learn/test_dqn.py](../tests/learn/test_dqn.py) | DQN regression tests, optional when torch is unavailable. |
| [tests/learn/test_draw_contempt.py](../tests/learn/test_draw_contempt.py) | Search/target agreement for symmetric draw contempt and collection schedules. |
| [tests/learn/test_eval_openings.py](../tests/learn/test_eval_openings.py) | Seeded evaluation openings, reproducible search and effective sample regression tests. |
| [tests/learn/test_evaluation.py](../tests/learn/test_evaluation.py) | Evaluation statistics, engine end reasons, and reproducible ladder tests. |
| [tests/learn/test_game_spec.py](../tests/learn/test_game_spec.py) | Game adapters preserve Hasami contracts and support independent toy engines. |
| [tests/learn/test_gumbel_search.py](../tests/learn/test_gumbel_search.py) | Hand-computed Gumbel search and frozen default-search regressions. |
| [tests/learn/test_league_collection_parity.py](../tests/learn/test_league_collection_parity.py) | batched collection acceptance 5.1: reference and batched league collection produce identical data (exact, on CPU). |
| [tests/learn/test_mcts.py](../tests/learn/test_mcts.py) | PUCT legality, history, batching, and resource-lifetime regressions. |
| [tests/learn/test_multiplayer.py](../tests/learn/test_multiplayer.py) | Executable GameSpec v2 contracts, including simultaneous hidden decisions. |
| [tests/learn/test_native_mcts.py](../tests/learn/test_native_mcts.py) | Native PUCT parity, ownership, and training integration regressions. |
| [tests/learn/test_nnue.py](../tests/learn/test_nnue.py) | NNUE cross-language contracts and deterministic learning regressions. |
| [tests/learn/test_nnue_train.py](../tests/learn/test_nnue_train.py) | Vectorised training, native streaming, selection and checkpoint regressions. |
| [tests/learn/test_parallel.py](../tests/learn/test_parallel.py) | Spawned self-play legality, deterministic ordering, and resource cleanup. |
| [tests/learn/test_poker_adapter.py](../tests/learn/test_poker_adapter.py) | Poker GameSpec, observer privacy, PPO league checkpoints and method-A advice. |
| [tests/learn/test_ppo.py](../tests/learn/test_ppo.py) | PPO rollout, objective, and checkpoint regression tests. |
| [tests/learn/test_practical.py](../tests/learn/test_practical.py) | Practical play contract, arithmetic, training and E1-E7 CPU smoke tests. |
| [tests/learn/test_practical_compare.py](../tests/learn/test_practical_compare.py) | Joint practical learning and matched E8 comparison regression tests. |
| [tests/learn/test_practical_general.py](../tests/learn/test_practical_general.py) | General practical search: seat perspective, hidden states, chance and ownership. |
| [tests/learn/test_practical_search.py](../tests/learn/test_practical_search.py) | Method A arithmetic, ownership, runtime imports and checkpoint smoke coverage. |
| [tests/learn/test_progress.py](../tests/learn/test_progress.py) | Reporter crash recovery stays independent of training and optional packages. |
| [tests/learn/test_replay_memory.py](../tests/learn/test_replay_memory.py) | Replay storage footprints and startup memory diagnostics. |
| [tests/learn/test_resilience.py](../tests/learn/test_resilience.py) | Regression tests for checkpoint publication, recovery, and graceful stops. |
| [tests/learn/test_resume_cleanup.py](../tests/learn/test_resume_cleanup.py) | Crash recovery must clean only owned staging files under an exclusive writer lock. |
| [tests/learn/test_riichi_contracts.py](../tests/learn/test_riichi_contracts.py) | Contracts prepared for riichi learning: riichi-history encoder, auxiliary heads, Decision.extra, hand-end step info. |
| [tests/learn/test_riichi_determinize.py](../tests/learn/test_riichi_determinize.py) | The riichi adapter's determinize hook (practical-play spec section 11). |
| [tests/learn/test_riichi_learning.py](../tests/learn/test_riichi_learning.py) | League, belief supervision and information-set search regression tests. |
| [tests/learn/test_riichi_native_backend.py](../tests/learn/test_riichi_native_backend.py) | Native adapter fast paths, cross-backend records, search, and PPO league smoke. |
| [tests/learn/test_schema_subset.py](../tests/learn/test_schema_subset.py) | Compare the dependency-free validator with Draft 2020-12 on packaged contracts. |
| [tests/learn/test_shogi_adapter.py](../tests/learn/test_shogi_adapter.py) | Standard shogi through the learning stack: GameSpec adapter, encoder, records, search and a tiny AlphaZero run. |
| [tests/learn/test_supervised.py](../tests/learn/test_supervised.py) | Record datasets and supervised learner integration tests. |
| [tests/learn/test_warm_start.py](../tests/learn/test_warm_start.py) | Weights-only initialization and checkpoint lineage contracts. |
| [tests/learn/test_werewolf_adapter.py](../tests/learn/test_werewolf_adapter.py) | Five-seat adapter, hidden-state constraints, training, and method-A integration. |
| [tests/oracle.py](../tests/oracle.py) | Slow array/set oracle, intentionally independent of bitboard operations. |
| [tests/poker_suite/__init__.py](../tests/poker_suite/__init__.py) | Dependency-free poker conformance tests. |
| [tests/poker_suite/leduc_oracle.py](../tests/poker_suite/leduc_oracle.py) | Independent rational-arithmetic Leduc best response for call/uniform opponents. |
| [tests/poker_suite/test_evaluator.py](../tests/poker_suite/test_evaluator.py) | Independent five-card reference and category boundary vectors. |
| [tests/poker_suite/test_game.py](../tests/poker_suite/test_game.py) | Betting, chance, hidden information and history contracts. |
| [tests/poker_suite/test_hidden_records.py](../tests/poker_suite/test_hidden_records.py) | Information-set consistency, seeded records and engine dependency boundaries. |
| [tests/poker_suite/test_leduc_evaluation.py](../tests/poker_suite/test_leduc_evaluation.py) | Leduc rule vectors, exact exploitability and duplicate-pair arithmetic. |
| [tests/riichi/__init__.py](../tests/riichi/__init__.py) | Riichi tests, including compatibility with discovery without an explicit top. |
| [tests/riichi/test_core.py](../tests/riichi/test_core.py) | Tile, distance, decomposition, and rule contract tests. |
| [tests/riichi/test_crosscheck.py](../tests/riichi/test_crosscheck.py) | Deterministic generative checks and an optional independent scoring reference. |
| [tests/riichi/test_game.py](../tests/riichi/test_game.py) | State-machine regressions for the four-player game. |
| [tests/riichi/test_hand_end.py](../tests/riichi/test_hand_end.py) | Hand-end revelations (the supervision for belief heads) and the current-hand move index. |
| [tests/riichi/test_native_core.py](../tests/riichi/test_native_core.py) | Deterministic native/reference parity, without optional dependencies. |
| [tests/riichi/test_native_parity.py](../tests/riichi/test_native_parity.py) | Bit-exact game, observation, encoding, replay, and rejection parity. |
| [tests/riichi/test_redeal.py](../tests/riichi/test_redeal.py) | RiichiGame.redeal: the hidden-state replacement contract shared by both engine backends. |
| [tests/riichi/test_scoring.py](../tests/riichi/test_scoring.py) | Independent score tables and winning-tile regression fixtures. |
| [tests/riichi/test_yaku.py](../tests/riichi/test_yaku.py) | Named yaku table with open/closed variants and negative boundary cases. |
| [tests/shogi_suite/__init__.py](../tests/shogi_suite/__init__.py) | Standard shogi reference-engine conformance tests. |
| [tests/shogi_suite/support.py](../tests/shogi_suite/support.py) | Small explicit positions and snapshot helpers for public API tests. |
| [tests/shogi_suite/test_agents.py](../tests/shogi_suite/test_agents.py) | Legal decisions, deterministic completed search, budgets and clone ownership. |
| [tests/shogi_suite/test_cli.py](../tests/shogi_suite/test_cli.py) | Exercise the public module entry points in a fresh interpreter. |
| [tests/shogi_suite/test_encoding.py](../tests/shogi_suite/test_encoding.py) | Every oracle position and random trajectories exercise the policy bijection. |
| [tests/shogi_suite/test_gui_lifecycle.py](../tests/shogi_suite/test_gui_lifecycle.py) | Worker ownership tests and opt-in, process-isolated real Tk lifecycles. |
| [tests/shogi_suite/test_gui_smoke.py](../tests/shogi_suite/test_gui_smoke.py) | Opt-in real Tk regression tests; run with HASAMI_GUI_TESTS=1. |
| [tests/shogi_suite/test_gui_support.py](../tests/shogi_suite/test_gui_support.py) | Display-free shogi input, presentation and optional-model contract tests. |
| [tests/shogi_suite/test_history.py](../tests/shogi_suite/test_history.py) | Game lifecycle, complete history, repetition and terminal precedence. |
| [tests/shogi_suite/test_limits.py](../tests/shogi_suite/test_limits.py) | Implementation limits shared with the native engine's 32-bit ABI fields (shogi/model.py). |
| [tests/shogi_suite/test_model_notation.py](../tests/shogi_suite/test_model_notation.py) | Wire conventions, strict validation and random-play move-text round trips. |
| [tests/shogi_suite/test_native_bridge.py](../tests/shogi_suite/test_native_bridge.py) | Native failure atomicity, ABI validation, buffers, and owned-handle lifecycle. |
| [tests/shogi_suite/test_native_parity.py](../tests/shogi_suite/test_native_parity.py) | Bit-exact native/reference lockstep, including complete scaled random games. |
| [tests/shogi_suite/test_native_search.py](../tests/shogi_suite/test_native_search.py) | Independent native search bridge and shared agent contract regressions. |
| [tests/shogi_suite/test_perft.py](../tests/shogi_suite/test_perft.py) | Compare the engine with immutable, independently generated oracle facts. |
| [tests/shogi_suite/test_records.py](../tests/shogi_suite/test_records.py) | Lossless record replay, interoperability examples and malformed input tests. |
| [tests/shogi_suite/test_usi.py](../tests/shogi_suite/test_usi.py) | USI subprocess integration and deterministic protocol boundary tests. |
| [tests/shogi_suite/test_usi_smoke.py](../tests/shogi_suite/test_usi_smoke.py) | Exercise the public protocol driver without optional inference dependencies. |
| [tests/shogi_suite/vectors/README.md](../tests/shogi_suite/vectors/README.md) | 補助データまたは利用者向け文書。 |
| [tests/shogi_suite/vectors/perft_vectors.jsonl](../tests/shogi_suite/vectors/perft_vectors.jsonl) | 補助データまたは利用者向け文書。 |
| [tests/shogi_suite/vectors/rule_cases.json](../tests/shogi_suite/vectors/rule_cases.json) | 設定またはスキーマ（JSON）。 |
| [tests/test_gui_spec.py](../tests/test_gui_spec.py) | Display-free tests for GUI budgets and model registration. |
| [tests/test_integration.py](../tests/test_integration.py) | Python entry point or supporting module. |
| [tests/test_release_entrypoints.py](../tests/test_release_entrypoints.py) | Check runnable release entry points using only the standard library. |
| [tests/test_rules.py](../tests/test_rules.py) | Python entry point or supporting module. |
| [tests/werewolf_suite/__init__.py](../tests/werewolf_suite/__init__.py) | Dependency-free tests for the five-player werewolf engine. |
| [tests/werewolf_suite/test_baselines.py](../tests/werewolf_suite/test_baselines.py) | Baseline legality, truthful seer reports, deception, and information isolation. |
| [tests/werewolf_suite/test_game.py](../tests/werewolf_suite/test_game.py) | Rule and information-boundary regression tests. |

## werewolf

5 人の人狼の規則・観測・基準エージェント。GUI はありません。

| ファイル | 役割 |
| --- | --- |
| [werewolf/__init__.py](../werewolf/__init__.py) | Standard-library-only five-player werewolf village. |
| [werewolf/actions.py](../werewolf/actions.py) | The fixed 45-action protocol; all targets are clockwise relative seats. |
| [werewolf/baselines.py](../werewolf/baselines.py) | Observation-only baselines; claims are evidence, never privileged knowledge. |
| [werewolf/evaluation.py](../werewolf/evaluation.py) | Duplicate seeded tables with the evaluated agent rotated through all roles. |
| [werewolf/game.py](../werewolf/game.py) | Seeded simultaneous five-seat village with reversible full history. |
