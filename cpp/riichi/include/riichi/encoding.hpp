#pragma once
#include "riichi/game.hpp"
namespace riichi {
inline constexpr int encoding_size = RQ_PLANES_SIZE;
inline constexpr int action_size = RQ_ACTION_SIZE;
inline void encode(const Game& game, int player, std::span<float> output) {
    game.encode(player, output);
}
}
