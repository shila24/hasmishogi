#include "shogi/game.hpp"
#include <array>
#include <iostream>

int main() {
    const shogi::Game game;
    constexpr std::array<std::int64_t, 6> expected{1, 30, 900, 25470, 719731, 19861490};
    for (int depth = 0; depth <= 5; ++depth) {
        const auto actual = game.perft(depth);
        std::cout << "perft(" << depth << ") = " << actual << '\n';
        if (actual != expected[depth]) return 1;
    }
    if (game.state().ply != 0 || game.state().position.sfen() != shogi::start_sfen) return 1;
}
