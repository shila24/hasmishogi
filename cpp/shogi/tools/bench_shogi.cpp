#include "shogi/game.hpp"
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>

namespace {
std::uint32_t number(std::string_view text) {
    std::uint32_t value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw shogi::Error("Expected an unsigned 32-bit integer");
    return value;
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw shogi::Error("Usage: shogi_bench_shogi STEPS SEED ENCODING MAX_MOVES");
        const auto steps = number(argv[1]), seed = number(argv[2]), encoding = number(argv[3]), maximum = number(argv[4]);
        if (steps == 0 || encoding > 1 || maximum == 0 || maximum > shogi::max_game_moves)
            throw shogi::Error("Invalid benchmark arguments");
        const shogi::Rules rules{static_cast<int>(maximum), 1};
        std::uint32_t rng = seed, checksum = 0, games = 1;
        double feature_checksum = 0;
        const auto start = std::chrono::steady_clock::now();
        shogi::Game game(rules);
        for (std::uint32_t step = 0; step < steps; ++step) {
            if (game.state().outcome.terminal()) { game = shogi::Game(rules); ++games; }
            const auto actions = game.legal_actions();
            if (encoding) {
                const int side = game.state().position.turn;
                const auto planes = game.encode(side);
                const auto mask = game.legal_mask(side);
                // Consume generated values, preventing dead-code elimination of encoding.
                feature_checksum += static_cast<double>(planes[step % shogi::planes_size]) + mask[step % shogi::action_size];
            }
            // Same explicitly wrapped LCG and action ordering in all three benchmark paths.
            rng = rng * 1664525U + 1013904223U;
            const int action = actions[rng % actions.size()];
            game.play(action);
            checksum = checksum * 31U + static_cast<std::uint32_t>(action);
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout.precision(17);
        std::cout << "{\"backend\":\"cpp_tool\",\"steps\":" << steps << ",\"seed\":" << seed
                  << ",\"encoding\":" << (encoding ? "true" : "false") << ",\"games_started\":" << games
                  << ",\"checksum\":" << checksum << ",\"feature_checksum\":" << feature_checksum
                  << ",\"seconds\":" << seconds << ",\"steps_per_second\":" << steps / seconds << "}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
