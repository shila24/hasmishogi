#include "riichi/game.hpp"
#include <array>
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        const int games = argc > 1 ? std::stoi(argv[1]) : 10;
        if (games < 1) throw std::invalid_argument("games must be positive");
        std::cout << '[';
        for (int encoding = 0; encoding < 2; ++encoding) {
            riichi::Random rng(24024);
            std::uint64_t steps = 0;
            std::array<float, RQ_PLANES_SIZE> planes{};
            std::array<float, RQ_ACTION_SIZE> mask{};
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < games; ++i) {
                riichi::Game game(i);
                while (!game.state().terminal) {
                    riichi::Joint joint{-1, -1, -1, -1};
                    for (int p : game.actors()) {
                        const auto legal = game.legal_actions(p);
                        if (encoding) { game.encode(p, planes); game.legal_mask(p, mask); }
                        joint[p] = legal[rng.randbelow(static_cast<unsigned>(legal.size()))];
                    }
                    game.play_joint(joint); ++steps;
                }
            }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (encoding) std::cout << ',';
            std::cout << "{\"backend\":\"cpp-tool\",\"encoding\":" << (encoding ? "true" : "false")
                      << ",\"games\":" << games << ",\"steps\":" << steps << ",\"seconds\":" << seconds
                      << ",\"steps_per_second\":" << steps / seconds << '}';
        }
        std::cout << "]\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
