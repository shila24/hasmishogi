#include "hasami/game.hpp"
#include "hasami/search.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

int main() {
    hasami::Game game;
    std::cout << "Hasami Lab C++ console. Enter: from to (0..80), ai, undo, or quit.\n";
    while (true) {
        const auto& p = game.position();
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                int s = r * 9 + c;
                std::cout << (p.black.test(s) ? 'P' : p.white.test(s) ? 'T' : '.') << ' ';
            }
            std::cout << '\n';
        }
        if (game.outcome().terminal())
            std::cout << "Game ended. Winner=" << game.outcome().winner << " (undo or quit)\n";
        else std::cout << "Turn=" << p.turn << " pending=" << p.pending << "\n";
        std::string line;
        if (!std::getline(std::cin, line) || line == "quit") break;
        try {
            if (line == "undo") { game.undo(); continue; }
            int action = -1;
            if (line == "ai") {
                hasami::AlphaBetaAgent agent;
                auto result = agent.choose(game, {3, 1000, 200000});
                action = result.action;
                std::cout << "Depth=" << result.depth << " nodes=" << result.nodes << '\n';
            } else {
                int from, to; std::string extra;
                std::istringstream input(line);
                if (!(input >> from >> to) || (input >> extra) || from < 0 || from > 80 || to < 0 || to > 80)
                    throw std::invalid_argument("Expected two square indices in [0, 80]");
                action = from * 81 + to;
            }
            game.play(action);
        } catch (const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; }
    }
}
