// SPDX-License-Identifier: MIT
#include "shogi_search/search.hpp"
#include <iostream>
#include <random>
int main(int argc,char** argv) {
    try {
        shogi_search::Limits limits;
        if (argc > 1) limits.depth = std::stoi(argv[1]);
        if (argc > 2) limits.nodes = std::stoull(argv[2]);
        limits.validate();
        shogi::Game middle({320,1}); std::mt19937 rng(490049);
        for (int i = 0; i < 60 && !middle.state().outcome.terminal(); ++i) {
            const auto moves = middle.legal_actions(); middle.play(moves[rng()%moves.size()]);
        }
        const std::array<shogi::Game,3> games{shogi::Game{},middle,shogi::Game({320,1},"4k4/9/5g3/9/9/9/3G5/9/4K4 b - 1")};
        const std::array<const char*,3> names{"start","middlegame","endgame"};
        std::cout << "{\"format\":\"shogi-search-bench-v1\",\"threads\":1,\"rows\":[";
        for (int i = 0; i < 3; ++i) {
            shogi_search::Searcher search;
            const auto result = search.go(shogi_search::Root(games[i]),limits);
            if (i) std::cout << ',';
            std::cout << "{\"class\":\"" << names[i] << "\",\"sfen\":\"" << games[i].state().position.sfen()
                << "\",\"depth\":" << result.depth << ",\"nodes\":" << result.nodes << ",\"elapsed_ms\":" << result.elapsed_ms
                << ",\"nodes_per_second\":" << result.nodes*1000.0/result.elapsed_ms << ",\"score\":" << result.score
                << ",\"move\":" << result.move << ",\"completed\":" << (result.completed?"true":"false") << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
