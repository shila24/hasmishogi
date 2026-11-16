// SPDX-License-Identifier: MIT
#include "../tests/mate_vectors.hpp"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if (argc != 2) throw shogi::Error("Usage: shogi_search_mates SUITE.tsv");
        const auto rows = mate_vectors::read(argv[1]);
        std::cout << "{\"format\":\"shogi-search-mate-results-v1\",\"rows\":[";
        bool comma = false;
        for (int depth : {1,3,5}) for (const auto& row : rows) {
            shogi_search::Searcher search({0,1,0}); shogi_search::Limits limits; limits.depth = depth;
            shogi::Game game({0,0},row.sfen);
            auto result = search.go(shogi_search::Root(game),limits);
            const bool solved = result.score == shogi_search::mate_score-row.plies;
            if (comma) std::cout << ',';
            comma = true;
            std::cout << "{\"id\":\"" << row.id << "\",\"mate_plies\":" << row.plies << ",\"depth\":" << depth
                << ",\"solved\":" << (solved?"true":"false") << ",\"nodes\":" << result.nodes << ",\"elapsed_ms\":" << result.elapsed_ms << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
