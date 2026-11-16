// SPDX-License-Identifier: MIT
#include "mate_vectors.hpp"
#include <iostream>
int main() {
    try {
        const auto rows = mate_vectors::read(std::string(SS_SOURCE_ROOT)+"/cpp/shogi_search/tests/data/mates-v1.tsv");
        for (const auto& row : rows) for (bool rotate : {false,true}) {
            auto p = shogi::parse_sfen(row.sfen);
            if (rotate) {
                auto copy = p;
                for (int sq = 0; sq < 81; ++sq) { const int c = copy.board[80-sq]; p.board[sq] = c ? (c & 15) + 16*(1-(c>>4)) : 0; }
                std::swap(p.hands[0],p.hands[1]); p.turn ^= 1;
            }
            if (!mate_vectors::forced(p,row.plies,p.turn) || (row.plies > 1 && mate_vectors::forced(p,row.plies-2,p.turn)))
                throw shogi::Error("Reference exhaustive mate proof mismatch: " + row.id);
            for (unsigned features : {0U,unsigned(shogi_search::all_features)}) {
                shogi_search::Searcher search({features,1,12}); shogi_search::Limits limits; limits.depth = row.plies;
                const auto result = search.go(shogi_search::Root(shogi::Game({0,0},p.sfen())),limits);
                if (!result.completed || result.score != shogi_search::mate_score-row.plies || result.mate_plies != row.plies)
                    throw shogi::Error("Mate search mismatch: " + row.id + " features=" + std::to_string(features) + " score=" + std::to_string(result.score));
                auto child = p; shogi::apply_move(child,result.move); child.turn ^= 1;
                if (!mate_vectors::forced(child,row.plies-1,p.turn)) throw shogi::Error("Mate best move has no reference proof: " + row.id + " features=" + std::to_string(features) + " move=" + std::to_string(result.move));
            }
        }
        std::cout << rows.size()*2 << " mate positions proven and solved\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
