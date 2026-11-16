// SPDX-License-Identifier: MIT
#include "shogi_search/search.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

namespace {
void require(bool ok, const char* text) { if (!ok) throw shogi::Error(text); }
void parity(const shogi::Position& ref, int depth, bool declaration) {
    shogi_search::Position fast(ref);
    auto legal = shogi::legal_moves(ref);
    if (declaration && shogi::declarable(ref)) legal.push_back(shogi::declare);
    const auto actual = fast.legal(declaration);
    require(std::vector<int>(actual.begin(),actual.end()) == legal,"Legal move mismatch");
    const auto key = fast.key();
    shogi_search::BootstrapEvaluator incremental;
    incremental.reset(fast);
    const int eval = incremental.evaluate(fast);
    for (int code : legal) {
        const auto undo = fast.make(code);
        incremental.push(fast,undo);
        require(fast.key() == fast.recompute_key(),"Incremental key mismatch");
        shogi_search::BootstrapEvaluator fresh; fresh.reset(fast);
        require(incremental.evaluate(fast) == fresh.evaluate(fast),"Incremental evaluation mismatch");
        if (code != shogi::declare) {
            auto next = ref; shogi::apply_move(next,code); next.turn ^= 1; ++next.move_number;
            require(next == fast.board(),"Make mismatch");
        }
        incremental.pop(); fast.unmake(undo);
        require(fast.board() == ref && fast.key() == key,"Unmake mismatch");
        require(incremental.evaluate(fast) == eval,"Evaluation undo mismatch");
        const auto restored = fast.legal(declaration);
        require(std::vector<int>(restored.begin(),restored.end()) == legal,"Occupancy undo mismatch");
    }
    const auto expected = shogi::perft(ref,depth,declaration);
    require(shogi_search::perft(fast,depth,declaration) == expected,"Perft mismatch");
}
}
int main() {
    try {
        const auto start = shogi::parse_sfen(shogi::start_sfen);
        parity(start,5,true);
        std::mt19937 rng(490015);
        shogi::Game game({180,1});
        auto root = shogi_search::Root(game);
        for (int i = 0; i < 200; ++i) {
            if (game.state().outcome.terminal()) { game = shogi::Game({180,1}); root = shogi_search::Root(game); }
            parity(game.state().position,i % 25 == 0 ? 3 : 2,i % 2 != 0);
            const auto legal = game.legal_actions();
            const int code = legal[rng()%legal.size()];
            game.play(code); root.position.make(code);
            root.history.push_back({root.position.key(),root.position.board().turn,root.position.check()});
            const auto outcome = code == shogi::declare ? shogi::Outcome{root.position.board().turn,3}
                : shogi_search::adjudicate(root.position,root.position.legal(),root.rules,root.history);
            require(outcome == game.state().outcome,"Random trajectory outcome mismatch");
        }
        std::ifstream input(std::string(SS_SOURCE_ROOT)+"/cpp/shogi_search/tests/data/parity-v1.tsv");
        require(static_cast<bool>(input),"Missing parity vectors");
        std::string line; int vectors = 0;
        while (std::getline(input,line)) if (!line.empty() && line[0] != '#') {
            parity(shogi::parse_sfen(line),2,false);
            parity(shogi::parse_sfen(line),2,true);
            ++vectors;
        }
        std::cout << "start perft(5), 200 seeded positions, " << vectors << " rule vectors: passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
