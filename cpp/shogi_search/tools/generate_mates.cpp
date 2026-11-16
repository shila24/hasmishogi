// SPDX-License-Identifier: MIT
#include "shogi_search/search.hpp"
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>

namespace {
std::uint64_t proof_nodes{};
// Existential attacker, universal defender: every legal defence is checked.
// Failed attacking alternatives may short-circuit; no evaluation or search pruning.
bool forced(shogi::Position p, int depth, int attacker) {
    ++proof_nodes;
    const auto moves = shogi::legal_moves(p);
    if (moves.empty()) return p.turn != attacker && shogi::in_check(p,p.turn);
    if (!depth) return false;
    for (int move : moves) {
        auto child = p; shogi::apply_move(child,move); child.turn ^= 1;
        const bool wins = forced(child,depth-1,attacker);
        if (p.turn == attacker && wins) return true;
        if (p.turn != attacker && !wins) return false;
    }
    return p.turn != attacker;
}
}
int main(int argc,char** argv) {
    try {
        if (argc != 2) throw shogi::Error("Usage: shogi_search_generate_mates OUTPUT.tsv");
        std::ofstream out(argv[1]);
        if (!out) throw shogi::Error("Cannot open mate output");
        out << "# shogi-search-mates-v1\n# seed=490049; own sparse-position generator; exhaustive reference proof at depth d and d-2.\n# id\tplies\tsfen\tproof_nodes\n";
        std::mt19937 rng(490049);
        std::array<bool,6> found{};
        for (int attempt = 0; attempt < 20000; ++attempt) {
            shogi::Position p; p.board[0] = 24; p.board[80] = 8;
            const std::array<int,5> types{5,7,3,4,5};
            for (int i = 0; i < 3 + static_cast<int>(rng()%3); ++i) {
                const int r = 1+static_cast<int>(rng()%4), c = static_cast<int>(rng()%4);
                if (!p.board[r*9+c]) p.board[r*9+c] = types[i];
            }
            try { p = shogi::parse_sfen(p.sfen()); } catch (const shogi::Error&) { continue; }
            shogi::Game game({0,0},p.sfen());
            if (game.state().outcome.terminal()) continue;
            shogi_search::Searcher search({shogi_search::pvs | shogi_search::ordering,1,0});
            shogi_search::Limits limits; limits.depth = 5; limits.nodes = 200000;
            const auto result = search.go(shogi_search::Root(game),limits);
            const int distance = shogi_search::mate_score-result.score;
            if (!result.completed || distance < 1 || distance > 5 || distance%2 == 0 || found[distance]) continue;
            proof_nodes = 0;
            if (!forced(p,distance,0) || (distance > 1 && forced(p,distance-2,0))) { std::cerr << "non-checkmate win candidate: " << p.sfen() << " score=" << result.score << '\n'; continue; }
            bool checkmate_choice = true;
            for (unsigned features : {0U,unsigned(shogi_search::all_features)}) for (bool rotate : {false,true}) {
                auto candidate = p;
                if (rotate) {
                    for (int sq = 0; sq < 81; ++sq) { const int c = p.board[80-sq]; candidate.board[sq] = c ? (c & 15)+16*(1-(c>>4)) : 0; }
                    candidate.turn = 1;
                }
                shogi_search::Searcher verify_search({features,1,12});
                shogi_search::Limits verify_limits; verify_limits.depth = distance;
                const auto chosen = verify_search.go(shogi_search::Root(shogi::Game({0,0},candidate.sfen())),verify_limits);
                const int attacker = candidate.turn;
                shogi::apply_move(candidate,chosen.move); candidate.turn ^= 1;
                if (!forced(candidate,distance-1,attacker)) checkmate_choice = false;
            }
            if (!checkmate_choice) continue;
            out << "generated-" << attempt << '\t' << distance << '\t' << p.sfen() << '\t' << proof_nodes << '\n';
            out.flush(); found[distance] = true;
            std::cerr << "verified mate in " << distance << " plies, " << proof_nodes << " reference nodes\n";
            if (found[1] && found[3] && found[5]) return 0;
        }
        throw shogi::Error("Generator did not find every required mate depth");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
