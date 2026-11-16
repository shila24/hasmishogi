#include "hasami/game.hpp"
#include "hasami/search.hpp"
#include "hasami/c_api.h"
#include <iostream>
#include <random>
#include <stdexcept>

using namespace hasami;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
Bits mask(std::initializer_list<int> squares) {
    Bits b;
    for (int s : squares) b.set(s);
    return b;
}
int main() {
    try {
        for (int square = 0; square < 81; ++square) {
            Bits b; b.set(square);
            check(b.count() == 1 && b.first() == square, "Bit indexing failed");
            for (unsigned shift = 0; shift <= 129; ++shift) {
                Bits expected_left, expected_right;
                if (square + shift < 81) expected_left.set(static_cast<int>(square + shift));
                if (shift <= static_cast<unsigned>(square)) expected_right.set(square - static_cast<int>(shift));
                check(b.left(shift) == expected_left, "Left shift failed");
                check(b.right(shift) == expected_right, "Right shift failed");
            }
        }
        check(!neighbours(mask({8})).test(9), "Horizontal row wrap");
        check(neighbours(mask({63})).test(64), "Word boundary neighbour missing");
        Game game;
        check(game.legal_actions().size() == 63, "Opening move count");
        auto start = game.position();
        auto nodes = perft(game, 2);
        check(nodes == 3717, "Opening depth-two tree mismatch");
        check(game.position() == start && game.moves().empty(), "Perft mutated root");
        Rules mechanics{true, 0, 0, 0, false};
        Game corner(mechanics, {mask({1,18}), mask({0}), 0, -1});
        check(corner.play(18 * 81 + 9).captured == mask({0}), "Corner capture");
        check(corner.undo() && corner.position().white == mask({0}), "Capture undo");
        Game insertion(mechanics, {mask({67}), mask({39,41}), 0, -1});
        check(!insertion.play(67 * 81 + 40).captured, "Self-insertion was captured");
        Game multi(mechanics, {mask({13,58,42,37}), mask({22,31,49,41}), 0, -1});
        check(multi.play(37 * 81 + 40).captured.count() == 4, "Multi-direction capture");
        Game lead({}, {mask({38,67,72,73,74,75,76,77,78}), mask({39,37,3,0,1,2,4}), 0, -1});
        lead.play(67 * 81 + 40);
        check(lead.position().pending == 0 && !lead.outcome().terminal(), "Missing reply state");
        lead.play(3 * 81 + 39);
        check(!lead.outcome().terminal() && lead.position().pending == -1, "Reply rescue failed");
        const int cycle[] = {72*81+63, 0*81+9, 63*81+72, 9*81+0};
        for (int i = 0; i < 12; ++i) game.play(cycle[i % 4]);
        check(game.outcome().winner == -1 && game.outcome().reason == Reason::Repetition, "Repetition failed");
        check(game.undo() && !game.outcome().terminal(), "Repetition undo failed");
        game.play(cycle[3]);
        check(game.outcome().winner == -1, "Repetition replay failed");
        Game search_game;
        AlphaBetaAgent agent;
        auto result = agent.choose(search_game, {2, 10000, 2000000});
        check(result.depth == 2 && result.action >= 0 && search_game.moves().empty(), "Native search failed");
        std::mt19937 rng(19);
        for (int match = 0; match < 20; ++match) {
            Game random_game;
            for (int ply = 0; ply < 180 && !random_game.outcome().terminal(); ++ply) {
                auto before = random_game.position();
                auto history = random_game.history();
                auto actions = random_game.legal_actions();
                int action = actions[rng() % actions.size()];
                random_game.play(action);
                check(!(random_game.position().black & random_game.position().white), "Overlap after play");
                auto after = random_game.position();
                check(random_game.undo() && random_game.position() == before && random_game.history() == history,
                      "Random undo mismatch");
                random_game.play(action);
                check(random_game.position() == after, "Random replay mismatch");
            }
        }
        void* handle = hs_create(nullptr, nullptr);
        check(handle != nullptr, "C ABI create failed");
        int32_t canary = -123;
        check(hs_legal_actions(handle, &canary, 1) == -1 && canary == -123, "C ABI buffer guard");
        check(hs_play(handle, -1) == -1, "C ABI action guard");
        hs_state state{};
        check(hs_get_state(nullptr, &state) == -1, "C ABI null guard");
        check(hs_get_state(handle, &state) == 0 && state.ply == 0, "C ABI state failed");
        hs_destroy(handle);
        std::cout << "Native tests passed. Opening perft(2)=" << nodes << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failure: " << e.what() << '\n';
        return 1;
    }
}
