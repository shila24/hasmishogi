#include "shogi/game.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>
#include <vector>

namespace {
void require(bool condition, std::string_view message) {
    if (!condition) throw shogi::Error(std::string(message));
}
int square(std::string_view text) { return (text[1] - 'a') * 9 + ('9' - text[0]); }
int move(std::string_view text) { return (square(text.substr(0, 2)) * 81 + square(text.substr(2, 2))) * 2; }
void same(const shogi::Game& a, const shogi::Game& b) {
    require(a.state().position == b.state().position, "position restored");
    require(a.state().outcome == b.state().outcome, "outcome restored");
    require(a.state().ply == b.state().ply, "ply restored");
    require(a.state().check == b.state().check, "check restored");
    require(a.moves() == b.moves(), "moves restored");
    require(a.history_keys() == b.history_keys(), "history restored");
    require(a.legal_actions() == b.legal_actions(), "legal actions restored");
    const int side = a.state().position.turn;
    require(a.encode(side) == b.encode(side), "encoding restored");
    require(a.legal_mask(side) == b.legal_mask(side), "mask restored");
    for (int i = 0; i <= a.state().ply; ++i)
        require(a.history_key(i) == b.history_key(i), "indexed history restored");
}
template<class F> void rejected(F&& action, std::string_view message) {
    try { action(); } catch (const shogi::Error& error) {
        require(error.what() == message, "error text mismatch");
        return;
    }
    throw shogi::Error("Expected rejection");
}
void history_and_clone() {
    shogi::Game game;
    const shogi::Game initial = game;
    require(!game.undo(), "initial undo is false");
    std::vector<shogi::Game> snapshots;
    for (int i = 0; i < 80 && !game.state().outcome.terminal(); ++i) {
        snapshots.push_back(game);
        const auto codes = game.legal_actions();
        game.play(codes[static_cast<std::size_t>(i * 17) % codes.size()]);
        require(game.state().ply == i + 1, "ply advances");
        require(game.history_keys().size() == game.moves().size() + 1, "one key per move plus initial");
        auto copy = game;
        copy.resign();
        require(copy.state().outcome == shogi::Outcome{1 - game.state().position.turn, 7}, "resign winner");
        require(copy.moves() == game.moves(), "resign adds no move");
        require(copy.history_keys() == game.history_keys(), "resign adds no key");
        const auto terminal = copy;
        rejected([&] { copy.resign(); }, "Game is over");
        rejected([&] { copy.play(0); }, "Game is over");
        same(copy, terminal);
        require(copy.undo(), "undo resignation");
        same(copy, game);
        require(copy.undo(), "clone undo move");
        same(copy, snapshots.back());
        const auto before = game;
        rejected([&] { game.play(0); }, "Illegal move");
        same(game, before);
    }
    while (!snapshots.empty()) {
        require(game.undo(), "undo move");
        same(game, snapshots.back());
        snapshots.pop_back();
    }
    same(game, initial);
    require(!game.undo(), "undo exhausted");
    // A clone keeps shared immutable history alive after its source is destroyed.
    const auto survivor = [] { shogi::Game local; local.play(local.legal_actions()[0]); return local; }();
    require(survivor.state().ply == 1 && survivor.history_keys().size() == 2, "clone ownership");
}
void repetition() {
    struct Case { std::string_view sfen; std::vector<std::string_view> prefix; std::array<std::string_view, 4> cycle;
                  int repeats; shogi::Outcome expected; };
    const std::array<Case, 4> cases{{
        {"4k4/9/9/9/9/9/9/9/4K4 b - 1", {}, {"5i6i", "5a6a", "6i5i", "6a5a"}, 3, {-1, 5}},
        {"4k4/9/5R3/9/9/9/9/9/4K4 b - 1", {}, {"4c5c", "5a4a", "5c4c", "4a5a"}, 3, {1, 4}},
        {"4k4/9/5R3/9/9/9/9/9/4K4 b - 1", {"4c3c", "5a4a", "3c4c", "4a5a"},
         {"4c5c", "5a4a", "5c4c", "4a5a"}, 2, {-1, 5}},
        {"4k4/9/9/6R2/9/9/9/9/4K4 b - 1", {"3d4d", "5a6a", "4d4c", "6a5a"},
         {"4c5c", "5a4a", "5c4c", "4a5a"}, 3, {1, 4}}
    }};
    for (const auto& test : cases) for (const bool rotate : {false, true}) {
        auto position = shogi::parse_sfen(test.sfen);
        auto expected = test.expected;
        std::vector<int> codes;
        for (auto text : test.prefix) codes.push_back(move(text));
        for (int i = 0; i < test.repeats; ++i) for (auto text : test.cycle) codes.push_back(move(text));
        if (rotate) {
            std::reverse(position.board.begin(), position.board.end());
            for (auto& cell : position.board) if (cell) cell = (cell & 15) + 16 * (1 - (cell >> 4));
            std::swap(position.hands[0], position.hands[1]);
            position.turn = 1 - position.turn;
            for (auto& code : codes) code = ((80 - code / 2 / 81) * 81 + 80 - code / 2 % 81) * 2;
            if (expected.winner >= 0) expected.winner = 1 - expected.winner;
        }
        shogi::Game game({static_cast<int>(codes.size()), 1}, position.sfen());
        for (std::size_t i = 0; i < codes.size(); ++i) {
            require(!game.state().outcome.terminal(), "no premature repetition");
            const auto before = game;
            game.play(codes[i]);
            if (i + 1 == codes.size()) {
                require(game.state().outcome == expected, "repetition precedes move cap");
                require(game.legal_actions().empty(), "terminal legal set empty");
                const auto terminal = game;
                auto copy = game;
                require(copy.undo(), "clone undo repetition");
                same(copy, before);
                copy.play(codes[i]);
                same(copy, terminal);
                require(game.undo(), "undo repetition");
                same(game, before);
                game.play(codes[i]);
                same(game, terminal);
            }
        }
    }
    require(shogi::repetition_outcome({true, true}, {true, true}) == shogi::Outcome{-1, 5}, "mutual checks draw");
    require(shogi::repetition_outcome({false, false}, {true, true}) == shogi::Outcome{-1, 5}, "empty span draw");
}
void terminal_precedence() {
    const std::array<std::pair<std::string_view, int>, 2> cases{{
        {"k8/9/1GN6/1R7/9/9/9/9/4K4 b - 1", move("8d9d")},
        {"k8/9/1G7/9/1N7/9/9/9/4K4 b - 1", move("8e7c")}
    }};
    int reason = 1;
    for (const auto& [sfen, code] : cases) {
        shogi::Game game({1, 1}, sfen);
        const auto initial = game;
        game.play(code);
        require(game.state().outcome == shogi::Outcome{0, reason++}, "no reply precedes move cap");
        require(game.undo(), "undo mate/no moves");
        same(game, initial);
    }
    shogi::Game limited({1, 1});
    limited.play(limited.legal_actions()[0]);
    require(limited.state().outcome == shogi::Outcome{-1, 6}, "move cap draws");
}
} // namespace
int main() {
    try { history_and_clone(); repetition(); terminal_precedence(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
