#include "shogi/game.hpp"
#include <array>
#include <iostream>
#include <string_view>

namespace {
void require(bool condition, std::string_view message) {
    if (!condition) throw shogi::Error(std::string(message));
}
int square(std::string_view text) { return (text[1] - 'a') * 9 + ('9' - text[0]); }
int code(std::string_view text) {
    if (text[1] == '*') {
        const auto hand = std::string_view("PLNSGBR").find(text[0]);
        return shogi::drop_base + static_cast<int>(hand) * 81 + square(text.substr(2, 2));
    }
    return (square(text.substr(0, 2)) * 81 + square(text.substr(2, 2))) * 2 + (text.size() == 5 ? 1 : 0);
}
// Data-only facts transcribed from tests/shogi_suite/vectors/rule_cases.json (two independent oracles).
struct MoveCase { std::string_view name; std::string_view sfen; std::string_view move; bool legal; int count; };
const std::array<MoveCase, 19> moves{{
    {"uchifuzume", "8k/6G2/9/7N1/9/9/9/9/4K4 b P 1", "P*1b", false, 80},
    {"pawn_drop_check_not_mate", "8k/9/9/7N1/9/9/9/9/4K4 b P 1", "P*1b", true, 77},
    {"nifu", "4k4/9/9/9/9/9/4P4/9/4K4 b P 1", "P*5e", false, 70},
    {"nifu_promoted_ok", "4k4/9/9/9/9/9/4+P4/9/4K4 b P 1", "P*5e", true, 81},
    {"pawn_drop_last_rank", "4k4/9/9/9/9/9/9/9/4K4 b P 1", "P*1a", false, 76},
    {"lance_drop_last_rank", "4k4/9/9/9/9/9/9/9/4K4 b L 1", "L*1a", false, 76},
    {"knight_drop_rank2", "4k4/9/9/9/9/9/9/9/4K4 b N 1", "N*1b", false, 67},
    {"knight_drop_rank3", "4k4/9/9/9/9/9/9/9/4K4 b N 1", "N*1c", true, 67},
    {"pawn_move_last_rank_unpromoted", "4k4/P8/9/9/9/9/9/9/4K4 b - 1", "9b9a", false, 6},
    {"pawn_move_last_rank_promoted", "4k4/P8/9/9/9/9/9/9/4K4 b - 1", "9b9a+", true, 6},
    {"pawn_nonpromote_zone_ok", "4k4/9/P8/9/9/9/9/9/4K4 b - 1", "9c9b", true, 7},
    {"knight_rank2_must_promote", "4k4/9/9/N8/9/9/9/9/4K4 b - 1", "9d8b", false, 6},
    {"knight_rank2_promote", "4k4/9/9/N8/9/9/9/9/4K4 b - 1", "9d8b+", true, 6},
    {"pinned_silver", "4k4/9/9/9/4r4/9/4S4/9/4K4 b - 1", "5g6f", false, 6},
    {"pinned_silver_along_line", "4k4/9/9/9/4r4/9/4S4/9/4K4 b - 1", "5g5f", true, 6},
    {"king_into_check", "4k4/9/9/9/9/9/9/3r5/4K4 b - 1", "5i5h", false, 2},
    {"gote_uchifuzume", "4k4/9/9/9/9/1n7/9/2g6/K8 w p 1", "P*9h", false, 80},
    {"gote_nifu", "4k4/4p4/9/9/9/9/9/9/4K4 w p 1", "P*5e", false, 69},
    {"gote_pawn_drop_last_rank", "4k4/9/9/9/9/9/9/9/4K4 w p 1", "P*1i", false, 76},
}};
struct DeclarationCase { std::string_view name; std::string_view sfen; bool legal; };
const std::array<DeclarationCase, 8> declarations{{
    {"sente_28_10pieces", "GGGG1SSSS/RB7/4K4/9/9/9/9/9/4k4 b RB 1", true},
    {"sente_28_9pieces", "GGGG2SSS/RB7/4K4/9/9/9/9/9/4k4 b RBP 1", false},
    {"sente_27_10pieces", "GGGG1SSSS/RB7/4K4/9/9/9/9/9/4k4 b B4P 1", false},
    {"sente_king_outside_camp", "GGGG1SSSS/RB7/9/4K4/9/9/9/9/4k4 b RB 1", false},
    {"sente_28_in_check", "GGGG1SSSS/RB7/4K4/9/4r4/9/9/9/4k4 b B5P 1", false},
    {"sente_28_rook_not_checking", "GGGG1SSSS/RB7/4K4/9/r8/9/9/9/4k4 b B5P 1", true},
    {"gote_27_10pieces", "4K4/9/9/9/9/9/4k4/7br/ssss1gggg w b4p 1", true},
    {"gote_26_10pieces", "4K4/9/9/9/9/9/4k4/7br/ssss1gggg w b3p 1", false},
}};
void rule_cases() {
    for (const auto& test : moves) {
        shogi::Game game({}, test.sfen);
        const auto initial = game.state().position;
        const auto actions = game.legal_actions();
        require(actions.size() == static_cast<std::size_t>(test.count), test.name);
        require(game.is_legal(code(test.move)) == test.legal, test.name);
        if (test.legal) {
            game.play(code(test.move));
            require(game.state().position.turn == 1 - initial.turn, "turn alternates");
            require(!shogi::in_check(game.state().position, initial.turn), "king stays safe");
            require(game.undo(), "undo rule move");
        } else {
            bool rejected = false;
            try { game.play(code(test.move)); }
            catch (const shogi::Error& error) { rejected = std::string_view(error.what()) == "Illegal move"; }
            require(rejected, test.name);
        }
        require(game.state().position == initial && game.legal_actions() == actions, "rule state preserved");
        require(game.moves().empty() && game.history_keys() == std::vector<std::string>{initial.key()}, "rule history preserved");
    }
}
void declaration_cases() {
    for (const auto& test : declarations) {
        shogi::Game game({1, 1}, test.sfen);
        require(game.can_declare() == test.legal, test.name);
        require(game.is_legal(shogi::declare) == test.legal, test.name);
        shogi::Game disabled({1, 0}, test.sfen);
        require(!disabled.can_declare(), "disabled declaration");
        if (!test.legal) continue;
        const auto initial = game.state().position;
        const auto result = game.play(shogi::declare);
        require(result.code == shogi::declare && result.captured == 0 && !result.check, "declaration result");
        require(game.state().outcome == shogi::Outcome{initial.turn, 3}, "declaration precedes move cap");
        require(game.state().position.board == initial.board && game.state().position.hands == initial.hands,
                "declaration preserves pieces");
        require(game.state().position.turn == initial.turn, "declaration preserves turn");
        require(game.state().position.move_number == initial.move_number + 1, "declaration advances move number");
        require(game.moves() == std::vector<int>{shogi::declare}, "declaration move recorded");
        require(game.history_keys() == std::vector<std::string>{initial.key(), initial.key()}, "declaration key recorded");
        require(game.legal_actions().empty() && !game.can_declare(), "declaration terminal");
        require(game.undo() && game.state().position == initial && game.can_declare(), "undo declaration");
    }
}
void capture_promoted_piece() {
    shogi::Game game({}, "4k4/9/9/4+p4/4R4/9/9/9/4K4 b - 1");
    const auto initial = game.state().position;
    const auto result = game.play(code("5e5d"));
    require(result.captured == 25 && result.check, "captured original cell and checking move");
    require(game.state().position.hands[0][0] == 1, "capture demotes pawn into hand");
    require(game.state().position.board[square("5d")] == 7, "rook occupies capture square");
    require(game.undo() && game.state().position == initial, "capture undo restores hands and board");
}
} // namespace
int main() {
    try { rule_cases(); declaration_cases(); capture_promoted_piece(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
