#include "shogi/game.hpp"
#include <algorithm>

namespace shogi {
Outcome repetition_outcome(const std::array<bool, 2>& seen, const std::array<bool, 2>& all_check) noexcept {
    const bool black = seen[0] && all_check[0], white = seen[1] && all_check[1];
    return black != white ? Outcome{black ? 1 : 0, 4} : Outcome{-1, 5};
}
Game::Game(Rules rules, std::string_view sfen) : rules_(rules) {
    rules_.validate();
    auto node = std::make_shared<Node>();
    node->position = parse_sfen(sfen);
    node->key = node->position.key();
    finish_node(*node);
    history_.push_back(std::move(node));
}
void Game::finish_node(Node& node) const {
    node.check = in_check(node.position, node.position.turn);
    node.legal = legal_moves(node.position);
    if (rules_.declaration && declarable(node.position)) node.legal.push_back(declare);
    if (node.legal.empty()) node.outcome = {1 - node.position.turn, node.check ? 1 : 2};
}
std::vector<int> Game::legal_actions() const {
    return state().outcome.terminal() ? std::vector<int>{} : state().legal;
}
bool Game::is_legal(int code) const noexcept {
    return !state().outcome.terminal() && std::binary_search(state().legal.begin(), state().legal.end(), code);
}
bool Game::can_declare() const noexcept { return is_legal(declare); }
MoveResult Game::play(int code) {
    const auto& previous = state();
    if (previous.outcome.terminal()) throw Error("Game is over");
    if (!is_legal(code)) throw Error("Illegal move");
    if (previous.ply >= max_game_moves) throw Error("Move limit exceeded");
    auto next = std::make_shared<Node>();
    next->position = previous.position;
    ++next->position.move_number;
    next->ply = previous.ply + 1;
    next->code = code;
    MoveResult result{code, 0, false};
    if (code == declare) {
        next->outcome = {previous.position.turn, 3};
        next->check = previous.check;
    } else {
        result.captured = apply_move(next->position, code);
        next->position.turn = 1 - previous.position.turn;
        finish_node(*next);
        result.check = next->check;
        next->gave_check = result.check;
    }
    next->key = next->position.key();
    if (!next->outcome.terminal()) {
        std::size_t first = history_.size();
        int count = 1;
        for (std::size_t i = 0; i < history_.size(); ++i) {
            if (history_[i]->key == next->key) { ++count; first = std::min(first, i); }
        }
        if (count >= 4) {
            std::array<bool, 2> seen{}, checks{true, true};
            for (std::size_t i = first + 1; i <= history_.size(); ++i) {
                const auto& node = i == history_.size() ? *next : *history_[i];
                const int actor = history_[i - 1]->position.turn;
                seen[actor] = true;
                checks[actor] = checks[actor] && node.gave_check;
            }
            next->outcome = repetition_outcome(seen, checks);
        }
        if (!next->outcome.terminal() && rules_.max_moves && next->ply == rules_.max_moves)
            next->outcome = {-1, 6};
    }
    // The only mutation is publication; vector growth also has the strong guarantee.
    history_.push_back(std::move(next));
    return result;
}
bool Game::undo() noexcept {
    if (history_.size() == 1) return false;
    history_.pop_back();
    return true;
}
void Game::resign() {
    if (state().outcome.terminal()) throw Error("Game is over");
    auto next = std::make_shared<Node>(state());
    next->outcome = {1 - next->position.turn, 7};
    next->resignation = true;
    history_.push_back(std::move(next));
}
std::vector<int> Game::moves() const {
    std::vector<int> result;
    result.reserve(state().ply);
    for (std::size_t i = 1; i < history_.size(); ++i)
        if (!history_[i]->resignation) result.push_back(history_[i]->code);
    return result;
}
std::vector<std::string> Game::history_keys() const {
    std::vector<std::string> result;
    result.reserve(state().ply + 1);
    for (const auto& node : history_) if (!node->resignation) result.push_back(node->key);
    return result;
}
std::string Game::history_key(int index) const {
    if (index < 0 || index > state().ply) throw Error("Invalid history index");
    return history_[static_cast<std::size_t>(index)]->key;
}
void Game::validate_player(int player) const {
    if (player < 0 || player > 1) throw Error("Invalid player");
    if (player != state().position.turn) throw Error("The encoder player must be the side to move");
}
std::array<float, planes_size> Game::encode(int player) const {
    validate_player(player);
    std::array<float, planes_size> result{};
    const auto& p = state().position;
    for (int sq = 0; sq < 81; ++sq) {
        const int cell = p.board[sq];
        if (!cell) continue;
        const int plane = (cell & 15) - 1 + (cell >> 4 != player ? 14 : 0);
        result[plane * 81 + (player ? 80 - sq : sq)] = 1.0F;
    }
    for (int relative = 0; relative < 2; ++relative) {
        const int owner = relative ? 1 - player : player, base = relative ? 35 : 28;
        for (int hand = 0; hand < 7; ++hand) {
            const auto value = static_cast<float>(static_cast<double>(p.hands[owner][hand]) / piece_limits[hand]);
            std::fill_n(result.begin() + (base + hand) * 81, 81, value);
        }
    }
    int earlier = -1;
    for (const auto& node : history_) if (!node->resignation && node->key == state().key) ++earlier;
    const std::array<double, 7> values{
        static_cast<double>(player == 1), static_cast<double>(state().check),
        static_cast<double>(earlier >= 1), static_cast<double>(earlier >= 2), static_cast<double>(earlier >= 3),
        rules_.max_moves ? std::min(1.0, static_cast<double>(state().ply) / rules_.max_moves) : 0.0, 1.0};
    for (int plane = 42; plane < 49; ++plane)
        std::fill_n(result.begin() + plane * 81, 81, static_cast<float>(values[plane - 42]));
    return result;
}
std::array<float, action_size> Game::legal_mask(int player) const {
    validate_player(player);
    std::array<float, action_size> result{};
    if (!state().outcome.terminal())
        for (const int code : state().legal) result[move_label(player, code)] = 1.0F;
    return result;
}
std::int64_t Game::perft(int depth) const {
    if (depth < 0) throw Error("Invalid depth");
    if (depth && state().outcome.terminal()) return 0;
    return shogi::perft(state().position, depth, rules_.declaration != 0);
}
} // namespace shogi
