#ifndef SHOGI_GAME_HPP
#define SHOGI_GAME_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace shogi {
inline constexpr int squares = 81;
inline constexpr int drop_base = 13122;
inline constexpr int declare = 13689;
inline constexpr int action_size = 2188;
inline constexpr int planes_size = 49 * squares;
inline constexpr int max_game_moves = 1000000;
inline constexpr std::array<int, 7> piece_limits{18, 4, 4, 4, 4, 2, 2};
inline constexpr std::string_view start_sfen =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
struct Rules {
    int max_moves{320};
    int declaration{1};
    void validate() const;
};
struct Position {
    std::array<int, squares> board{};
    std::array<std::array<int, 7>, 2> hands{};
    int turn{};
    int move_number{1};
    bool operator==(const Position&) const = default;
    std::string key() const;
    std::string sfen() const;
};
struct Outcome {
    int winner{-2};
    int reason{};
    bool terminal() const noexcept { return winner != -2; }
    bool operator==(const Outcome&) const = default;
};
struct MoveResult { int code; int captured; bool check; };

int promoted(int piece) noexcept;
int unpromoted(int piece) noexcept;
bool in_zone(int square, int side) noexcept;
bool dead_square(int piece, int square, int side) noexcept;
int king_square(const Position& position, int side);
std::vector<int> attackers(const Position& position, int square, int by);
bool attacked(const Position& position, int square, int by);
bool in_check(const Position& position, int side);
bool declarable(const Position& position);
std::vector<int> legal_moves(Position position, bool drops = true);
int apply_move(Position& position, int code);
Position parse_sfen(std::string_view text);
int move_label(int side, int code);
Outcome repetition_outcome(const std::array<bool, 2>& seen,
                           const std::array<bool, 2>& all_check) noexcept;
std::int64_t perft(Position position, int depth, bool declaration);

// Nodes are immutable after publication. Clones share history, never mutable state.
struct Node {
    Position position;
    Outcome outcome;
    std::vector<int> legal;
    std::string key;
    int ply{};
    int code{-1};
    bool check{};
    bool gave_check{};
    bool resignation{};
};

class Game {
public:
    explicit Game(Rules rules = {}, std::string_view sfen = start_sfen);
    const Node& state() const noexcept { return *history_.back(); }
    const Rules& rules() const noexcept { return rules_; }
    std::vector<int> legal_actions() const;
    bool is_legal(int code) const noexcept;
    bool can_declare() const noexcept;
    MoveResult play(int code);
    bool undo() noexcept;
    void resign();
    std::vector<int> moves() const;
    std::vector<std::string> history_keys() const;
    std::string history_key(int index) const;
    std::array<float, planes_size> encode(int player) const;
    std::array<float, action_size> legal_mask(int player) const;
    std::int64_t perft(int depth) const;
private:
    Rules rules_;
    // A flat ownership list avoids recursive destruction even at the move limit.
    std::vector<std::shared_ptr<const Node>> history_;
    void finish_node(Node& node) const;
    void validate_player(int player) const;
};
} // namespace shogi
#endif
