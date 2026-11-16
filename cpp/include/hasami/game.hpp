#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hasami {
constexpr int kSquares = 81;
constexpr int kActions = 6561;
constexpr std::uint64_t kHighMask = (std::uint64_t{1} << 17) - 1;

struct Bits {
    std::uint64_t lo = 0, hi = 0;
    bool operator==(const Bits&) const = default;
    explicit operator bool() const { return lo || hi; }
    bool test(int sq) const;
    void set(int sq);
    void clear(int sq);
    int count() const { return std::popcount(lo) + std::popcount(hi); }
    int first() const;
    Bits operator|(Bits b) const { return {lo | b.lo, hi | b.hi}; }
    Bits operator&(Bits b) const { return {lo & b.lo, hi & b.hi}; }
    Bits operator^(Bits b) const { return {lo ^ b.lo, hi ^ b.hi}; }
    Bits operator~() const { return {~lo, ~hi & kHighMask}; }
    Bits left(unsigned shift) const;
    Bits right(unsigned shift) const;
};
Bits neighbours(Bits board);
Bits edge_mask();

struct Rules {
    bool edge_capture = true;
    int capture_target = 5;
    int lead_target = 3;
    int repetition_limit = 4;
    bool no_moves_draw = false;
    void validate() const;
};

struct Position {
    Bits black, white;
    int turn = 0, pending = -1;
    bool operator==(const Position&) const = default;
    Bits pieces(int player) const { return player == 0 ? black : white; }
    int captures(int player) const { return 9 - pieces(1 - player).count(); }
    void validate() const;
    static Position initial();
};
struct PositionHash {
    std::size_t operator()(const Position& p) const noexcept;
};
// The hash is only a bucket selector; full position equality resolves collisions.
enum class Reason : int { Ongoing, FiveCaptures, LeadConfirmed, Repetition, NoMoves };
struct Outcome {
    int winner = -2;  // -2: ongoing, -1: draw, 0/1: winner.
    Reason reason = Reason::Ongoing;
    bool terminal() const { return winner != -2; }
};
struct MoveResult { int action; Bits captured; Outcome outcome; };
std::vector<int> geometric_actions(const Position& p);
Bits captured_after_move(Bits me, Bits enemy, int target, bool edge_capture);

class Game {
public:
    explicit Game(Rules rules = {}, Position initial = Position::initial());
    const Position& position() const { return position_; }
    const Rules& rules() const { return rules_; }
    const Outcome& outcome() const { return outcome_; }
    Bits last_capture() const { return last_capture_; }
    const std::vector<int>& moves() const { return moves_; }
    const std::vector<Position>& history() const { return keys_; }
    std::vector<int> legal_actions() const;
    MoveResult play(int action);
    bool undo();
private:
    struct Undo { Position position; Outcome outcome; Bits captured; };
    Rules rules_;
    Position position_;
    Outcome outcome_;
    Bits last_capture_;
    std::vector<Undo> undo_;
    std::vector<int> moves_;
    std::vector<Position> keys_;
    std::unordered_map<Position, int, PositionHash> counts_;
    void finish(int winner, Reason reason);
    void validate_setup() const;
    void adjudicate_setup();
    void decrement(const Position& key);
};
std::uint64_t perft(Game& game, int depth);
}  // namespace hasami
