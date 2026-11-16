// SPDX-License-Identifier: MIT
#ifndef SHOGI_SEARCH_POSITION_HPP
#define SHOGI_SEARCH_POSITION_HPP
#include "shogi/game.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace shogi_search {
constexpr int max_ply = 128;
constexpr int mate_score = 1000000;
constexpr int infinity = 1100000;
struct MoveList {
    // Even the loose bound 40 board pieces * 32 targets + 7 * 81 drops fits.
    std::array<int, 2048> values;
    int size{};
    MoveList() = default;
    MoveList(const MoveList& other) : size(other.size) { std::copy_n(other.values.begin(),size,values.begin()); }
    MoveList& operator=(const MoveList& other) {
        if (this != &other) { size = other.size; std::copy_n(other.values.begin(),size,values.begin()); }
        return *this;
    }
    void add(int code);
    int* begin() { return values.data(); }
    int* end() { return begin() + size; }
    const int* begin() const { return values.data(); }
    const int* end() const { return begin() + size; }
};
struct Undo {
    int code{}, source{-1}, target{-1}, moved{}, captured{}, hand{-1};
    std::uint64_t key{};
};
class Position {
public:
    explicit Position(const shogi::Position& position);
    const shogi::Position& board() const noexcept { return p_; }
    std::uint64_t key() const noexcept { return key_; }
    std::uint64_t recompute_key() const noexcept;
    bool attacked(int square, int by) const noexcept;
    bool check() const noexcept { return attacked(kings_[p_.turn], 1 - p_.turn); }
    bool declarable() const noexcept;
    MoveList legal(bool declaration = true, bool drops = true);
    Undo make(int code) noexcept; // Requires a legal code, or -1 for a search-only null move.
    void unmake(const Undo& undo) noexcept;
    int see(int code) const;
    int gain(int code) const noexcept;
private:
    shogi::Position p_;
    std::array<int, 2> kings_{};
    std::array<std::array<std::uint64_t, 2>, 2> occupied_{};
    std::uint64_t key_{};
    void cell(int square, int value) noexcept;
    void hand(int side, int index, int delta) noexcept;
};
struct Event {
    std::uint64_t key;
    int turn;
    bool check;
};
struct Root {
    Position position;
    shogi::Rules rules;
    shogi::Outcome outcome;
    std::vector<Event> history;
    explicit Root(const shogi::Game& game);
};
shogi::Outcome adjudicate(const Position& position, const MoveList& legal,
                         const shogi::Rules& rules, std::span<const Event> history);
std::int64_t perft(Position& position, int depth, bool declaration = true);
int piece_value(int type) noexcept;
int square_value(int cell, int square) noexcept;
}
#endif
