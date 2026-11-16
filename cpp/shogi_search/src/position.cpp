// SPDX-License-Identifier: MIT
#include "shogi_search/position.hpp"
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>

namespace shogi_search {
namespace {
// Clockwise compass, followed by the two forward knight jumps.
constexpr std::array<int, 10> dr{-1,-1,0,1,1,1,0,-1,-2,-2};
constexpr std::array<int, 10> dc{0,1,1,1,0,-1,-1,-1,-1,1};
constexpr int promotion(int t) { return t <= 4 ? t + 8 : t == 6 || t == 7 ? t + 7 : 0; }
constexpr int base(int t) { return t >= 13 ? t - 7 : t >= 9 ? t - 8 : t; }
constexpr bool zone(int sq, int side) { return side ? sq >= 54 : sq < 27; }
constexpr bool dead(int t, int sq, int side) {
    const int r = side ? 8 - sq / 9 : sq / 9;
    return ((t == 1 || t == 2) && r == 0) || (t == 3 && r < 2);
}
constexpr bool slide(int t, int d) {
    return (t == 2 && d == 0) || ((t == 6 || t == 13) && d < 8 && d % 2 == 1)
        || ((t == 7 || t == 14) && d < 8 && d % 2 == 0);
}
constexpr bool step(int t, int d) {
    if (t == 1) return d == 0;
    if (t == 3) return d >= 8;
    if (t == 4) return d == 0 || (d < 8 && d % 2 == 1);
    if (t == 5 || (t >= 9 && t <= 12)) return d == 0 || d == 1 || d == 2 || d == 4 || d == 6 || d == 7;
    if (t == 8) return d < 8;
    if (t == 13) return d < 8 && d % 2 == 0;
    if (t == 14) return d < 8 && d % 2 == 1;
    return false;
}
struct Geometry {
    std::array<std::array<int, 10>, 81> next{};
    std::array<std::array<unsigned, 8>, 2> near{}, far{};
    Geometry() {
        for (int s = 0; s < 81; ++s) for (int d = 0; d < 10; ++d) {
            const int r = s / 9 + dr[d], c = s % 9 + dc[d];
            next[s][d] = r < 0 || r > 8 || c < 0 || c > 8 ? -1 : r * 9 + c;
        }
        for (int side = 0; side < 2; ++side) for (int d = 0; d < 8; ++d) {
            const int toward = (d + (side ? 0 : 4)) % 8;
            for (int t = 1; t <= 14; ++t) {
                if (step(t, toward) || slide(t, toward)) near[side][d] |= 1U << t;
                if (slide(t, toward)) far[side][d] |= 1U << t;
            }
        }
    }
};
const Geometry geometry;
std::uint64_t random_word(std::uint64_t& state) noexcept {
    // A fixed SplitMix64 stream gives reproducible, machine-independent keys.
    std::uint64_t x = (state += 0x9e3779b97f4a7c15ULL);
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
struct Zobrist {
    std::array<std::array<std::uint64_t, 32>, 81> cells{};
    std::array<std::array<std::array<std::uint64_t, 19>, 7>, 2> hands{};
    std::uint64_t turn{};
    Zobrist() {
        std::uint64_t seed = 0x484c30303439ULL;
        for (auto& sq : cells) for (auto& value : sq) value = random_word(seed);
        for (auto& side : hands) for (auto& piece : side) for (auto& value : piece) value = random_word(seed);
        turn = random_word(seed);
    }
};
const Zobrist zobrist;
}
void MoveList::add(int code) {
    if (size == static_cast<int>(values.size())) throw shogi::Error("Search move buffer overflow");
    values[size++] = code;
}
Position::Position(const shogi::Position& p) : p_(p) {
    for (int sq = 0; sq < 81; ++sq) if (p.board[sq]) {
        const int side = p.board[sq] >> 4;
        occupied_[side][sq / 64] |= std::uint64_t{1} << (sq % 64);
        if ((p.board[sq] & 15) == 8) kings_[side] = sq;
    }
    key_ = recompute_key();
}
std::uint64_t Position::recompute_key() const noexcept {
    std::uint64_t key = p_.turn ? zobrist.turn : 0;
    for (int sq = 0; sq < 81; ++sq) if (p_.board[sq]) key ^= zobrist.cells[sq][p_.board[sq]];
    for (int s = 0; s < 2; ++s) for (int h = 0; h < 7; ++h) key ^= zobrist.hands[s][h][p_.hands[s][h]];
    return key;
}
void Position::cell(int sq, int value) noexcept {
    const int old = p_.board[sq];
    const auto bit = std::uint64_t{1} << (sq % 64);
    if (old) { key_ ^= zobrist.cells[sq][old]; occupied_[old >> 4][sq / 64] &= ~bit; }
    p_.board[sq] = value;
    if (value) {
        key_ ^= zobrist.cells[sq][value];
        occupied_[value >> 4][sq / 64] |= bit;
        if ((value & 15) == 8) kings_[value >> 4] = sq;
    }
}
void Position::hand(int side, int index, int delta) noexcept {
    int& n = p_.hands[side][index];
    key_ ^= zobrist.hands[side][index][n];
    n += delta;
    key_ ^= zobrist.hands[side][index][n];
}
bool Position::attacked(int sq, int by) const noexcept {
    for (int d = 0; d < 8; ++d) {
        int at = geometry.next[sq][d];
        unsigned mask = geometry.near[by][d];
        while (at >= 0) {
            const int c = p_.board[at];
            if (c) {
                if (c >> 4 == by && (mask & (1U << (c & 15)))) return true;
                break;
            }
            mask = geometry.far[by][d];
            at = geometry.next[at][d];
        }
    }
    const int r = sq / 9 + (by ? -2 : 2);
    if (r >= 0 && r <= 8) for (int c : {sq % 9 - 1, sq % 9 + 1})
        if (c >= 0 && c <= 8 && p_.board[r * 9 + c] == 3 + 16 * by) return true;
    return false;
}
bool Position::declarable() const noexcept {
    const int side = p_.turn;
    if (!zone(kings_[side], side) || check()) return false;
    int count = 0, points = 0;
    for (int sq = side ? 54 : 0; sq < (side ? 81 : 27); ++sq) {
        const int c = p_.board[sq], t = c & 15;
        if (!c || c >> 4 != side || t == 8) continue;
        ++count;
        points += base(t) == 6 || base(t) == 7 ? 5 : 1;
    }
    for (int h = 0; h < 7; ++h) points += p_.hands[side][h] * (h >= 5 ? 5 : 1);
    return count >= 10 && points >= (side ? 27 : 28);
}
MoveList Position::legal(bool declaration, bool drops) {
    MoveList moves;
    const int side = p_.turn, king = kings_[side];
    const bool checked = check();
    std::array<bool, 81> pinned{};
    for (int d = 0; d < 8; ++d) {
        int candidate = -1;
        for (int sq = geometry.next[king][d]; sq >= 0; sq = geometry.next[sq][d]) {
            const int c = p_.board[sq];
            if (!c) continue;
            if (candidate < 0 && c >> 4 == side) { candidate = sq; continue; }
            if (candidate >= 0 && c >> 4 != side && (geometry.far[1-side][d] & (1U << (c & 15))))
                pinned[candidate] = true;
            break;
        }
    }
    for (int word = 0; word < 2; ++word) {
        auto bits = occupied_[side][word];
        while (bits) {
            const int src = 64 * word + std::countr_zero(bits);
            bits &= bits - 1;
            const int cell_value = p_.board[src], t = cell_value & 15;
            for (int direction = 0; direction < 10; ++direction) {
                const bool sliding = slide(t, direction);
                if (!sliding && !step(t, direction)) continue;
                const int oriented = side && direction < 8 ? (direction + 4) % 8 : direction;
                int dst = -1;
                if (side && direction >= 8) {
                    const int r = src / 9 + 2, c = src % 9 - dc[direction];
                    if (r < 9 && c >= 0 && c < 9) dst = r * 9 + c;
                } else dst = geometry.next[src][oriented];
                while (dst >= 0) {
                    const int captured = p_.board[dst];
                    if (captured && (captured >> 4 == side || (captured & 15) == 8)) break;
                    bool safe = true;
                    if (checked || pinned[src] || t == 8) {
                        p_.board[src] = 0; p_.board[dst] = cell_value;
                        safe = !attacked(t == 8 ? dst : king, 1-side);
                        p_.board[src] = cell_value; p_.board[dst] = captured;
                    }
                    if (safe) {
                        const int code = (src * 81 + dst) * 2;
                        if (!dead(t, dst, side)) moves.add(code);
                        if (promotion(t) && (zone(src, side) || zone(dst, side))) moves.add(code + 1);
                    }
                    if (captured || !sliding) break;
                    dst = geometry.next[dst][oriented];
                }
            }
        }
    }
    if (drops) {
        unsigned pawn_files = 0;
        for (int sq = 0; sq < 81; ++sq) if (p_.board[sq] == 1 + 16 * side) pawn_files |= 1U << (sq % 9);
        for (int h = 0; h < 7; ++h) if (p_.hands[side][h]) for (int dst = 0; dst < 81; ++dst) {
            if (p_.board[dst] || dead(h+1, dst, side) || (h == 0 && (pawn_files & (1U << (dst % 9))))) continue;
            const int code = shogi::drop_base + h * 81 + dst;
            const bool pawn_check = h == 0 && dst + (side ? 9 : -9) == kings_[1-side];
            if (checked || pawn_check) {
                const auto undo = make(code);
                bool safe = !attacked(king, 1-side);
                // An adjacent pawn check has no interposition/drop evasion.
                if (safe && pawn_check) safe = legal(false, false).size != 0;
                unmake(undo);
                if (!safe) continue;
            }
            moves.add(code);
        }
    }
    if (declaration && declarable()) moves.add(shogi::declare);
    std::sort(moves.begin(), moves.end());
    return moves;
}
Undo Position::make(int code) noexcept {
    Undo u; u.code = code; u.key = key_;
    const int side = p_.turn;
    ++p_.move_number;
    if (code == shogi::declare) return u;
    if (code >= shogi::drop_base) {
        u.hand = (code - shogi::drop_base) / 81;
        u.target = (code - shogi::drop_base) % 81;
        hand(side, u.hand, -1);
        cell(u.target, u.hand + 1 + 16 * side);
    } else if (code >= 0) {
        u.source = code / 162; u.target = code / 2 % 81;
        u.moved = p_.board[u.source]; u.captured = p_.board[u.target];
        if (u.captured) { u.hand = base(u.captured & 15) - 1; hand(side, u.hand, 1); }
        cell(u.source, 0);
        cell(u.target, code & 1 ? promotion(u.moved & 15) + side * 16 : u.moved);
    }
    p_.turn ^= 1; key_ ^= zobrist.turn;
    return u;
}
void Position::unmake(const Undo& u) noexcept {
    --p_.move_number;
    if (u.code == shogi::declare) return;
    p_.turn ^= 1;
    if (u.source >= 0) {
        cell(u.target, u.captured); cell(u.source, u.moved);
        if (u.hand >= 0) hand(p_.turn, u.hand, -1);
    } else if (u.target >= 0) { cell(u.target, 0); hand(p_.turn, u.hand, 1); }
    key_ = u.key;
}
int piece_value(int t) noexcept {
    constexpr std::array<int, 15> values{0,100,300,320,450,550,800,1000,0,300,500,520,650,1000,1200};
    return values[t];
}
int square_value(int c, int sq) noexcept {
    if (!c) return 0;
    const int relative = c >> 4 ? 80-sq : sq, t = c & 15;
    // Bootstrap constants, not fitted: 2 cp/rank advancement, 2 cp/file centrality.
    const int pst = t == 8 ? 0 : 2 * (8-relative/9) + 2 * (4-std::abs(relative%9-4));
    return (c >> 4 ? -1 : 1) * (piece_value(t) + pst);
}
int Position::gain(int code) const noexcept {
    if (code == shogi::declare) return mate_score;
    if (code >= shogi::drop_base) return 0;
    const int c = p_.board[code / 2 % 81] & 15;
    return piece_value(c) + piece_value(base(c)) + ((code & 1) ? 200 : 0);
}
int Position::see(int code) const {
    if (code >= shogi::drop_base) return gain(code);
    Position work(*this);
    const int target = code / 2 % 81;
    std::array<int, 42> gains{};
    gains[0] = gain(code);
    work.make(code);
    int count = 0;
    while (count < 40) {
        auto moves = work.legal(false, false);
        int cheapest = -1, value = infinity;
        for (int m : moves) if (m / 2 % 81 == target) {
            const int type = work.p_.board[m / 162] & 15;
            const int cost = type == 8 ? infinity - 1 : piece_value(type);
            // Prefer promotion for an equal-valued attacker.
            if (cost < value || (cost == value && (m & 1))) { cheapest = m; value = cost; }
        }
        if (cheapest < 0) break;
        ++count;
        gains[count] = work.gain(cheapest) - gains[count-1];
        work.make(cheapest);
    }
    while (count > 0) { gains[count-1] = -std::max(-gains[count-1], gains[count]); --count; }
    return gains[0];
}
Root::Root(const shogi::Game& game) : position(game.state().position), rules(game.rules()), outcome(game.state().outcome) {
    const auto keys = game.history_keys();
    history.reserve(keys.size() + max_ply + 2);
    for (const auto& key : keys) {
        Position p(shogi::parse_sfen(key + " 1"));
        history.push_back({p.key(), p.board().turn, p.check()});
    }
}
shogi::Outcome adjudicate(const Position& p, const MoveList& legal, const shogi::Rules& rules,
                         std::span<const Event> events) {
    if (!legal.size) return {1-p.board().turn, p.check() ? 1 : 2};
    int occurrences = 0;
    std::size_t first = events.size();
    for (std::size_t i = 0; i < events.size(); ++i) if (events[i].key == p.key()) {
        ++occurrences; first = std::min(first, i);
    }
    if (occurrences >= 4) {
        std::array<bool, 2> seen{}, checks{true,true};
        for (std::size_t i = first+1; i < events.size(); ++i) {
            const int actor = events[i-1].turn;
            seen[actor] = true; checks[actor] = checks[actor] && events[i].check;
        }
        const bool a = seen[0] && checks[0], b = seen[1] && checks[1];
        return a != b ? shogi::Outcome{a ? 1 : 0,4} : shogi::Outcome{-1,5};
    }
    if (rules.max_moves && events.size()-1 == static_cast<std::size_t>(rules.max_moves)) return {-1,6};
    return {};
}
std::int64_t perft(Position& p, int depth, bool declaration) {
    if (depth < 0) throw shogi::Error("Invalid depth");
    if (!depth) return 1;
    const auto moves = p.legal(declaration);
    if (depth == 1) return moves.size;
    std::int64_t count = 0;
    for (int code : moves) if (code != shogi::declare) {
        const auto u = p.make(code);
        const auto n = perft(p, depth-1, declaration);
        p.unmake(u);
        if (n > std::numeric_limits<std::int64_t>::max()-count) throw shogi::Error("Perft overflow");
        count += n;
    }
    return count;
}
}
