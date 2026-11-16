#include "shogi/game.hpp"
#include <algorithm>
#include <array>
#include <limits>

namespace shogi {
namespace {
struct Direction { int row; int col; };
constexpr std::array<Direction, 10> directions{{
    {-1, 0}, {-1, -1}, {-1, 1}, {0, -1}, {0, 1}, {1, 0}, {1, -1}, {1, 1}, {-2, -1}, {-2, 1}
}};
struct Line { std::array<int, 8> squares{}; int size{}; };
struct Movement { std::array<int, 8> steps{}; int size{}; std::array<Line, 4> slides{}; int count{}; };
struct Mask { unsigned near{}; unsigned far{}; };
constexpr bool step_direction(int piece, int d) {
    switch (piece) {
    case 1: return d == 0;
    case 3: return d >= 8;
    case 4: return d == 0 || d == 1 || d == 2 || d == 6 || d == 7;
    case 5: case 9: case 10: case 11: case 12: return d < 6;
    case 8: return d < 8;
    case 13: return d == 0 || d == 3 || d == 4 || d == 5;
    case 14: return d == 1 || d == 2 || d == 6 || d == 7;
    default: return false;
    }
}
constexpr bool slide_direction(int piece, int d) {
    if (piece == 2) return d == 0;
    if (piece == 6 || piece == 13) return d == 1 || d == 2 || d == 6 || d == 7;
    if (piece == 7 || piece == 14) return d == 0 || d == 3 || d == 4 || d == 5;
    return false;
}
constexpr Line ray(int square, int dr, int dc) {
    Line result;
    int row = square / 9 + dr, col = square % 9 + dc;
    while (row >= 0 && row < 9 && col >= 0 && col < 9) {
        result.squares[result.size++] = row * 9 + col;
        row += dr;
        col += dc;
    }
    return result;
}
struct Tables {
    std::array<std::array<Line, 8>, 81> rays{};
    std::array<std::array<std::array<Movement, 81>, 15>, 2> moves{};
    std::array<std::array<Mask, 8>, 2> attacks{};
    std::array<std::array<std::array<int, 2>, 81>, 2> knights{};
    Tables() {
        for (int sq = 0; sq < 81; ++sq)
            for (int d = 0; d < 8; ++d)
                rays[sq][d] = ray(sq, directions[d].row, directions[d].col);
        for (int side = 0; side < 2; ++side) {
            const int sign = side == 0 ? 1 : -1;
            for (int p = 1; p <= 14; ++p) {
                for (int sq = 0; sq < 81; ++sq) {
                    auto& m = moves[side][p][sq];
                    for (int d = 0; d < 10; ++d) {
                        const auto line = ray(sq, sign * directions[d].row, sign * directions[d].col);
                        if (step_direction(p, d) && line.size) m.steps[m.size++] = line.squares[0];
                        if (slide_direction(p, d)) m.slides[m.count++] = line;
                    }
                }
            }
            for (int d = 0; d < 8; ++d) {
                const int dr = -sign * directions[d].row, dc = -sign * directions[d].col;
                for (int toward = 0; toward < 8; ++toward) {
                    if (directions[toward].row != dr || directions[toward].col != dc) continue;
                    for (int p = 1; p <= 14; ++p) {
                        if (step_direction(p, toward) || slide_direction(p, toward)) attacks[side][d].near |= 1U << p;
                        if (slide_direction(p, toward)) attacks[side][d].far |= 1U << p;
                    }
                }
            }
            for (int sq = 0; sq < 81; ++sq) {
                for (int k = 0; k < 2; ++k) {
                    const auto line = ray(sq, 2 * sign, k == 0 ? -1 : 1);
                    knights[side][sq][k] = line.size ? line.squares[0] : -1;
                }
            }
        }
    }
};
const Tables tables;

std::array<bool, 81> pinned_sources(const Position& p, int king) {
    std::array<bool, 81> pins{};
    for (int d = 0; d < 8; ++d) {
        const auto& line = tables.rays[king][d];
        int candidate = -1;
        for (int i = 0; i < line.size; ++i) {
            const int sq = line.squares[i], cell = p.board[sq];
            if (!cell) continue;
            if (candidate < 0 && cell >> 4 == p.turn) { candidate = sq; continue; }
            if (candidate >= 0 && cell >> 4 != p.turn &&
                (tables.attacks[1 - p.turn][d].far & (1U << (cell & 15)))) pins[candidate] = true;
            break;
        }
    }
    return pins;
}
} // namespace

int promoted(int p) noexcept {
    return p >= 1 && p <= 4 ? p + 8 : p == 6 || p == 7 ? p + 7 : 0;
}
int unpromoted(int p) noexcept { return p >= 13 ? p - 7 : p >= 9 ? p - 8 : p; }
bool in_zone(int sq, int side) noexcept { return side == 0 ? sq < 27 : sq >= 54; }
bool dead_square(int p, int sq, int side) noexcept {
    const int row = side == 0 ? sq / 9 : 8 - sq / 9;
    return ((p == 1 || p == 2) && row == 0) || (p == 3 && row < 2);
}
int king_square(const Position& p, int side) {
    const auto it = std::find(p.board.begin(), p.board.end(), 8 + 16 * side);
    if (it == p.board.end()) throw Error("Exactly one king per side is required");
    return static_cast<int>(it - p.board.begin());
}
std::vector<int> attackers(const Position& p, int square, int by) {
    std::vector<int> found;
    for (int d = 0; d < 8; ++d) {
        const auto& line = tables.rays[square][d];
        const auto mask = tables.attacks[by][d];
        for (int i = 0; i < line.size; ++i) {
            const int sq = line.squares[i], cell = p.board[sq];
            if (!cell) continue;
            if (cell >> 4 == by && ((i == 0 ? mask.near : mask.far) & (1U << (cell & 15)))) found.push_back(sq);
            break;
        }
    }
    for (const int sq : tables.knights[by][square])
        if (sq >= 0 && p.board[sq] == 3 + 16 * by) found.push_back(sq);
    std::sort(found.begin(), found.end());
    return found;
}
bool attacked(const Position& p, int square, int by) {
    for (int d = 0; d < 8; ++d) {
        const auto& line = tables.rays[square][d];
        const auto mask = tables.attacks[by][d];
        for (int i = 0; i < line.size; ++i) {
            const int cell = p.board[line.squares[i]];
            if (!cell) continue;
            if (cell >> 4 == by && ((i == 0 ? mask.near : mask.far) & (1U << (cell & 15)))) return true;
            break;
        }
    }
    for (const int sq : tables.knights[by][square])
        if (sq >= 0 && p.board[sq] == 3 + 16 * by) return true;
    return false;
}
bool in_check(const Position& p, int side) { return attacked(p, king_square(p, side), 1 - side); }

std::vector<int> legal_moves(Position p, bool drops) {
    const int side = p.turn, king = king_square(p, side);
    const bool checked = attacked(p, king, 1 - side);
    const auto pins = pinned_sources(p, king);
    std::vector<int> result;
    result.reserve(96);
    for (int source = 0; source < 81; ++source) {
        const int cell = p.board[source], piece = cell & 15;
        if (!cell || cell >> 4 != side) continue;
        const auto visit = [&](int target) {
            const int captured = p.board[target];
            if (captured && (captured >> 4 == side || (captured & 15) == 8)) return;
            if (checked || piece == 8 || pins[source]) {
                p.board[source] = 0;
                p.board[target] = cell;
                const bool safe = !attacked(p, piece == 8 ? target : king, 1 - side);
                p.board[source] = cell;
                p.board[target] = captured;
                if (!safe) return;
            }
            const int code = (source * 81 + target) * 2;
            if (!dead_square(piece, target, side)) result.push_back(code);
            if (promoted(piece) && (in_zone(source, side) || in_zone(target, side))) result.push_back(code + 1);
        };
        const auto& movement = tables.moves[side][piece][source];
        for (int i = 0; i < movement.size; ++i) visit(movement.steps[i]);
        for (int d = 0; d < movement.count; ++d) {
            const auto& line = movement.slides[d];
            for (int i = 0; i < line.size; ++i) {
                const int target = line.squares[i];
                visit(target);
                if (p.board[target]) break;
            }
        }
    }
    if (drops) {
        std::array<bool, 9> pawn_files{};
        for (int sq = 0; sq < 81; ++sq) if (p.board[sq] == 1 + 16 * side) pawn_files[sq % 9] = true;
        const int enemy_king = king_square(p, 1 - side);
        for (int hand = 0; hand < 7; ++hand) {
            if (!p.hands[side][hand]) continue;
            const int piece = hand + 1;
            for (int target = 0; target < 81; ++target) {
                if (p.board[target] || dead_square(piece, target, side) || (piece == 1 && pawn_files[target % 9])) continue;
                const bool pawn_check = piece == 1 && target + (side == 0 ? -9 : 9) == enemy_king;
                if (checked || pawn_check) {
                    p.board[target] = piece + 16 * side;
                    bool safe = !checked || !attacked(p, king, 1 - side);
                    if (safe && pawn_check) {
                        p.turn = 1 - side;
                        safe = !legal_moves(p, false).empty();
                        p.turn = side;
                    }
                    p.board[target] = 0;
                    if (!safe) continue;
                }
                result.push_back(drop_base + hand * 81 + target);
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}
int apply_move(Position& p, int code) {
    const int side = p.turn;
    if (code >= drop_base) {
        const int hand = (code - drop_base) / 81, target = (code - drop_base) % 81;
        --p.hands[side][hand];
        p.board[target] = hand + 1 + 16 * side;
        return 0;
    }
    const int source = code / 2 / 81, target = code / 2 % 81;
    const int cell = p.board[source], captured = p.board[target];
    if (captured) ++p.hands[side][unpromoted(captured & 15) - 1];
    p.board[source] = 0;
    p.board[target] = code & 1 ? promoted(cell & 15) + 16 * side : cell;
    return captured;
}
bool declarable(const Position& p) {
    const int side = p.turn, king = king_square(p, side);
    if (!in_zone(king, side) || attacked(p, king, 1 - side)) return false;
    int pieces = 0, points = 0;
    for (int sq = 0; sq < 81; ++sq) {
        const int cell = p.board[sq], piece = cell & 15;
        if (!cell || cell >> 4 != side || piece == 8 || !in_zone(sq, side)) continue;
        ++pieces;
        points += piece == 6 || piece == 7 || piece == 13 || piece == 14 ? 5 : 1;
    }
    for (int hand = 0; hand < 7; ++hand) points += p.hands[side][hand] * (hand >= 5 ? 5 : 1);
    return pieces >= 10 && points >= (side == 0 ? 28 : 27);
}
std::int64_t perft(Position p, int depth, bool declaration) {
    if (depth == 0) return 1;
    const auto actions = legal_moves(p);
    if (depth == 1) return static_cast<std::int64_t>(actions.size()) + (declaration && declarable(p));
    std::int64_t count = 0;
    for (const int code : actions) {
        auto next = p;
        apply_move(next, code);
        next.turn = 1 - p.turn;
        const auto children = perft(next, depth - 1, declaration);
        if (children > std::numeric_limits<std::int64_t>::max() - count) throw Error("Perft count overflow");
        count += children;
    }
    return count;
}
} // namespace shogi
