#include "shogi/game.hpp"
#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace shogi {
namespace {
constexpr std::string_view letters = " PLNSGBRK";
bool digit(char ch) { return ch >= '0' && ch <= '9'; }
int piece_type(char ch) {
    const auto upper = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    const auto index = letters.find(upper);
    return index != std::string_view::npos && index != 0 ? static_cast<int>(index) : 0;
}
std::vector<std::string> split(std::string_view text, char separator) {
    std::vector<std::string> result;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == separator) {
            result.emplace_back(text.substr(start, i - start));
            start = i + 1;
        }
    }
    return result;
}
// Python str.split accepts Unicode whitespace, including ASCII information separators.
std::vector<std::string> fields(std::string_view text) {
    std::vector<std::string> result;
    std::string field;
    for (std::size_t i = 0; i < text.size();) {
        const auto start = i;
        auto cp = static_cast<unsigned char>(text[i++]);
        unsigned value = cp;
        int extra = 0;
        if ((cp & 0xe0) == 0xc0) { value = cp & 0x1f; extra = 1; }
        else if ((cp & 0xf0) == 0xe0) { value = cp & 0x0f; extra = 2; }
        else if ((cp & 0xf8) == 0xf0) { value = cp & 7; extra = 3; }
        for (int n = 0; n < extra && i < text.size(); ++n)
            value = (value << 6) | (static_cast<unsigned char>(text[i++]) & 0x3f);
        const bool space = (value >= 9 && value <= 13) || (value >= 28 && value <= 32) ||
            value == 0x85 || value == 0xa0 || value == 0x1680 || (value >= 0x2000 && value <= 0x200a) ||
            value == 0x2028 || value == 0x2029 || value == 0x202f || value == 0x205f || value == 0x3000;
        if (space) {
            if (!field.empty()) { result.push_back(std::move(field)); field.clear(); }
        } else {
            field.append(text.substr(start, i - start));
        }
    }
    if (!field.empty()) result.push_back(std::move(field));
    return result;
}
void validate_position(const Position& p) {
    std::array<int, 7> totals{};
    std::array<std::array<bool, 9>, 2> pawns{};
    for (int hand = 0; hand < 7; ++hand) totals[hand] = p.hands[0][hand] + p.hands[1][hand];
    for (int side = 0; side < 2; ++side)
        if (std::count(p.board.begin(), p.board.end(), 8 + 16 * side) != 1)
            throw Error("Exactly one king per side is required");
    for (int sq = 0; sq < 81; ++sq) {
        const int cell = p.board[sq], side = cell >> 4, piece = cell & 15;
        if (!cell) continue;
        if (dead_square(piece, sq, side)) throw Error("Dead unpromoted piece");
        if (piece == 1) {
            if (pawns[side][sq % 9]) throw Error("Two unpromoted pawns on a file");
            pawns[side][sq % 9] = true;
        }
        if (piece != 8) ++totals[unpromoted(piece) - 1];
    }
    for (int hand = 0; hand < 7; ++hand)
        if (totals[hand] > piece_limits[hand]) throw Error("Too many pieces");
    if (in_check(p, 1 - p.turn)) throw Error("The side not to move is in check");
}
} // namespace

void Rules::validate() const {
    if (max_moves < 0 || max_moves > max_game_moves) throw Error("Invalid max_moves");
    if (declaration != 0 && declaration != 1) throw Error("Invalid declaration rule");
}
Position parse_sfen(std::string_view text) {
    const auto parts = fields(text);
    if (parts.size() != 4) throw Error("SFEN must have four fields");
    const auto& number = parts[3];
    if ((parts[1] != "b" && parts[1] != "w") || number.empty() || number[0] < '1' || number[0] > '9' ||
        number.size() > 7 || !std::all_of(number.begin(), number.end(), digit))
        throw Error("Invalid SFEN turn or move number");
    const int move_number = std::stoi(number);
    if (move_number > max_game_moves) throw Error("Invalid SFEN turn or move number");
    const auto ranks = split(parts[0], '/');
    if (ranks.size() != 9) throw Error("SFEN must have nine ranks");
    Position p;
    p.turn = parts[1] == "w";
    p.move_number = move_number;
    for (int row = 0; row < 9; ++row) {
        int col = 0;
        const auto& rank = ranks[row];
        for (std::size_t i = 0; i < rank.size();) {
            char ch = rank[i++];
            if (ch >= '1' && ch <= '9') { col += ch - '0'; continue; }
            const bool promote = ch == '+';
            if (promote) {
                if (i == rank.size()) throw Error("Incomplete promotion marker");
                ch = rank[i++];
            }
            // U+017F uppercases to S in Python; the hand grammar remains ASCII.
            if (static_cast<unsigned char>(ch) == 0xc5 && i < rank.size() &&
                static_cast<unsigned char>(rank[i]) == 0xbf) { ch = 's'; ++i; }
            int piece = piece_type(ch);
            if (!piece) throw Error("Invalid SFEN piece");
            if (promote) {
                piece = promoted(piece);
                if (!piece) throw Error("Invalid promoted piece");
            }
            if (col < 9) p.board[row * 9 + col] = piece + (ch >= 'a' && ch <= 'z' ? 16 : 0);
            ++col;
        }
        if (col != 9) throw Error("SFEN rank must have nine cells");
    }
    if (parts[2] != "-") {
        struct Hand { int side; int hand; int count; };
        std::vector<Hand> tokens;
        const auto& hand = parts[2];
        for (std::size_t i = 0; i < hand.size();) {
            int count = 0;
            if (digit(hand[i])) {
                if (hand[i] == '0') throw Error("Invalid SFEN hands");
                while (i < hand.size() && digit(hand[i])) {
                    // Saturation avoids overflow; Position rejects out-of-range hand counts later.
                    count = std::min(100, count * 10 + hand[i++] - '0');
                }
            } else count = 1;
            if (i == hand.size()) throw Error("Invalid SFEN hands");
            const char ch = hand[i++];
            const int piece = piece_type(ch);
            if (piece < 1 || piece > 7) throw Error("Invalid SFEN hands");
            tokens.push_back({ch >= 'a' && ch <= 'z' ? 1 : 0, piece - 1, count});
        }
        if (tokens.empty()) throw Error("Invalid SFEN hands");
        for (const auto token : tokens) {
            auto& count = p.hands[token.side][token.hand];
            if (count) throw Error("Duplicate hand piece");
            count = token.count;
        }
    }
    for (const auto& hand : p.hands)
        for (int i = 0; i < 7; ++i)
            if (hand[i] > piece_limits[i]) throw Error("Invalid hand count");
    validate_position(p);
    return p;
}
std::string Position::key() const {
    std::string text;
    text.reserve(128);
    for (int row = 0; row < 9; ++row) {
        if (row) text += '/';
        int empty = 0;
        for (int col = 0; col < 9; ++col) {
            const int cell = board[row * 9 + col], piece = cell & 15;
            if (!cell) { ++empty; continue; }
            if (empty) { text += static_cast<char>('0' + empty); empty = 0; }
            if (piece >= 9) text += '+';
            char letter = letters[unpromoted(piece)];
            if (cell >> 4) letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
            text += letter;
        }
        if (empty) text += static_cast<char>('0' + empty);
    }
    text += turn ? " w " : " b ";
    const auto hand_start = text.size();
    for (int side = 0; side < 2; ++side) {
        for (int hand = 6; hand >= 0; --hand) {
            const int count = hands[side][hand];
            if (!count) continue;
            if (count > 1) text += std::to_string(count);
            char letter = letters[hand + 1];
            if (side) letter = static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
            text += letter;
        }
    }
    if (text.size() == hand_start) text += '-';
    return text;
}
std::string Position::sfen() const { return key() + ' ' + std::to_string(move_number); }
int move_label(int side, int code) {
    if (code == declare) return action_size - 1;
    const int to = code >= drop_base ? (code - drop_base) % 81 : code / 2 % 81;
    const int target = side ? 80 - to : to;
    if (code >= drop_base) return (20 + (code - drop_base) / 81) * 81 + target;
    const int from = code / 2 / 81, source = side ? 80 - from : from;
    const int dr = target / 9 - source / 9, dc = target % 9 - source % 9;
    int direction = 0;
    if (dr == -2 && (dc == -1 || dc == 1)) direction = dc == -1 ? 8 : 9;
    else if (dr < 0) direction = dc < 0 ? 1 : dc > 0 ? 2 : 0;
    else if (dr == 0) direction = dc < 0 ? 3 : 4;
    else direction = dc < 0 ? 6 : dc > 0 ? 7 : 5;
    return (direction + 10 * (code & 1)) * 81 + target;
}
} // namespace shogi
