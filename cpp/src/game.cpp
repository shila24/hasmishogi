#include "hasami/game.hpp"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace hasami {
namespace {
using RayTable = std::array<std::array<std::vector<int>, 4>, 81>;
const RayTable& rays() {
    static const RayTable table = [] {
        RayTable out;
        constexpr int dr[] = {-1, 1, 0, 0};
        constexpr int dc[] = {0, 0, -1, 1};
        for (int s = 0; s < 81; ++s)
            for (int d = 0; d < 4; ++d) {
                int r = s / 9 + dr[d], c = s % 9 + dc[d];
                while (r >= 0 && r < 9 && c >= 0 && c < 9) {
                    out[s][d].push_back(r * 9 + c);
                    r += dr[d]; c += dc[d];
                }
            }
        return out;
    }();
    return table;
}
Bits file_mask(int col) {
    Bits out;
    for (int row = 0; row < 9; ++row) out.set(row * 9 + col);
    return out;
}
void valid_square(int sq) {
    if (sq < 0 || sq >= 81) throw std::invalid_argument("Square index out of range");
}
}
bool Bits::test(int sq) const {
    valid_square(sq);
    return sq < 64 ? ((lo >> sq) & 1) != 0 : ((hi >> (sq - 64)) & 1) != 0;
}
void Bits::set(int sq) {
    valid_square(sq);
    if (sq < 64) lo |= std::uint64_t{1} << sq;
    else hi |= std::uint64_t{1} << (sq - 64);
}
void Bits::clear(int sq) {
    valid_square(sq);
    if (sq < 64) lo &= ~(std::uint64_t{1} << sq);
    else hi &= ~(std::uint64_t{1} << (sq - 64));
}
int Bits::first() const {
    if (lo) return std::countr_zero(lo);
    if (hi) return 64 + std::countr_zero(hi);
    throw std::invalid_argument("Cannot scan an empty bitboard");
}
Bits Bits::left(unsigned n) const {
    if (n == 0) return *this;
    if (n >= 128) return {};
    if (n >= 64) return {0, (lo << (n - 64)) & kHighMask};
    return {lo << n, ((hi << n) | (lo >> (64 - n))) & kHighMask};
}
Bits Bits::right(unsigned n) const {
    if (n == 0) return *this;
    if (n >= 128) return {};
    if (n >= 64) return {hi >> (n - 64), 0};
    return {(lo >> n) | (hi << (64 - n)), hi >> n};
}
Bits neighbours(Bits b) {
    static const Bits left = file_mask(0), right = file_mask(8);
    return ((b & ~right).left(1) | (b & ~left).right(1)
            | b.left(9) | b.right(9));
}
Bits edge_mask() {
    static const Bits result = [] {
        Bits b = file_mask(0) | file_mask(8);
        for (int c = 0; c < 9; ++c) { b.set(c); b.set(72 + c); }
        return b;
    }();
    return result;
}
void Rules::validate() const {
    if (capture_target < 0 || capture_target > 9 || lead_target < 0 || lead_target > 9)
        throw std::invalid_argument("Invalid capture or lead target");
    if (repetition_limit != 0 && (repetition_limit < 2 || repetition_limit > 100))
        throw std::invalid_argument("Invalid repetition limit");
}
Position Position::initial() {
    Position p;
    for (int i = 0; i < 9; ++i) { p.black.set(72 + i); p.white.set(i); }
    return p;
}
void Position::validate() const {
    if ((black.hi | white.hi) & ~kHighMask)
        throw std::invalid_argument("Off-board bits are set");
    if (black & white) throw std::invalid_argument("Bitboards overlap");
    if (black.count() > 9 || white.count() > 9)
        throw std::invalid_argument("Too many pieces");
    if ((turn != 0 && turn != 1) || pending < -1 || pending > 1 || pending == turn)
        throw std::invalid_argument("Invalid turn or pending leader");
}
std::size_t PositionHash::operator()(const Position& p) const noexcept {
    std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
    const std::uint64_t words[] = {p.black.lo, p.black.hi, p.white.lo, p.white.hi,
                                  static_cast<std::uint64_t>(p.turn),
                                  static_cast<std::uint64_t>(p.pending + 1)};
    for (auto x : words) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        hash ^= (x ^ (x >> 31)) + (hash << 6) + (hash >> 2);
    }
    return static_cast<std::size_t>(hash);
}
std::vector<int> geometric_actions(const Position& p) {
    std::vector<int> actions;
    actions.reserve(144);
    Bits mine = p.pieces(p.turn), occupied = p.black | p.white;
    while (mine) {
        const int source = mine.first(); mine.clear(source);
        for (const auto& ray : rays()[source])
            for (int target : ray) {
                if (occupied.test(target)) break;
                actions.push_back(source * 81 + target);
            }
    }
    std::sort(actions.begin(), actions.end());
    return actions;
}
Bits captured_after_move(Bits me, Bits enemy, int target, bool edge_capture) {
    Bits removed;
    for (const auto& ray : rays()[target]) {
        Bits run;
        for (int sq : ray) {
            if (enemy.test(sq)) run.set(sq);
            else {
                if (me.test(sq)) removed = removed | run;
                break;
            }
        }
    }
    if (edge_capture) {
        Bits unseen = enemy;
        while (unseen) {
            Bits component; component.set(unseen.first());
            while (true) {
                Bits next = component | (neighbours(component) & enemy);
                if (next == component) break;
                component = next;
            }
            unseen = unseen & ~component;
            Bits boundary = neighbours(component) & ~component;
            if ((component & edge_mask()) && boundary.test(target) && !(boundary & ~me))
                removed = removed | component;
        }
    }
    return removed;
}
Game::Game(Rules rules, Position initial) : rules_(rules), position_(initial) {
    rules_.validate(); position_.validate(); validate_setup(); adjudicate_setup();
    keys_.push_back(position_); counts_[position_] = 1;
}
void Game::validate_setup() const {
    int lead = position_.captures(0) - position_.captures(1);
    if (position_.pending != -1) {
        int signed_lead = position_.pending == 0 ? lead : -lead;
        if (!rules_.lead_target || signed_lead < rules_.lead_target)
            throw std::invalid_argument("Pending leader inconsistent with material");
    }
    if (rules_.capture_target && position_.captures(0) >= rules_.capture_target &&
        position_.captures(1) >= rules_.capture_target)
        throw std::invalid_argument("Both players cannot already have won");
    bool immediate = rules_.capture_target &&
        std::max(position_.captures(0), position_.captures(1)) >= rules_.capture_target;
    if (rules_.lead_target && std::abs(lead) >= rules_.lead_target && !immediate &&
        position_.pending != (lead > 0 ? 0 : 1))
        throw std::invalid_argument("A decisive setup lead needs an explicit reply state");
}
void Game::adjudicate_setup() {
    if (rules_.capture_target)
        for (int player = 0; player < 2; ++player)
            if (position_.captures(player) >= rules_.capture_target) {
                finish(player, Reason::FiveCaptures); return;
            }
    if (geometric_actions(position_).empty()) {
        if (position_.pending != -1) finish(position_.pending, Reason::LeadConfirmed);
        else finish(rules_.no_moves_draw ? -1 : 1 - position_.turn, Reason::NoMoves);
    }
}
void Game::finish(int winner, Reason reason) {
    position_.pending = -1;
    outcome_ = {winner, reason};
}
std::vector<int> Game::legal_actions() const {
    return outcome_.terminal() ? std::vector<int>{} : geometric_actions(position_);
}
void Game::decrement(const Position& key) {
    auto it = counts_.find(key);
    if (it == counts_.end()) throw std::logic_error("Missing repetition key");
    if (--it->second == 0) counts_.erase(it);
}
MoveResult Game::play(int action) {
    if (action < 0 || action >= kActions) throw std::invalid_argument("Action out of range");
    if (outcome_.terminal()) throw std::invalid_argument("Game already ended");
    const int source = action / 81, target = action % 81;
    Position old = position_;
    Bits me = old.pieces(old.turn), enemy = old.pieces(1 - old.turn), occupied = me | enemy;
    if (!me.test(source) || occupied.test(target))
        throw std::invalid_argument("Invalid source or occupied destination");
    bool valid = false;
    for (const auto& ray : rays()[source]) {
        for (int sq : ray) {
            if (occupied.test(sq)) break;
            if (sq == target) { valid = true; break; }
        }
        if (valid) break;
    }
    if (!valid) throw std::invalid_argument("Move is blocked or not orthogonal");
    me.clear(source); me.set(target);
    Bits captured = captured_after_move(me, enemy, target, rules_.edge_capture);
    enemy = enemy & ~captured;
    undo_.push_back({position_, outcome_, last_capture_});
    moves_.push_back(action);
    position_ = old.turn == 0 ? Position{me, enemy, 1, -1} : Position{enemy, me, 0, -1};
    last_capture_ = captured; outcome_ = {};
    if (rules_.capture_target && position_.captures(old.turn) >= rules_.capture_target)
        finish(old.turn, Reason::FiveCaptures);
    else if (old.pending != -1 && position_.captures(old.pending) -
             position_.captures(1 - old.pending) >= rules_.lead_target)
        finish(old.pending, Reason::LeadConfirmed);
    else if (rules_.lead_target && position_.captures(old.turn) -
             position_.captures(1 - old.turn) >= rules_.lead_target)
        position_.pending = old.turn;
    Position key = position_;
    keys_.push_back(key); ++counts_[key];
    if (!outcome_.terminal()) {
        if (rules_.repetition_limit && counts_[key] >= rules_.repetition_limit)
            finish(-1, Reason::Repetition);
        else if (geometric_actions(position_).empty()) {
            if (position_.pending != -1) finish(position_.pending, Reason::LeadConfirmed);
            else finish(rules_.no_moves_draw ? -1 : old.turn, Reason::NoMoves);
        }
    }
    if (!(position_ == key)) {
        decrement(key); keys_.back() = position_; ++counts_[position_];
    }
    return {action, captured, outcome_};
}
bool Game::undo() {
    if (undo_.empty()) return false;
    decrement(keys_.back()); keys_.pop_back();
    Undo old = undo_.back(); undo_.pop_back(); moves_.pop_back();
    position_ = old.position; outcome_ = old.outcome; last_capture_ = old.captured;
    return true;
}
std::uint64_t perft(Game& game, int depth) {
    if (depth < 0) throw std::invalid_argument("Depth must be nonnegative");
    if (depth == 0) return 1;
    if (game.outcome().terminal()) return 0;
    std::uint64_t nodes = 0;
    for (int a : game.legal_actions()) {
        game.play(a);
        nodes += perft(game, depth - 1);
        game.undo();
    }
    return nodes;
}
}  // namespace hasami
