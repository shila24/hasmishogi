#include "hasami/search.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace hasami {
namespace {
constexpr int kWin = 1000000;
constexpr int kInfinity = 2000000;
using Clock = std::chrono::steady_clock;
struct Stop {};

class Search {
public:
    Search(SearchLimits value, const std::atomic_bool* token) : limits(value), cancel(token) {}
    SearchLimits limits;
    const std::atomic_bool* cancel;
    Clock::time_point start = Clock::now();
    std::uint64_t nodes = 0;
    // Store moves only. History-dependent scores are deliberately NOT cached.
    std::unordered_map<Position, int, PositionHash> hints;

    double elapsed() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    void check() const {
        if ((cancel && cancel->load()) || nodes >= limits.node_limit || elapsed() >= limits.time_ms)
            throw Stop{};
    }
    std::vector<int> ordered(const Game& game) {
        auto actions = game.legal_actions();
        const auto& p = game.position();
        int hint = -1;
        auto it = hints.find(p);
        if (it != hints.end()) hint = it->second;
        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(actions.size());
        for (int a : actions) {
            check();
            Bits me = p.pieces(p.turn);
            me.clear(a / 81); me.set(a % 81);
            Bits taken = captured_after_move(me, p.pieces(1 - p.turn), a % 81,
                                             game.rules().edge_capture);
            int priority = taken.count() * 100;
            if (a == hint) priority += 100000;
            ranked.emplace_back(-priority, a);
        }
        std::sort(ranked.begin(), ranked.end());
        actions.clear();
        for (auto [priority, action] : ranked) { (void)priority; actions.push_back(action); }
        return actions;
    }
    std::pair<int, int> negamax(Game& game, int depth, int alpha, int beta, int ply) {
        check(); ++nodes;
        if (game.outcome().terminal()) {
            if (game.outcome().winner == -1) return {0, -1};
            int score = game.outcome().winner == game.position().turn ? kWin - ply : -kWin + ply;
            return {score, -1};
        }
        // A reply to a pending lead is compulsory. Never use a static stand-pat here.
        if (depth <= 0 && game.position().pending == -1) return {evaluate(game.position()), -1};
        if (depth <= 0) depth = 1;
        int best = -kInfinity, best_action = -1;
        for (int a : ordered(game)) {
            game.play(a);
            int score;
            try { score = -negamax(game, depth - 1, -beta, -alpha, ply + 1).first; }
            catch (...) { game.undo(); throw; }
            game.undo();
            if (score > best) { best = score; best_action = a; }
            alpha = std::max(alpha, score);
            if (alpha >= beta) break;
        }
        if (hints.size() > 50000) hints.clear();
        hints[game.position()] = best_action;
        return {best, best_action};
    }
};
}
void SearchLimits::validate() const {
    if (max_depth < 1 || max_depth > 64 || time_ms < 1 || time_ms > 3600000 || node_limit < 1)
        throw std::invalid_argument("Invalid search limits");
}
int evaluate(const Position& p) {
    int score = 1000 * (p.black.count() - p.white.count());
    Position black = p, white = p;
    black.turn = 0; white.turn = 1;
    score += 3 * (static_cast<int>(geometric_actions(black).size()) -
                  static_cast<int>(geometric_actions(white).size()));
    for (int side = 0; side < 2; ++side) {
        Bits b = p.pieces(side);
        while (b) {
            int sq = b.first(); b.clear(sq);
            int centrality = 8 - std::abs(sq / 9 - 4) - std::abs(sq % 9 - 4);
            score += (side == 0 ? 1 : -1) * centrality;
        }
    }
    return p.turn == 0 ? score : -score;
}
SearchResult AlphaBetaAgent::choose(const Game& source, const SearchLimits& limits,
                                   const std::atomic_bool* cancel) {
    limits.validate();
    Search search{limits, cancel};
    auto legal = source.legal_actions();
    if (legal.empty()) throw std::invalid_argument("No action is available");
    SearchResult result;
    result.action = legal.front(); result.score = evaluate(source.position());
    Game game = source;
    for (int depth = 1; depth <= limits.max_depth; ++depth) {
        try {
            auto [score, action] = search.negamax(game, depth, -kInfinity, kInfinity, 0);
            result.action = action; result.score = score; result.depth = depth;
        } catch (const Stop&) { break; }
    }
    result.nodes = search.nodes;
    result.elapsed_ms = search.elapsed();
    result.completed = result.depth == limits.max_depth;
    return result;
}
SearchResult RandomAgent::choose(const Game& game, const SearchLimits& limits,
                                const std::atomic_bool* cancel) {
    limits.validate();
    if (cancel && cancel->load()) throw std::runtime_error("Agent cancelled");
    auto legal = game.legal_actions();
    if (legal.empty()) throw std::invalid_argument("No action is available");
    std::mt19937_64 rng(seed_++);
    std::uniform_int_distribution<std::size_t> choice(0, legal.size() - 1);
    SearchResult result; result.action = legal[choice(rng)]; result.completed = true;
    return result;
}
}  // namespace hasami
