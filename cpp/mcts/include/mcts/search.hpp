#ifndef MCTS_SEARCH_HPP
#define MCTS_SEARCH_HPP

#include "hasami/game.hpp"
#include "shogi/game.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mcts {
// Search-only reason bits, independent of either engine's outcome enum.
inline constexpr std::uint32_t draw_repetition = 1, draw_max_moves = 2;
struct DrawPolicy {
    double value{};
    std::uint32_t reasons{};
};

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Match NumPy's contiguous float64 pairwise reduction, including its eight lanes.
// Selection must also be compiled without reassociation or fused multiply-add.
inline double pairwise_sum(std::span<const double> values) {
    const auto n = values.size();
    if (n < 8) {
        double sum = -0.0;
        for (const auto value : values) sum += value;
        return sum;
    }
    if (n <= 128) {
        std::array<double, 8> sums{};
        std::copy_n(values.begin(), 8, sums.begin());
        std::size_t i = 8;
        for (; i < n - n % 8; i += 8)
            for (std::size_t j = 0; j < 8; ++j) sums[j] += values[i + j];
        double sum = ((sums[0] + sums[1]) + (sums[2] + sums[3]))
                   + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
        for (; i < n; ++i) sum += values[i];
        return sum;
    }
    auto half = n / 2;
    half -= half % 8;
    return pairwise_sum(values.first(half)) + pairwise_sum(values.subspan(half));
}

struct HasamiAdapter {
    static constexpr int action_size = hasami::kActions;
    static constexpr int planes_size = 9 * hasami::kSquares;
    hasami::Game game;
    int player() const { return game.position().turn; }
    bool terminal() const { return game.outcome().terminal(); }
    std::uint32_t draw_reason() const {
        return game.outcome().winner == -1 && game.outcome().reason == hasami::Reason::Repetition
            ? draw_repetition : 0;
    }
    double value() const {
        const auto winner = game.outcome().winner;
        return winner == -1 ? 0.0 : winner == player() ? 1.0 : -1.0;
    }
    // Neither production adapter truncates. Training caps belong to the runner.
    bool step(int action) { game.play(action); return false; }
    void undo() { game.undo(); }
    void observe(std::span<float> planes, std::span<float> mask) const {
        std::fill(planes.begin(), planes.end(), 0.0F);
        std::fill(mask.begin(), mask.end(), 0.0F);
        const auto& p = game.position();
        for (int s = 0; s < hasami::kSquares; ++s) {
            planes[s] = p.pieces(p.turn).test(s) ? 1.0F : 0.0F;
            planes[81 + s] = p.pieces(1 - p.turn).test(s) ? 1.0F : 0.0F;
        }
        const auto repetitions = std::count(game.history().begin(), game.history().end(), p);
        const std::array<double, 7> constants{
            p.pending == p.turn ? 1.0 : 0.0, p.pending == 1 - p.turn ? 1.0 : 0.0,
            static_cast<double>(p.turn), p.captures(p.turn) / 9.0, p.captures(1 - p.turn) / 9.0,
            game.rules().repetition_limit ? static_cast<double>(repetitions) / game.rules().repetition_limit : 0.0,
            1.0};
        for (std::size_t i = 0; i < constants.size(); ++i)
            std::fill_n(planes.begin() + (i + 2) * 81, 81, static_cast<float>(constants[i]));
        for (const auto action : game.legal_actions()) mask[action] = 1.0F;
    }
};

struct ShogiAdapter {
    static constexpr int action_size = shogi::action_size;
    static constexpr int planes_size = shogi::planes_size;
    shogi::Game game;
    int player() const { return game.state().position.turn; }
    bool terminal() const { return game.state().outcome.terminal(); }
    std::uint32_t draw_reason() const {
        const auto outcome = game.state().outcome;
        if (outcome.winner != -1) return 0;
        // shogi-std-v1: reason 5 is repetition, reason 6 is the rule move limit.
        return outcome.reason == 5 ? draw_repetition : outcome.reason == 6 ? draw_max_moves : 0;
    }
    double value() const {
        const auto winner = game.state().outcome.winner;
        return winner == -1 ? 0.0 : winner == player() ? 1.0 : -1.0;
    }
    bool step(int label) {
        for (const auto code : game.legal_actions()) {
            if (shogi::move_label(player(), code) == label) {
                game.play(code);
                return false;
            }
        }
        throw Error("No legal shogi move for label");
    }
    void undo() { game.undo(); }
    void observe(std::span<float> planes, std::span<float> mask) const {
        const auto encoded = game.encode(player());
        const auto legal = game.legal_mask(player());
        std::copy(encoded.begin(), encoded.end(), planes.begin());
        std::copy(legal.begin(), legal.end(), mask.begin());
    }
};

template<class T>
concept GameAdapter = requires(T adapter, const T immutable, int action,
                               std::span<float> planes, std::span<float> mask) {
    { T::action_size } -> std::convertible_to<int>;
    { T::planes_size } -> std::convertible_to<int>;
    { immutable.player() } -> std::same_as<int>;
    { immutable.terminal() } -> std::same_as<bool>;
    { immutable.value() } -> std::same_as<double>;
    { immutable.draw_reason() } -> std::same_as<std::uint32_t>;
    { adapter.step(action) } -> std::same_as<bool>;
    { adapter.undo() } -> std::same_as<void>;
    { immutable.observe(planes, mask) } -> std::same_as<void>;
};

struct Node {
    std::vector<int> actions;
    std::vector<double> prior, value_sum;
    std::vector<std::int64_t> visits;
    std::vector<std::size_t> children;
    std::int64_t total{};
    double value{};
    int player{-1};
    bool stopped{};
    bool symmetric{};
    static constexpr auto absent = std::numeric_limits<std::size_t>::max();

    std::vector<double> completed_q() const {
        std::vector<double> q(actions.size()), probabilities, weighted;
        for (std::size_t i = 0; i < actions.size(); ++i) {
            q[i] = value_sum[i] / static_cast<double>(std::max<std::int64_t>(1, visits[i]));
            if (visits[i]) {
                probabilities.push_back(prior[i]);
                weighted.push_back(prior[i] * q[i]);
            }
        }
        const auto mass = pairwise_sum(probabilities);
        const auto mixed = total && mass > 0
            ? (value + static_cast<double>(total) * (pairwise_sum(weighted) / mass)) / static_cast<double>(1 + total)
            : value;
        for (std::size_t i = 0; i < q.size(); ++i) if (!visits[i]) q[i] = mixed;
        return q;
    }
    std::vector<double> improved_logits(double c_visit, double c_scale) const {
        auto scores = completed_q();
        const auto maximum = *std::max_element(visits.begin(), visits.end());
        const auto scale = (c_visit + static_cast<double>(maximum)) * c_scale;
        for (std::size_t i = 0; i < scores.size(); ++i)
            scores[i] = (prior[i] > 0 ? std::log(prior[i]) : -std::numeric_limits<double>::infinity()) + scale * scores[i];
        return scores;
    }
    std::vector<double> improved_policy(double c_visit, double c_scale) const {
        auto policy = improved_logits(c_visit, c_scale);
        const auto peak = *std::max_element(policy.begin(), policy.end());
        for (auto& p : policy) p = std::exp(p - peak);
        const auto sum = pairwise_sum(policy);
        for (auto& p : policy) p /= sum;
        return policy;
    }
    std::size_t select_improved(double c_visit, double c_scale) const {
        auto scores = improved_policy(c_visit, c_scale);
        for (std::size_t i = 0; i < scores.size(); ++i)
            scores[i] -= static_cast<double>(visits[i]) / static_cast<double>(1 + total);
        return static_cast<std::size_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());
    }

    std::size_t select(double c_puct) const {
        const double scale = std::sqrt(static_cast<double>(std::max<std::int64_t>(1, total)));
        std::size_t best{};
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < actions.size(); ++i) {
            const double q = value_sum[i] / static_cast<double>(std::max<std::int64_t>(1, visits[i]));
            const double u = c_puct * prior[i] * scale / static_cast<double>(1 + visits[i]);
            const double score = q + u;
            // NumPy argmax returns the first NaN, including overflow-induced NaNs.
            if (std::isnan(score)) return i;
            if (score > maximum) { maximum = score; best = i; }
        }
        return best;
    }
    void expand(std::span<const float> mask, std::span<const double> probs, double evaluated_value) {
        if (!actions.empty()) throw Error("Leaf already expanded");
        std::vector<int> legal;
        std::vector<double> probabilities;
        for (std::size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] <= 0) continue;
            legal.push_back(static_cast<int>(i));
            probabilities.push_back(probs[i]);
        }
        if (legal.empty()) throw Error("A nonterminal MCTS leaf must have legal actions");
        const double sum = pairwise_sum(probabilities);
        if (std::any_of(probabilities.begin(), probabilities.end(), [](double p) {
                return !std::isfinite(p) || p < 0;
            }) || sum <= 0)
            throw Error("Evaluator must return a finite nonnegative legal policy");
        if (!std::isfinite(evaluated_value)) throw Error("Evaluator must return finite values");
        for (auto& p : probabilities) p /= sum;
        visits.assign(legal.size(), 0);
        value_sum.assign(legal.size(), 0.0);
        children.assign(legal.size(), absent);
        actions = std::move(legal);
        prior = std::move(probabilities);
        value = evaluated_value;
    }
};

// Same integer-round schedule as Python _SequentialHalving; never owns a node.
struct SequentialHalving {
    std::vector<std::size_t> candidates;
    std::vector<double> noise;
    std::int64_t budget{}, begin{}, end{};
    double c_visit{50}, c_scale{1};

    bool enabled() const { return !candidates.empty(); }
    std::int64_t phase_end(std::int64_t total) const {
        const auto count = static_cast<std::int64_t>(candidates.size());
        const auto remaining = budget - total;
        const auto phases = std::max(1, static_cast<int>(std::bit_width(static_cast<std::uint64_t>(count - 1))));
        const auto allocation = phases == 1 ? remaining : std::max(count, remaining / (phases * count) * count);
        return total + std::min(remaining, allocation);
    }
    static void rank(std::vector<std::size_t>& indices, const std::vector<double>& scores) {
        std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
            return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
        });
    }
    void initialize(const Node& root, std::span<const double> samples, std::int64_t simulations,
                    std::int64_t m, double visit, double scale) {
        if (enabled() || root.actions.empty() || root.total || samples.size() != root.actions.size()
            || simulations < 0 || m < 1 || !std::isfinite(visit) || visit < 0
            || !std::isfinite(scale) || scale < 0
            || std::any_of(samples.begin(), samples.end(), [](double x) { return !std::isfinite(x); }))
            throw Error("Invalid Gumbel configuration or root phase");
        budget = simulations; c_visit = visit; c_scale = scale;
        noise.assign(samples.begin(), samples.end());
        std::vector<double> scores(root.actions.size());
        candidates.resize(scores.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        for (std::size_t i = 0; i < scores.size(); ++i)
            scores[i] = (root.prior[i] > 0 ? std::log(root.prior[i]) : -std::numeric_limits<double>::infinity()) + noise[i];
        rank(candidates, scores);
        candidates.resize(static_cast<std::size_t>(std::min({m, static_cast<std::int64_t>(scores.size()),
                                                            std::max<std::int64_t>(1, budget)})));
        std::sort(candidates.begin(), candidates.end());
        end = phase_end(0);
    }
    std::vector<std::size_t> ranked(const Node& root) const {
        auto scores = root.improved_logits(c_visit, c_scale);
        for (std::size_t i = 0; i < scores.size(); ++i) scores[i] += noise[i];
        auto order = candidates;
        rank(order, scores);
        return order;
    }
    std::size_t select(const Node& root) {
        if (root.total >= budget) throw Error("Sequential halving simulation budget exhausted");
        if (root.total >= end && candidates.size() > 1) {
            candidates = ranked(root);
            candidates.resize((candidates.size() + 1) / 2);
            std::sort(candidates.begin(), candidates.end());
            begin = root.total;
            end = phase_end(root.total);
        }
        return candidates[static_cast<std::size_t>(root.total - begin) % candidates.size()];
    }
    int winner(const Node& root) const { return root.actions[ranked(root).front()]; }
};

template<GameAdapter Adapter>
class Tree {
public:
    explicit Tree(Adapter adapter, DrawPolicy draw = {}) : work(std::move(adapter)), draw_policy(draw) {
        nodes.emplace_back();
        nodes.front().player = work.player();
        if (work.terminal()) set_terminal(nodes.front());
    }
    Adapter work;
    DrawPolicy draw_policy;

    void set_terminal(Node& node) const {
        node.symmetric = (work.draw_reason() & draw_policy.reasons) != 0;
        node.value = node.symmetric ? draw_policy.value : work.value();
    }
    // A flat arena avoids recursive destruction of arbitrarily deep search paths.
    std::deque<Node> nodes;
    std::vector<std::pair<std::size_t, std::size_t>> path;
    std::array<float, Adapter::planes_size> planes{};
    std::array<float, Adapter::action_size> mask{};
    std::size_t leaf{};
    bool awaiting{}, pending{};
    SequentialHalving halving;

    bool prepare(bool descend, double c_puct, double truncation_value) {
        if (awaiting || pending) throw Error("Tree already has an outstanding leaf");
        leaf = 0;
        path.clear();
        std::size_t played = 0;
        // Always restore the complete engine state, including on allocation/encoding errors.
        struct Restore {
            Adapter& adapter;
            std::size_t& count;
            ~Restore() { while (count) { adapter.undo(); --count; } }
        } restore{work, played};
        if (descend) {
            while (!nodes[leaf].actions.empty() && !work.terminal()) {
                auto& parent = nodes[leaf];
                const auto edge = halving.enabled()
                    ? (leaf == 0 ? halving.select(parent) : parent.select_improved(halving.c_visit, halving.c_scale))
                    : parent.select(c_puct);
                const bool truncated = work.step(parent.actions[edge]);
                ++played;
                path.emplace_back(leaf, edge);
                if (parent.children[edge] == Node::absent) {
                    nodes.emplace_back();
                    parent.children[edge] = nodes.size() - 1;
                }
                leaf = parent.children[edge];
                if (truncated && !work.terminal()) {
                    auto& node = nodes[leaf];
                    node.stopped = true;
                    node.player = parent.player;
                    node.value = truncation_value;
                    pending = true;
                    return false;
                }
            }
        }
        auto& node = nodes[leaf];
        node.player = work.player();
        if (work.terminal()) {
            set_terminal(node);
            pending = descend;
            return false;
        }
        if (!node.actions.empty()) throw Error("Root already initialized");
        work.observe(planes, mask);
        awaiting = true;
        pending = descend;
        return true;
    }
    void expand(std::span<const double> probs, double value) {
        if (!awaiting) throw Error("Tree has no leaf awaiting evaluation");
        nodes[leaf].expand(mask, probs, value);
        awaiting = false;
    }
    void backup() {
        if (!pending || awaiting) throw Error("Tree has no completed pending simulation");
        const auto& node = nodes[leaf];
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            auto& parent = nodes[it->first];
            const auto edge = it->second;
            const double value = node.stopped || node.symmetric || parent.player == node.player ? node.value : -node.value;
            ++parent.visits[edge];
            parent.value_sum[edge] += value;
            ++parent.total;
        }
        pending = false;
    }
    void result(std::span<float> visits, std::span<float> q, double& value, std::int64_t& simulations) const {
        const auto& root = nodes.front();
        std::fill(visits.begin(), visits.end(), 0.0F);
        std::fill(q.begin(), q.end(), std::numeric_limits<float>::quiet_NaN());
        for (std::size_t i = 0; i < root.actions.size(); ++i) {
            visits[root.actions[i]] = static_cast<float>(root.total ? static_cast<double>(root.visits[i]) : root.prior[i]);
            if (root.visits[i]) q[root.actions[i]] = static_cast<float>(root.value_sum[i] / static_cast<double>(root.visits[i]));
        }
        value = root.total ? pairwise_sum(root.value_sum) / static_cast<double>(root.total) : root.value;
        simulations = root.total;
    }
};
} // namespace mcts
#endif
