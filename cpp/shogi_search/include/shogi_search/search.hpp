// SPDX-License-Identifier: MIT
#ifndef SHOGI_SEARCH_SEARCH_HPP
#define SHOGI_SEARCH_SEARCH_HPP
#include "shogi_search/position.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>

namespace shogi_search {
enum Feature : unsigned {
    pvs = 1U, aspiration = 2U, transposition = 4U, quiescence = 8U,
    null_move = 16U, lmr = 32U, futility = 64U, razoring = 128U,
    check_extension = 256U, see_pruning = 512U, ordering = 1024U,
    all_features = 2047U
};
struct Options {
    unsigned features{all_features};
    int hash_mb{16};
    int qdepth{12};
    void validate() const;
};
struct Limits {
    int depth{6};
    std::uint64_t nodes{}; // Zero disables this limit.
    int time_ms{};         // Hard limit; zero disables.
    int soft_time_ms{};    // Stop between completed iterations; zero disables.
    int multipv{1};        // Zero means every legal root move.
    std::vector<int> root_moves;
    void validate() const;
};
struct RootScore {
    int move{-1}, score{}, depth{};
    std::vector<int> pv;
};
struct Result {
    int move{-1}, score{}, depth{}, mate_plies{};
    std::uint64_t nodes{};
    double elapsed_ms{};
    bool completed{}, stopped{};
    std::vector<int> pv;
    std::vector<RootScore> roots;
};
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void reset(const Position&) = 0;
    virtual void push(const Position& after, const Undo&) = 0;
    virtual void pop() noexcept = 0;
    virtual int evaluate(const Position&) const = 0;
};
class BootstrapEvaluator final : public Evaluator {
public:
    void reset(const Position&) override;
    void push(const Position&, const Undo&) override;
    void pop() noexcept override;
    int evaluate(const Position&) const override;
private:
    int material_{};
    std::array<int, max_ply + 2> stack_{};
    int count_{};
};
class Searcher {
public:
    explicit Searcher(Options options = {}, std::unique_ptr<Evaluator> evaluator = {});
    // Called synchronously after each completed iteration, with the root restored.
    // The observer may call stop(), but must not reenter any other search method.
    using Progress = std::function<void(const Result&)>;
    Result go(const Root&, const Limits&, const Progress& progress = {});
    void set_evaluator(std::unique_ptr<Evaluator> evaluator);
    // stop() alone may be called concurrently, including just before go().
    void stop() noexcept { stopped_.store(true, std::memory_order_relaxed); }
    void clear_stop() noexcept { stopped_.store(false, std::memory_order_relaxed); }
private:
    struct Entry {
        std::uint64_t key{};
        int score{}, move{-1};
        std::int16_t depth{-1};
        std::uint8_t bound{}, age{};
    };
    struct Value { int score; bool path_dependent; };
    Options options_;
    Limits limits_;
    std::unique_ptr<Evaluator> evaluator_;
    std::atomic<bool> stopped_{false};
    std::vector<Entry> table_;
    std::uint8_t age_{};
    std::array<std::array<int, 2>, max_ply> killers_{};
    std::array<std::array<int, 13690>, 2> history_{};
    std::array<std::array<int, max_ply>, max_ply> pv_{};
    std::array<int, max_ply> pv_size_{};
    std::vector<Event> events_;
    shogi::Rules rules_;
    std::uint64_t nodes_{};
    std::uint64_t repetitions_{};
    bool aborted_{};
    std::chrono::steady_clock::time_point start_;
    bool enabled(Feature f) const noexcept { return (options_.features & f) != 0; }
    double elapsed() const;
    bool budget();
    void clear_ordering_state();
    Value visit(Position&, int depth, int alpha, int beta, int ply, int qleft,
                int extensions, bool synthetic, bool allow_null);
    bool prove_mate(Position&, int remaining, int attacker);
    void order(Position&, MoveList&, int hint, int ply);
    void update_pv(int ply, int move);
    int terminal(const shogi::Outcome&, int side, int ply) const;
};
}
#endif
