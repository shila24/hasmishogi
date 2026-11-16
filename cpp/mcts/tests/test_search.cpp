#include "mcts/search.hpp"
#include "mcts/c_api.h"
#include "mcts/pool.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw mcts::Error(message);
}
template<class Adapter> void expand(mcts::Tree<Adapter>& tree, double value = 0.25) {
    std::vector<double> prior(Adapter::action_size, 1.0);
    tree.expand(prior, value);
}
struct CappedAdapter {
    static constexpr int action_size = 1, planes_size = 1;
    int ply{}, cap{2};
    bool extra_turn{};
    int player() const { return extra_turn ? 0 : ply % 2; }
    bool terminal() const { return false; }
    double value() const { return 0; }
    std::uint32_t draw_reason() const { return 0; }
    bool step(int) {
        check(ply < cap, "Stepped past truncation");
        return ++ply == cap;
    }
    void undo() { --ply; }
    void observe(std::span<float> planes, std::span<float> mask) const {
        check(ply < cap, "Evaluated a truncated state");
        planes[0] = static_cast<float>(ply); mask[0] = 1;
    }
};
struct DrawnAdapter : CappedAdapter {
    bool terminal() const { return ply == cap; }
    std::uint32_t draw_reason() const { return terminal() ? mcts::draw_repetition : 0; }
    bool step(int) { ++ply; return false; }
};
void tree_tests() {
    for (int depth : {1, 2, 3}) {
        for (bool extra_turn : {false, true}) {
            for (bool gumbel : {false, true}) {
                mcts::Tree<DrawnAdapter> draw{DrawnAdapter{{0, depth, extra_turn}}, {-0.2, mcts::draw_repetition}};
                draw.prepare(false, 1.5, 0.4);
                expand(draw, 0);
                if (gumbel) draw.halving.initialize(draw.nodes[0], std::vector<double>{0}, 8, 1, 50, 1);
                for (int simulation = 0; simulation < 8; ++simulation) {
                    if (draw.prepare(true, 1.5, 0.4)) expand(draw, 0);
                    draw.backup();
                }
                check(std::abs(draw.nodes[0].value_sum[0] + (9 - depth) * 0.2) < 1e-12,
                      "Draw utility is symmetric across odd/even depths and extra turns");
                check(draw.work.ply == 0, "Draw search restores history");
            }
        }
    }
    mcts::Node completed;
    completed.actions = {0, 1, 2}; completed.prior = {.2, .3, .5};
    completed.visits = {2, 1, 0}; completed.value_sum = {1, -.5, 0};
    completed.total = 3; completed.value = .4;
    const auto cq = completed.completed_q();
    check(cq[0] == .5 && cq[1] == -.5 && std::abs(cq[2] - .025) < 1e-15, "Completed Q value mix");
    const auto policy = completed.improved_policy(0, 1);
    const auto normalizer = .2 * std::exp(1.) + .3 * std::exp(-1.) + .5 * std::exp(.05);
    check(std::abs(policy[2] - .5 * std::exp(.05) / normalizer) < 1e-15, "Improved policy");
    check(completed.select_improved(0, 1) == 2, "Interior visit matching");
    for (const int count : {1, 3, 5, 16}) {
        for (const int budget : {0, 1, 2, 7, 64}) {
            mcts::Node root;
            root.expand(std::vector<float>(count, 1), std::vector<double>(count, 1), .25);
            mcts::SequentialHalving halving;
            halving.initialize(root, std::vector<double>(count, 0), budget, 16, 50, 0);
            for (int i = 0; i < budget; ++i) {
                const auto edge = halving.select(root);
                check(edge < static_cast<std::size_t>(count), "Legal halving selection");
                ++root.visits[edge]; ++root.total;
            }
            check(root.total == budget && halving.winner(root) == 0, "Exact halving budget and ties");
        }
    }
    mcts::Node overflow;
    overflow.actions = {0, 1}; overflow.prior = {1, 0};
    overflow.visits = {1, 0}; overflow.value_sum = {-std::numeric_limits<double>::infinity(), 0};
    overflow.total = 4;
    check(overflow.select(std::numeric_limits<double>::max()) == 0, "NumPy first-NaN argmax");
    mcts::Tree<mcts::HasamiAdapter> tree{mcts::HasamiAdapter{hasami::Game{}}};
    const auto history = tree.work.game.history();
    check(tree.prepare(false, 1.5, 0), "Initial evaluation");
    expand(tree);
    check(tree.nodes[0].actions.size() == 63, "63 legal actions");
    std::vector<float> visits(6561), q(6561);
    double value{}; std::int64_t simulations{};
    tree.result(visits, q, value, simulations);
    check(simulations == 0 && value == 0.25, "Zero simulations preserves root value");
    check(visits[tree.nodes[0].actions[0]] == static_cast<float>(1.0 / 63), "Prior fallback");
    check(std::all_of(q.begin(), q.end(), [](float x) { return std::isnan(x); }), "Unvisited Q is NaN");
    for (int i = 0; i < 100; ++i) {
        if (tree.prepare(true, 1.5, 0)) expand(tree);
        tree.backup();
        check(tree.nodes[0].total == i + 1, "Exactly one simulation");
        check(tree.work.game.history() == history, "Complete history restored");
    }
    hasami::Game repeated;
    const std::array<int, 4> cycle{72*81+63, 9, 63*81+72, 9*81};
    for (int i = 0; i < 12; ++i) repeated.play(cycle[i % 4]);
    check(repeated.outcome().winner == -1, "Fourfold draw");
    mcts::Tree<mcts::HasamiAdapter> terminal{mcts::HasamiAdapter{repeated}};
    check(!terminal.prepare(false, 1.5, 0), "No evaluation for terminal root");
    terminal.result(visits, q, value, simulations);
    check(simulations == 0 && value == 0, "Terminal root zero budget");
    check(std::all_of(visits.begin(), visits.end(), [](float x) { return x == 0; }), "Terminal visits zero");
    repeated.undo();
    mcts::Tree<mcts::HasamiAdapter> near{mcts::HasamiAdapter{repeated}};
    near.prepare(false, 1.5, 0);
    std::vector<double> forced(6561, 0);
    forced[cycle.back()] = 1;
    near.expand(forced, 0.75);
    check(!near.prepare(true, 1.5, 0), "History-dependent leaf draw");
    near.backup();
    check(near.nodes[0].value_sum[near.nodes[0].select(1.5)] == 0, "Draw backed up");
    hasami::Position p;
    for (int s : {38,67,72,73,74,75,76,77,78}) p.black.set(s);
    for (int s : {39,37,3,0,1,2,4}) p.white.set(s);
    hasami::Game pending({}, p);
    pending.play(67*81+40);
    check(pending.position().pending == 0, "Forced reply pending");
    mcts::Tree<mcts::HasamiAdapter> reply{mcts::HasamiAdapter{pending}};
    reply.prepare(false, 1.5, 0);
    forced.assign(6561, 0); forced[9] = 1;
    reply.expand(forced, 0);
    check(!reply.prepare(true, 1.5, 0), "Unrescued lead terminal");
    reply.backup();
    check(mcts::pairwise_sum(reply.nodes[0].value_sum) == -1, "Reply player loses");
    check(reply.work.game.position().pending == 0, "Pending restored");
    shogi::Game capped({1, 0});
    mcts::Tree<mcts::ShogiAdapter> shogi_tree{mcts::ShogiAdapter{capped}};
    shogi_tree.prepare(false, 1.5, 0.4); expand(shogi_tree);
    check(!shogi_tree.prepare(true, 1.5, 0.4), "Shogi rule cap stops");
    shogi_tree.backup();
    check(mcts::pairwise_sum(shogi_tree.nodes[0].value_sum) == 0, "Rule cap remains a draw");
    for (const bool extra : {false, true}) {
      for (const bool gumbel : {false, true}) {
        mcts::Tree<CappedAdapter> stopped{CappedAdapter{0, 2, extra}};
        stopped.prepare(false, 1.5, 0.4); expand(stopped, 0);
        if (gumbel) stopped.halving.initialize(stopped.nodes.front(), std::array<double, 1>{0}, 8, 16, 50, 1);
        for (int i = 0; i < 8; ++i) {
            if (stopped.prepare(true, 1.5, 0.4)) expand(stopped, 0);
            stopped.backup();
        }
        check(stopped.nodes[0].total == 8 && stopped.work.ply == 0, "Capped paths undone");
        check(std::abs(stopped.nodes[0].value_sum[0] - 2.8) < 1e-14, "Truncation value same for both players");
      }
    }
    mcts::Tree<CappedAdapter> extra{CappedAdapter{0, 10, true}};
    extra.prepare(false, 1.5, 0); expand(extra);
    extra.prepare(true, 1.5, 0); expand(extra, 0.7); extra.backup();
    check(extra.nodes[0].value_sum[0] == 0.7, "Extra turn uses player identity");
}
void abi_tests() {
    using Handle = std::unique_ptr<void, decltype(&mz_destroy)>;
    check(mz_abi_version() == MZ_ABI_VERSION, "ABI version");
    check(mz_create(2, 1, 1.5, 0) == nullptr, "Invalid kind");
    for (double value : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity(), -1.1, 1.1}) {
        Handle handle(mz_create(0, 1, 1.5, 0), mz_destroy);
        check(mz_draw_policy(handle.get(), value, 1) == -1, "Invalid draw value rejected");
    }
    {
        Handle handle(mz_create(0, 1, 1.5, 0), mz_destroy);
        check(mz_draw_policy(handle.get(), -.2, 4) == -1, "Unknown draw reason bit rejected");
    }
    check(mz_draw_policy(nullptr, -.2, 1) == -1, "Null draw policy handle rejected");
    for (const int kind : {0, 1}) {
        Handle handle(mz_create(kind, 4, 1.5, 0), mz_destroy);
        check(bool(handle), "Create search");
        check(mz_draw_policy(handle.get(), -.2, 3) == 0, "Set symmetric draw policy");
        const auto p = hasami::Position::initial();
        const std::array<uint64_t, 4> boards{p.black.lo,p.black.hi,p.white.lo,p.white.hi};
        const std::array<int32_t, 7> settings{0,-1,1,5,3,4,0};
        const std::array<int32_t, 2> shogi_settings{320, 1};
        for (int i = 0; i < 16; ++i) {
            const auto index = kind == 0
                ? mz_add_hasami(handle.get(), boards.data(), 4, settings.data(), 7, nullptr, 0)
                : mz_add_shogi(handle.get(), shogi::start_sfen.data(), shogi::start_sfen.size(),
                               shogi_settings.data(), 2, nullptr, 0, 0);
            check(index == i, "Import game");
        }
        std::vector<int32_t> ids(16), leaves(16);
        for (int i = 0; i < 16; ++i) ids[i] = i;
        std::vector<float> planes(16 * (kind == 0 ? 729 : 3969)), masks(16 * (kind == 0 ? 6561 : 2188));
        check(mz_prepare(handle.get(), ids.data(), 16, 0, planes.data(), planes.size(), masks.data(), masks.size(),
                         leaves.data(), leaves.size()) == 16, "Threaded root preparation");
        std::vector<double> probs(masks.begin(), masks.end()), values(16, 0.25);
        check(mz_expand(handle.get(), leaves.data(), 16, probs.data(), probs.size(), values.data(), values.size()) == 0,
              "Batch expansion");
        for (int i = 0; i < 32; ++i) {
            const auto count = mz_prepare(handle.get(), ids.data(), 16, 1, planes.data(), planes.size(), masks.data(),
                                          masks.size(), leaves.data(), leaves.size());
            check(count == 16, "Threaded descent");
            probs.assign(masks.begin(), masks.end());
            check(mz_expand(handle.get(), leaves.data(), count, probs.data(), probs.size(), values.data(), values.size()) == 0,
                  "Leaf expansion");
            check(mz_backup(handle.get(), ids.data(), 16) == 0, "Threaded backup");
        }
        check(mz_observe(handle.get(), planes.data(), 1, masks.data(), masks.size()) == -1, "Small buffer rejected");
        std::array<char, 1024> error{};
        check(mz_last_error(error.data(), error.size()) > 0, "Error text");
        check(mz_last_error(error.data(), 1) == -1, "Small error buffer rejected");
    }
    check(mz_backup(nullptr, nullptr, 0) == -1, "Null handle rejected");
    {
        Handle handle(mz_create(1, 1, 1.5, 0), mz_destroy);
        const std::array<int32_t, 2> settings{320, 1};
        check(mz_add_shogi(handle.get(), shogi::start_sfen.data(), shogi::start_sfen.size(),
                           settings.data(), 2, nullptr, 0, 0) == 0, "Add before invalid policy update");
        check(mz_draw_policy(handle.get(), -.2, 3) == -1, "Cannot change policy after tree import");
    }
    mcts::Pool pool(4);
    std::vector<int> output(100);
    pool.run(output.size(), [&](std::size_t i) { output[i] = static_cast<int>(i); });
    for (int i = 0; i < 100; ++i) check(output[i] == i, "Independent tasks");
    bool threw = false;
    try { pool.run(100, [](std::size_t i) { if (i == 7) throw mcts::Error("worker error"); }); }
    catch (const mcts::Error&) { threw = true; }
    check(threw, "Worker exception propagated after joining phase");
    pool.run(100, [&](std::size_t i) { output[i] += 1; });
}
} // namespace
int main() {
    try { tree_tests(); abi_tests(); std::cout << "Native MCTS tests passed\n"; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
