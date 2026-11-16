#include "mcts/c_api.h"
#include "mcts/search.hpp"
#include "mcts/pool.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string_view>
#include <variant>

namespace {
thread_local std::array<char, 1024> last_error{};
template<class F, class R> R guarded(F&& function, R failure) noexcept {
    last_error[0] = '\0';
    try { return function(); }
    catch (const std::exception& error) { std::snprintf(last_error.data(), last_error.size(), "%s", error.what()); }
    catch (...) { std::snprintf(last_error.data(), last_error.size(), "%s", "Unknown native MCTS exception"); }
    return failure;
}
template<class T> std::span<T> buffer(T* pointer, std::size_t capacity, std::size_t required) {
    if (capacity < required || (required && !pointer)) throw mcts::Error("Missing or undersized buffer");
    return {pointer, required};
}
std::size_t product(std::size_t rows, int width) {
    if (rows > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(width))
        throw mcts::Error("Buffer size overflow");
    return rows * static_cast<std::size_t>(width);
}
template<mcts::GameAdapter Adapter> using Trees = std::vector<mcts::Tree<Adapter>>;
struct Search {
    using Storage = std::variant<Trees<mcts::HasamiAdapter>, Trees<mcts::ShogiAdapter>>;
    Storage trees;
    mcts::Pool pool;
    double c_puct, truncation_value;
    mcts::DrawPolicy draw_policy;
    bool sealed{};
    Search(int kind, int threads, double c, double value)
        : trees(kind == 0 ? Storage{Trees<mcts::HasamiAdapter>{}} : Storage{Trees<mcts::ShogiAdapter>{}}),
          pool(threads), c_puct(c), truncation_value(value) {}
    template<class F> auto visit(F&& function) { return std::visit(std::forward<F>(function), trees); }
};
Search& search(void* handle) {
    if (!handle) throw mcts::Error("Null MCTS handle");
    return *static_cast<Search*>(handle);
}
template<class TreesType>
void validate_ids(const TreesType& trees, std::span<const int32_t> ids) {
    std::vector<bool> seen(trees.size(), false);
    for (const auto id : ids) {
        if (id < 0 || static_cast<std::size_t>(id) >= trees.size()) throw mcts::Error("Invalid tree index");
        if (seen[id]) throw mcts::Error("Duplicate tree index");
        seen[id] = true;
    }
}
template<class Adapter> int append(Search& s, Adapter adapter) {
    if (s.sealed) throw mcts::Error("Cannot add trees after search initialization");
    auto* trees = std::get_if<Trees<Adapter>>(&s.trees);
    if (!trees) throw mcts::Error("Wrong game kind for search handle");
    if (trees->size() >= static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw mcts::Error("Too many trees");
    trees->emplace_back(std::move(adapter), s.draw_policy);
    return static_cast<int>(trees->size() - 1);
}
void flag(int value) {
    if (value != 0 && value != 1) throw mcts::Error("Boolean setting must be 0 or 1");
}
} // namespace

extern "C" {
int32_t mz_abi_version(void) { return MZ_ABI_VERSION; }
int32_t mz_last_error(char* out, size_t capacity) {
    const auto length = std::strlen(last_error.data());
    if (!out && capacity == 0) return static_cast<int32_t>(length);
    if (!out || capacity <= length) return -1;
    std::memcpy(out, last_error.data(), length + 1);
    return static_cast<int32_t>(length);
}
void* mz_create(int32_t kind, int32_t threads, double c_puct, double truncation_value) {
    return guarded([&]() -> void* {
        if ((kind != 0 && kind != 1) || threads < 1 || !std::isfinite(c_puct) || c_puct < 0
            || !std::isfinite(truncation_value) || std::abs(truncation_value) > 1)
            throw mcts::Error("Invalid native MCTS configuration");
        return std::make_unique<Search>(kind, threads, c_puct, truncation_value).release();
    }, static_cast<void*>(nullptr));
}
void mz_destroy(void* handle) {
    guarded([&]() { const std::unique_ptr<Search> owner(static_cast<Search*>(handle)); return 0; }, -1);
}
int32_t mz_draw_policy(void* handle, double value, uint32_t reasons) {
    return guarded([&]() {
        auto& s = search(handle);
        if (!std::isfinite(value) || std::abs(value) > 1
            || (reasons & ~(mcts::draw_repetition | mcts::draw_max_moves)))
            throw mcts::Error("Invalid native MCTS draw policy");
        if (s.sealed || s.visit([](const auto& trees) { return !trees.empty(); }))
            throw mcts::Error("Set draw policy before adding trees");
        s.draw_policy = {value, reasons};
        return 0;
    }, -1);
}
int32_t mz_add_hasami(void* handle, const uint64_t* boards, size_t boards_capacity,
                     const int32_t* settings, size_t settings_capacity,
                     const int32_t* moves, size_t move_count) {
    return guarded([&]() {
        auto& s = search(handle);
        const auto b = buffer(boards, boards_capacity, 4);
        const auto r = buffer(settings, settings_capacity, 7);
        const auto history = buffer(moves, move_count, move_count);
        flag(r[2]); flag(r[6]);
        hasami::Game game({r[2] != 0, r[3], r[4], r[5], r[6] != 0},
                          {{b[0], b[1]}, {b[2], b[3]}, r[0], r[1]});
        for (const auto action : history) game.play(action);
        return append(s, mcts::HasamiAdapter{std::move(game)});
    }, -1);
}
int32_t mz_add_shogi(void* handle, const char* sfen, size_t sfen_length,
                    const int32_t* settings, size_t settings_capacity,
                    const int32_t* moves, size_t move_count, int32_t resigned) {
    return guarded([&]() {
        auto& s = search(handle);
        buffer(sfen, sfen_length, sfen_length);
        if (!sfen_length) throw mcts::Error("Missing initial SFEN");
        const auto r = buffer(settings, settings_capacity, 2);
        const auto history = buffer(moves, move_count, move_count);
        flag(resigned);
        shogi::Game game({r[0], r[1]}, std::string_view(sfen, sfen_length));
        for (const auto code : history) game.play(code);
        if (resigned) game.resign();
        return append(s, mcts::ShogiAdapter{std::move(game)});
    }, -1);
}
int32_t mz_prepare(void* handle, const int32_t* ids, size_t count, int32_t descend,
                   float* planes, size_t planes_capacity, float* masks, size_t masks_capacity,
                   int32_t* leaf_ids, size_t ids_capacity) {
    return guarded([&]() {
        auto& s = search(handle);
        const auto indices = buffer(ids, count, count);
        const auto output_ids = buffer(leaf_ids, ids_capacity, count);
        flag(descend);
        return s.visit([&](auto& trees) {
            using Adapter = decltype(trees.front().work);
            validate_ids(trees, indices);
            buffer(planes, planes_capacity, product(count, Adapter::planes_size));
            buffer(masks, masks_capacity, product(count, Adapter::action_size));
            std::vector<unsigned char> needs(count);
            s.sealed = true;
            s.pool.run(count, [&](std::size_t i) {
                needs[i] = trees[indices[i]].prepare(descend != 0, s.c_puct, s.truncation_value);
            });
            std::size_t rows = 0;
            for (std::size_t i = 0; i < count; ++i) {
                if (!needs[i]) continue;
                const auto& tree = trees[indices[i]];
                std::copy(tree.planes.begin(), tree.planes.end(), planes + rows * Adapter::planes_size);
                std::copy(tree.mask.begin(), tree.mask.end(), masks + rows * Adapter::action_size);
                output_ids[rows++] = indices[i];
            }
            return static_cast<int32_t>(rows);
        });
    }, -1);
}
int32_t mz_expand(void* handle, const int32_t* ids, size_t count,
                  const double* probabilities, size_t probabilities_capacity,
                  const double* values, size_t values_capacity) {
    return guarded([&]() {
        auto& s = search(handle);
        const auto indices = buffer(ids, count, count);
        const auto evaluated_values = buffer(values, values_capacity, count);
        return s.visit([&](auto& trees) {
            using Adapter = decltype(trees.front().work);
            validate_ids(trees, indices);
            const auto probs = buffer(probabilities, probabilities_capacity, product(count, Adapter::action_size));
            // Preserve evaluator-error precedence in batch order.
            for (std::size_t i = 0; i < count; ++i)
                trees[indices[i]].expand(probs.subspan(i * Adapter::action_size, Adapter::action_size), evaluated_values[i]);
            return 0;
        });
    }, -1);
}
int32_t mz_noise(void* handle, int32_t index, const double* noise, size_t count, double epsilon) {
    return guarded([&]() {
        auto& s = search(handle);
        const auto values = buffer(noise, count, count);
        if (!std::isfinite(epsilon) || epsilon < 0 || epsilon > 1) throw mcts::Error("Invalid noise fraction");
        return s.visit([&](auto& trees) {
            validate_ids(trees, std::span<const int32_t>(&index, 1));
            auto& root = trees[index].nodes.front();
            if (root.actions.empty() || root.prior.size() != count || root.total)
                throw mcts::Error("Noise must match the initialized root's legal actions");
            for (const auto value : values)
                if (!std::isfinite(value) || value < 0) throw mcts::Error("Invalid noise");
            for (std::size_t i = 0; i < count; ++i)
                root.prior[i] = (1 - epsilon) * root.prior[i] + epsilon * values[i];
            return 0;
        });
    }, -1);
}
int32_t mz_gumbel(void* handle, int32_t index, const double* noise, size_t count,
                  int64_t simulations, int64_t m, double c_visit, double c_scale) {
    return guarded([&]() {
        auto& s = search(handle);
        const auto samples = buffer(noise, count, count);
        return s.visit([&](auto& trees) {
            validate_ids(trees, std::span<const int32_t>(&index, 1));
            auto& tree = trees[index];
            if (tree.awaiting || tree.pending) throw mcts::Error("Gumbel requires a completed root evaluation");
            tree.halving.initialize(tree.nodes.front(), samples, simulations, m, c_visit, c_scale);
            return 0;
        });
    }, -1);
}
int32_t mz_policy(void* handle, double* policy, size_t capacity, int32_t* selected, size_t selected_capacity) {
    return guarded([&]() {
        auto& s = search(handle);
        return s.visit([&](auto& trees) {
            using Adapter = decltype(trees.front().work);
            const auto output = buffer(policy, capacity, product(trees.size(), Adapter::action_size));
            const auto actions = buffer(selected, selected_capacity, trees.size());
            std::fill(output.begin(), output.end(), 0.0);
            std::fill(actions.begin(), actions.end(), -1);
            for (std::size_t i = 0; i < trees.size(); ++i) {
                const auto& tree = trees[i];
                if (!tree.halving.enabled()) continue;
                const auto& root = tree.nodes.front();
                const auto probabilities = root.improved_policy(tree.halving.c_visit, tree.halving.c_scale);
                for (std::size_t j = 0; j < probabilities.size(); ++j)
                    output[i * Adapter::action_size + root.actions[j]] = probabilities[j];
                actions[i] = tree.halving.winner(root);
            }
            return 0;
        });
    }, -1);
}
int32_t mz_backup(void* handle, const int32_t* ids, size_t count) {
    return guarded([&]() {
        auto& s = search(handle);
        const auto indices = buffer(ids, count, count);
        return s.visit([&](auto& trees) {
            validate_ids(trees, indices);
            s.pool.run(count, [&](std::size_t i) { trees[indices[i]].backup(); });
            return 0;
        });
    }, -1);
}
int32_t mz_observe(void* handle, float* planes, size_t planes_capacity, float* masks, size_t masks_capacity) {
    return guarded([&]() {
        auto& s = search(handle);
        return s.visit([&](auto& trees) {
            using Adapter = decltype(trees.front().work);
            const auto p = buffer(planes, planes_capacity, product(trees.size(), Adapter::planes_size));
            const auto m = buffer(masks, masks_capacity, product(trees.size(), Adapter::action_size));
            s.sealed = true;
            s.pool.run(trees.size(), [&](std::size_t i) {
                trees[i].work.observe(p.subspan(i * Adapter::planes_size, Adapter::planes_size),
                                      m.subspan(i * Adapter::action_size, Adapter::action_size));
            });
            return 0;
        });
    }, -1);
}
int32_t mz_results(void* handle, float* visits, size_t visits_capacity, float* q_values, size_t q_capacity,
                   double* root_values, size_t values_capacity, int64_t* simulations, size_t simulations_capacity) {
    return guarded([&]() {
        auto& s = search(handle);
        return s.visit([&](auto& trees) {
            using Adapter = decltype(trees.front().work);
            const auto n = trees.size();
            const auto v = buffer(visits, visits_capacity, product(n, Adapter::action_size));
            const auto q = buffer(q_values, q_capacity, product(n, Adapter::action_size));
            const auto values = buffer(root_values, values_capacity, n);
            const auto sims = buffer(simulations, simulations_capacity, n);
            for (std::size_t i = 0; i < n; ++i)
                trees[i].result(v.subspan(i * Adapter::action_size, Adapter::action_size),
                                q.subspan(i * Adapter::action_size, Adapter::action_size), values[i], sims[i]);
            return 0;
        });
    }, -1);
}
} // extern C
