#include "shogi/c_api.h"
#include "shogi/game.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>

struct sg_game { shogi::Game value; };
namespace {
// Error reporting must not allocate while handling an allocation failure.
thread_local std::array<char, 1024> last_error{};
void error(const char* message) noexcept { std::snprintf(last_error.data(), last_error.size(), "%s", message); }
template<class F, class R> R guarded(F&& function, R failure) noexcept {
    last_error[0] = '\0';
    try { return function(); }
    catch (const std::exception& exc) { error(exc.what()); }
    catch (...) { error("Unknown native exception"); }
    return failure;
}
const shogi::Game& game(const sg_game* handle) {
    if (!handle) throw shogi::Error("Null game handle");
    return handle->value;
}
shogi::Game& game(sg_game* handle) {
    if (!handle) throw shogi::Error("Null game handle");
    return handle->value;
}
template<class T> void require_output(T* out) {
    if (!out) throw shogi::Error("Null output pointer");
}
template<class T, class Range> int32_t elements(const Range& values, T* out, int32_t capacity) {
    const auto size = static_cast<int32_t>(values.size());
    if (!out && capacity == 0) return size;
    if (capacity < size || capacity < 0) throw shogi::Error("Output buffer too small");
    require_output(out);
    std::copy(values.begin(), values.end(), out);
    return size;
}
int32_t string(const std::string& value, char* out, int32_t capacity) {
    const auto size = static_cast<int32_t>(value.size());
    if (!out && capacity == 0) return size;
    if (capacity <= size) throw shogi::Error("Output buffer too small");
    require_output(out);
    std::memcpy(out, value.c_str(), value.size() + 1);
    return size;
}
} // namespace
extern "C" {
int32_t sg_abi_version(void) { return SG_ABI_VERSION; }
size_t sg_sizeof_rules(void) { return sizeof(sg_rules); }
size_t sg_sizeof_state(void) { return sizeof(sg_state); }
const char* sg_last_error(void) { return last_error.data(); }
int32_t sg_default_rules(sg_rules* out) {
    return guarded([&]() -> int32_t { require_output(out); *out = {320, 1}; return 0; }, -1);
}
sg_game* sg_create(const sg_rules* rules, const char* sfen) {
    return guarded([&]() {
        const shogi::Rules r = rules ? shogi::Rules{rules->max_moves, rules->declaration} : shogi::Rules{};
        auto handle = std::make_unique<sg_game>(sg_game{shogi::Game(r, sfen ? sfen : shogi::start_sfen)});
        return handle.release();
    }, static_cast<sg_game*>(nullptr));
}
sg_game* sg_clone(const sg_game* handle) {
    return guarded([&]() { auto copy = std::make_unique<sg_game>(sg_game{game(handle)}); return copy.release(); },
                   static_cast<sg_game*>(nullptr));
}
void sg_destroy(sg_game* handle) {
    guarded([&]() { const std::unique_ptr<sg_game> owner(handle); return 0; }, -1);
}
int32_t sg_get_state(const sg_game* handle, sg_state* out) {
    return guarded([&]() -> int32_t {
        const auto& g = game(handle);
        require_output(out);
        const auto& n = g.state();
        *out = {n.position.turn, n.ply, n.position.move_number, n.outcome.winner, n.outcome.reason,
                n.check ? 1 : 0, g.can_declare() ? 1 : 0};
        return 0;
    }, -1);
}
int32_t sg_board(const sg_game* handle, int32_t* out, int32_t capacity) {
    return guarded([&]() { return elements(game(handle).state().position.board, out, capacity); }, -1);
}
int32_t sg_hands(const sg_game* handle, int32_t* out, int32_t capacity) {
    return guarded([&]() {
        const auto& hands = game(handle).state().position.hands;
        std::array<int32_t, 14> flat{};
        for (int side = 0; side < 2; ++side) std::copy(hands[side].begin(), hands[side].end(), flat.begin() + 7 * side);
        return elements(flat, out, capacity);
    }, -1);
}
int32_t sg_moves(const sg_game* handle, int32_t* out, int32_t capacity) {
    return guarded([&]() { return elements(game(handle).moves(), out, capacity); }, -1);
}
int32_t sg_sfen(const sg_game* handle, char* out, int32_t capacity) {
    return guarded([&]() { return string(game(handle).state().position.sfen(), out, capacity); }, -1);
}
int32_t sg_history_key(const sg_game* handle, int32_t index, char* out, int32_t capacity) {
    return guarded([&]() { return string(game(handle).history_key(index), out, capacity); }, -1);
}
int32_t sg_legal_moves(const sg_game* handle, int32_t* out, int32_t capacity) {
    return guarded([&]() { return elements(game(handle).legal_actions(), out, capacity); }, -1);
}
int32_t sg_legal_labels(const sg_game* handle, int32_t* out, int32_t capacity) {
    return guarded([&]() {
        const auto& g = game(handle);
        auto values = g.legal_actions();
        for (auto& code : values) code = shogi::move_label(g.state().position.turn, code);
        return elements(values, out, capacity);
    }, -1);
}
int32_t sg_is_legal(const sg_game* handle, int32_t code) {
    return guarded([&]() -> int32_t { return game(handle).is_legal(code); }, -1);
}
int32_t sg_move_label(const sg_game* handle, int32_t code) {
    return guarded([&]() -> int32_t {
        const auto& g = game(handle);
        if (!g.is_legal(code)) throw shogi::Error("Illegal move");
        return shogi::move_label(g.state().position.turn, code);
    }, -1);
}
int32_t sg_label_to_move(const sg_game* handle, int32_t label) {
    return guarded([&]() -> int32_t {
        const auto& g = game(handle);
        if (label < 0 || label >= SG_ACTION_SIZE) throw shogi::Error("Invalid policy label");
        for (const auto code : g.legal_actions()) if (shogi::move_label(g.state().position.turn, code) == label) return code;
        throw shogi::Error("No legal move has this policy label");
    }, -1);
}
int32_t sg_play(sg_game* handle, int32_t code, int32_t* captured, int32_t* check) {
    return guarded([&]() -> int32_t {
        auto& g = game(handle);
        require_output(captured);
        require_output(check);
        const auto result = g.play(code);
        *captured = result.captured;
        *check = result.check;
        return 0;
    }, -1);
}
int32_t sg_undo(sg_game* handle) {
    return guarded([&]() -> int32_t { return game(handle).undo(); }, -1);
}
int32_t sg_resign(sg_game* handle) {
    return guarded([&]() -> int32_t { game(handle).resign(); return 0; }, -1);
}
int32_t sg_encode(const sg_game* handle, int32_t player, float* out, int32_t capacity) {
    return guarded([&]() { return elements(game(handle).encode(player), out, capacity); }, -1);
}
int32_t sg_legal_mask(const sg_game* handle, int32_t player, float* out, int32_t capacity) {
    return guarded([&]() { return elements(game(handle).legal_mask(player), out, capacity); }, -1);
}
int64_t sg_perft(const sg_game* handle, int32_t depth) {
    return guarded([&]() { return game(handle).perft(depth); }, int64_t{-1});
}
} // extern C
