#include "hasami/c_api.h"
#include "hasami/game.hpp"
#include "hasami/search.hpp"
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string error;
const hasami::Game& game(const void* p) {
    if (!p) throw std::invalid_argument("Null game handle");
    return *static_cast<const hasami::Game*>(p);
}
hasami::Game& game(void* p) {
    if (!p) throw std::invalid_argument("Null game handle");
    return *static_cast<hasami::Game*>(p);
}
hs_position pack(const hasami::Position& p) {
    return {p.black.lo, p.black.hi, p.white.lo, p.white.hi, p.turn, p.pending};
}
void store_exception() {
    try { throw; }
    catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "Unknown native exception"; }
}
}
extern "C" {
int32_t hs_abi_version(void) { return HS_ABI_VERSION; }
int32_t hs_sizeof_position(void) { return sizeof(hs_position); }
int32_t hs_sizeof_rules(void) { return sizeof(hs_rules); }
int32_t hs_sizeof_state(void) { return sizeof(hs_state); }
int32_t hs_sizeof_search_result(void) { return sizeof(hs_search_result); }
const char* hs_last_error(void) { return error.c_str(); }
void* hs_create(const hs_rules* rules, const hs_position* position) {
    try {
        error.clear();
        hasami::Rules r;
        if (rules) {
            if ((rules->edge_capture != 0 && rules->edge_capture != 1) ||
                (rules->no_moves_draw != 0 && rules->no_moves_draw != 1))
                throw std::invalid_argument("Rule switches must be zero or one");
            r = {rules->edge_capture != 0, rules->capture_target, rules->lead_target,
                 rules->repetition_limit, rules->no_moves_draw != 0};
        }
        auto p = position ? hasami::Position{{position->black_lo, position->black_hi},
                   {position->white_lo, position->white_hi}, position->turn, position->pending}
                          : hasami::Position::initial();
        return new hasami::Game(r, p);
    } catch (...) { store_exception(); return nullptr; }
}
void hs_destroy(void* handle) { delete static_cast<hasami::Game*>(handle); }
void* hs_clone(const void* handle) {
    try { error.clear(); return new hasami::Game(game(handle)); }
    catch (...) { store_exception(); return nullptr; }
}
int32_t hs_get_state(const void* handle, hs_state* out) {
    try {
        error.clear();
        if (!out) throw std::invalid_argument("Null state output");
        const auto& g = game(handle);
        *out = {}; out->position = pack(g.position());
        out->winner = g.outcome().winner; out->reason = static_cast<int>(g.outcome().reason);
        out->ply = static_cast<int32_t>(g.moves().size());
        out->captured_lo = g.last_capture().lo; out->captured_hi = g.last_capture().hi;
        return 0;
    } catch (...) { store_exception(); return -1; }
}
int32_t hs_legal_actions(const void* handle, int32_t* out, int32_t capacity) {
    try {
        error.clear();
        if (capacity < 0 || (!out && capacity)) throw std::invalid_argument("Invalid output buffer");
        auto actions = game(handle).legal_actions();
        int count = static_cast<int>(actions.size());
        if (out && capacity < count) throw std::invalid_argument("Output buffer is too small");
        if (out) std::copy(actions.begin(), actions.end(), out);
        return count;
    } catch (...) { store_exception(); return -1; }
}
int32_t hs_history(const void* handle, hs_position* out, int32_t capacity) {
    try {
        error.clear();
        if (capacity < 0 || (!out && capacity)) throw std::invalid_argument("Invalid output buffer");
        const auto& history = game(handle).history();
        int count = static_cast<int>(history.size());
        if (out && capacity < count) throw std::invalid_argument("Output buffer is too small");
        if (out) for (int i = 0; i < count; ++i) out[i] = pack(history[i]);
        return count;
    } catch (...) { store_exception(); return -1; }
}
int32_t hs_play(void* handle, int32_t action) {
    try { error.clear(); game(handle).play(action); return 0; }
    catch (...) { store_exception(); return -1; }
}
int32_t hs_undo(void* handle) {
    try { error.clear(); return game(handle).undo() ? 1 : 0; }
    catch (...) { store_exception(); return -1; }
}
int32_t hs_search(const void* handle, int32_t depth, int32_t time_ms,
                  uint64_t node_limit, hs_search_result* out) {
    try {
        error.clear();
        if (!out) throw std::invalid_argument("Null search output");
        hasami::AlphaBetaAgent agent;
        auto r = agent.choose(game(handle), {depth, time_ms, node_limit});
        *out = {r.action, r.score, r.depth, r.completed ? 1 : 0, r.nodes, r.elapsed_ms};
        return 0;
    } catch (...) { store_exception(); return -1; }
}
int32_t hs_perft(const void* handle, int32_t depth, uint64_t* out) {
    try {
        error.clear();
        if (!out || depth < 0 || depth > 6) throw std::invalid_argument("Invalid perft request");
        auto copy = game(handle); *out = hasami::perft(copy, depth); return 0;
    } catch (...) { store_exception(); return -1; }
}
}
