#include "riichi/game.hpp"
#include <algorithm>
#include <cstring>
#include <memory>

namespace {
using riichi::abi::guarded;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
const riichi::Game& game(const void* handle) {
    require(handle, "game handle must not be NULL"); return *static_cast<const riichi::Game*>(handle);
}
riichi::Game& game(void* handle) {
    require(handle, "game handle must not be NULL"); return *static_cast<riichi::Game*>(handle);
}
template<class F> void* guarded_pointer(F&& function) noexcept {
    void* result = nullptr;
    guarded([&] { result = function(); return 0; });
    return result;
}
int write_json(const std::string& value, char* out, int capacity) {
    require(capacity >= 0, "invalid capacity");
    const int size = static_cast<int>(value.size());
    if (!out && !capacity) return size;
    require(out && capacity > size, "JSON output capacity too small");
    std::memcpy(out, value.c_str(), value.size() + 1); return size;
}
int write_elements(const std::vector<int>& values, int32_t* out, int capacity) {
    require(capacity >= 0, "invalid capacity");
    const int size = static_cast<int>(values.size());
    if (!out && !capacity) return size;
    require(out && capacity >= size, "element output capacity too small");
    std::copy(values.begin(), values.end(), out); return size;
}
}
extern "C" {
void* rq_create(uint64_t seed, const rq_rules* rules) {
    return guarded_pointer([&] {
        return std::make_unique<riichi::Game>(seed, rules ? std::optional<riichi::Rules>{riichi::Rules(*rules)} : std::nullopt).release();
    });
}
void rq_destroy(void* handle) {
    guarded([&] { const std::unique_ptr<riichi::Game> owned(static_cast<riichi::Game*>(handle)); return 0; });
}
void* rq_clone(const void* handle) {
    return guarded_pointer([&] { return std::make_unique<riichi::Game>(game(handle)).release(); });
}
int32_t rq_get_state(const void* handle, rq_state* out) {
    return guarded([&] { require(out, "state output must not be NULL"); *out = game(handle).state(); return 0; });
}
int32_t rq_get_rules(const void* handle, rq_rules* out) {
    return guarded([&] { require(out, "rules output must not be NULL"); *out = game(handle).rules(); return 0; });
}
int32_t rq_actors(const void* handle, int32_t* out) {
    return guarded([&] { require(out, "actors output must not be NULL"); const auto actors = game(handle).actors();
        std::copy(actors.begin(), actors.end(), out); return static_cast<int>(actors.size()); });
}
int32_t rq_legal_actions(const void* handle, int32_t player, int32_t* out, int32_t capacity) {
    return guarded([&] { return write_elements(game(handle).legal_actions(player), out, capacity); });
}
int32_t rq_play_joint(void* handle, const int32_t* actions) {
    return guarded([&] { require(actions, "joint actions must not be NULL"); riichi::Joint joint{};
        std::copy_n(actions, 4, joint.begin()); game(handle).play_joint(joint); return 0; });
}
int32_t rq_encode(const void* handle, int32_t player, float* out, int32_t size) {
    return guarded([&] { require(out && size == RQ_PLANES_SIZE, "invalid planes buffer");
        game(handle).encode(player, {out, RQ_PLANES_SIZE}); return 0; });
}
int32_t rq_legal_mask(const void* handle, int32_t player, float* out, int32_t size) {
    return guarded([&] { require(out && size == RQ_ACTION_SIZE, "invalid mask buffer");
        game(handle).legal_mask(player, {out, RQ_ACTION_SIZE}); return 0; });
}
int32_t rq_moves(const void* handle, int32_t* out, int32_t capacity) {
    return guarded([&] {
        const auto& moves = game(handle).moves();
        require(moves.size() <= INT32_MAX / 4, "move log exceeds ABI capacity");
        const int size = static_cast<int>(moves.size()) * 4;
        require(capacity >= 0, "invalid capacity");
        if (!out && !capacity) return size;
        require(out && capacity >= size, "moves output capacity too small");
        for (std::size_t i = 0; i < moves.size(); ++i) std::copy(moves[i].begin(), moves[i].end(), out + 4 * i);
        return size;
    });
}
int32_t rq_observation_json(const void* handle, int32_t player, char* out, int32_t capacity) {
    return guarded([&] { return write_json(game(handle).observation_json(player), out, capacity); });
}
int32_t rq_last_hand_json(const void* handle, char* out, int32_t capacity) {
    return guarded([&] { return write_json(game(handle).last_hand_json(), out, capacity); });
}
int32_t rq_redeal(void* handle, const rq_hidden* hidden) {
    return guarded([&] { require(hidden, "hidden state must not be NULL"); game(handle).redeal(*hidden); return 0; });
}
}
