#pragma once
#include "riichi/core.hpp"
#include <array>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace riichi {
struct Discard {
    int tile;
    bool tsumogiri, riichi = false, called = false;
};
struct DeclaredMeld {
    std::vector<int> tiles;
    std::string kind;
    bool open;
    int from_player;
    std::optional<int> called_tile{};
    Meld analysis() const;
};
struct PendingKan { int player; std::string kind; int tile; };
using Joint = std::array<int, 4>;

// Value-owned full match state. Public transitions provide the strong guarantee.
class Game {
public:
    explicit Game(std::uint64_t seed = 0, std::optional<Rules> rules = {});
    std::vector<int> actors() const;
    std::vector<int> legal_actions(int player) const;
    void play_joint(const Joint& actions);
    void redeal(const rq_hidden& hidden);
    rq_state state() const;
    std::string observation_json(int player) const;
    const std::string& last_hand_json() const { return last_hand_; }
    void encode(int player, std::span<float> output) const;
    void legal_mask(int player, std::span<float> output) const;
    const Rules& rules() const { return rules_; }
    const std::vector<Joint>& moves() const { return moves_; }
    static void validate_player(int player);
private:
    Rules rules_;
    std::uint64_t seed_;
    Random rng_;
    std::vector<Joint> moves_;
    std::array<int, 4> scores_{};
    int dealer_ = 0, round_index_ = 0, honba_ = 0, riichi_sticks_ = 0;
    int hand_number_ = 0, hand_start_ = 0;
    bool terminal_ = false;
    int winner_ = -2, reason_ = RQ_END_ONGOING;
    std::array<double, 4> final_points_{};
    std::string last_hand_ = "{}";
    std::vector<int> live_wall_;
    std::array<int, 14> dead_wall_{};
    std::array<std::vector<int>, 4> hands_;
    std::array<std::vector<DeclaredMeld>, 4> melds_;
    std::array<std::vector<Discard>, 4> discards_;
    std::array<bool, 4> riichi_{}, double_riichi_{}, ippatsu_{}, temporary_furiten_{}, riichi_furiten_{};
    std::array<int, 4> drawn_tiles_{-1, -1, -1, -1}, draw_counts_{};
    bool uninterrupted_ = true, pending_four_kans_ = false, rinshan_ = false;
    int kan_count_ = 0, dora_count_ = 1, pending_dora_ = 0, pending_riichi_ = -1;
    int turn_ = 0, phase_ = RQ_PHASE_DRAW;
    std::optional<PendingKan> pending_kan_;
    std::optional<std::pair<int, int>> last_discard_;
    std::set<int> forbidden_discards_;
    mutable std::array<std::optional<std::vector<int>>, 4> legal_cache_, waits_cache_;
    mutable std::array<std::optional<int>, 4> shanten_cache_;
    void clear_cache();
    void start_hand();
    void draw(int player, bool rinshan = false);
    std::vector<Meld> analysis_melds(int player) const;
    bool closed(int player) const;
    std::optional<Counts> before_draw_counts(int player) const;
    const std::vector<int>& player_waits(int player) const;
    int player_shanten(int player) const;
    std::array<bool, 3> furiten(int player) const;
    std::optional<Score> score(int player, int tile, bool tsumo = false, bool chankan = false, int honba = -1) const;
    std::vector<std::pair<int, std::string>> kan_options(int player) const;
    static std::array<int, 3> chi_tiles(int action, int tile);
    static std::set<int> call_forbidden(int action, int tile);
    std::vector<int> take(int player, int tile, int count = 1);
    int kan_tile(int player, int tile) const;
    void apply_joint(const Joint& actions);
    void discard(int player, int tile, bool riichi);
    void accept_riichi();
    void resolve_claims(const Joint& actions);
    void call(int player, int action);
    void complete_kan(int player, const std::string& kind, int tile);
    void after_kan(int player, bool opened);
    void win(const std::vector<int>& winners, int tile, bool tsumo = false, bool chankan = false);
    void exhaustive_draw();
    std::array<int, 4> ranking() const;
    std::array<int, 4> placement() const;
    std::vector<int> dora_indicators() const;
    void finish_hand(const std::string& reason, bool repeat, bool abort = false,
                     const std::vector<int>& winners = {}, std::optional<std::array<bool, 4>> tenpai = {},
                     int winning_tile = -1, int discarder = -1);
    void replace_hidden(const rq_hidden& hidden);
};
}
