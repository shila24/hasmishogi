#include "riichi/game.hpp"
#include "game_json.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <tuple>

namespace riichi {
namespace {
constexpr int chi_low = 68, chi_high = 70, pon = 71, kan = 72, kan_base = 73;
constexpr int ron = 107, tsumo_action = 108, pass = 109, nine_terminals = 110;
template<class Range, class T> bool contains(const Range& xs, const T& x) {
    return std::find(xs.begin(), xs.end(), x) != xs.end();
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void remove_tile(std::vector<int>& hand, int tile) {
    const auto it = std::find(hand.begin(), hand.end(), tile);
    require(it != hand.end(), "tile is not in hand");
    hand.erase(it);
}
int checked_score(std::int64_t score) {
    if (score < INT32_MIN || score > INT32_MAX) throw std::overflow_error("score exceeds int32 range");
    return static_cast<int>(score);
}
std::string meld_json(const DeclaredMeld& m, std::optional<int> observer = {}) {
    json::Object out;
    out.list("tiles", m.tiles); out.add("kind", m.kind); out.add("open", m.open);
    out.add("from_player", observer ? (m.from_player - *observer + 4) % 4 : m.from_player);
    if (observer) out.add("called_tile", m.called_tile);
    return out.str();
}
constexpr std::array<const char*, 5> phases{"draw", "discard", "claim", "kan_claim", "terminal"};
}

Meld DeclaredMeld::analysis() const {
    std::vector<int> types;
    for (int tile : tiles) types.push_back(tile / 4);
    return Meld(std::move(types), kind, open);
}
Game::Game(std::uint64_t seed, std::optional<Rules> rules)
    : rules_(rules.value_or(Rules{})), seed_(seed), rng_(seed) {
    if (!rules) rules_.nagashi_mangan = false;
    rules_.validate();
    require(!rules_.pao && rules_.atozuke, "the match engine requires pao=False and atozuke=True");
    scores_.fill(rules_.start_points);
    start_hand();
}
void Game::validate_player(int player) {
    require(player >= 0 && player < 4, "player must be an integer in 0..3");
}
void Game::clear_cache() {
    for (int p = 0; p < 4; ++p) {
        legal_cache_[p].reset(); waits_cache_[p].reset(); shanten_cache_[p].reset();
    }
}
void Game::start_hand() {
    const auto wall = build_wall(rng_);
    live_wall_.assign(wall.live.begin(), wall.live.end()); dead_wall_ = wall.dead;
    for (auto& hand : hands_) hand.clear();
    for (int i = 0; i < 13; ++i) for (int offset = 0; offset < 4; ++offset)
        hands_[(dealer_ + offset) % 4].push_back(live_wall_[4 * i + offset]);
    live_wall_.erase(live_wall_.begin(), live_wall_.begin() + 52);
    for (auto& melds : melds_) melds.clear();
    for (auto& discards : discards_) discards.clear();
    riichi_.fill(false); double_riichi_.fill(false); ippatsu_.fill(false);
    temporary_furiten_.fill(false); riichi_furiten_.fill(false);
    drawn_tiles_.fill(-1); draw_counts_.fill(0);
    uninterrupted_ = true; kan_count_ = 0; dora_count_ = 1; pending_dora_ = 0;
    pending_riichi_ = -1; pending_kan_.reset(); pending_four_kans_ = false;
    forbidden_discards_.clear(); last_discard_.reset(); rinshan_ = false;
    ++hand_number_;
    draw(dealer_);
}
void Game::draw(int player, bool rinshan) {
    if (live_wall_.empty()) { exhaustive_draw(); return; }
    int tile;
    if (rinshan) {
        const int index = kan_count_ - 1;
        tile = dead_wall_.at(index); dead_wall_.at(index) = live_wall_.back(); live_wall_.pop_back();
    } else {
        tile = live_wall_.front(); live_wall_.erase(live_wall_.begin());
    }
    hands_[player].push_back(tile);
    std::sort(hands_[player].begin(), hands_[player].end());
    drawn_tiles_[player] = tile; ++draw_counts_[player]; temporary_furiten_[player] = false;
    turn_ = player; phase_ = RQ_PHASE_DRAW; rinshan_ = rinshan;
    forbidden_discards_.clear(); clear_cache();
}
std::vector<int> Game::actors() const {
    if (terminal_) return {};
    if (phase_ == RQ_PHASE_CLAIM || phase_ == RQ_PHASE_KAN_CLAIM)
        return {(turn_ + 1) % 4, (turn_ + 2) % 4, (turn_ + 3) % 4};
    return {turn_};
}
std::vector<Meld> Game::analysis_melds(int player) const {
    std::vector<Meld> result;
    for (const auto& m : melds_[player]) result.push_back(m.analysis());
    return result;
}
bool Game::closed(int player) const {
    return std::none_of(melds_[player].begin(), melds_[player].end(), [](const auto& m) { return m.open; });
}
std::optional<Counts> Game::before_draw_counts(int player) const {
    auto counts = to_counts(hands_[player]);
    if (hands_[player].size() == 14 - 3 * melds_[player].size()) {
        const int drawn = drawn_tiles_[player];
        if (drawn < 0) return {};
        --counts[drawn / 4];
    }
    return counts;
}
const std::vector<int>& Game::player_waits(int player) const {
    if (!waits_cache_[player]) {
        const auto counts = before_draw_counts(player);
        waits_cache_[player] = counts ? waits(*counts, analysis_melds(player)) : std::vector<int>{};
    }
    return *waits_cache_[player];
}
int Game::player_shanten(int player) const {
    if (!shanten_cache_[player]) shanten_cache_[player] = shanten(to_counts(hands_[player]), analysis_melds(player));
    return *shanten_cache_[player];
}
std::array<bool, 3> Game::furiten(int player) const {
    bool own = false;
    if (rules_.discard_furiten) for (const auto& d : discards_[player])
        own = own || contains(player_waits(player), d.tile / 4);
    return {own, rules_.temporary_furiten && temporary_furiten_[player], rules_.riichi_furiten && riichi_furiten_[player]};
}
std::vector<int> Game::dora_indicators() const {
    std::vector<int> result;
    for (int i = 0; i < dora_count_; ++i) result.push_back(dead_wall_.at(4 + 2 * i));
    return result;
}
std::optional<Score> Game::score(int player, int tile, bool tsumo, bool chankan, int honba) const {
    auto hand = hands_[player];
    if (!tsumo) {
        if (std::count_if(hand.begin(), hand.end(), [tile](int t) { return t / 4 == tile / 4; }) == 4) return {};
        hand.push_back(tile);
    }
    const auto counts = to_counts(hand);
    const auto melds = analysis_melds(player);
    if (decompose(counts, melds, tile / 4).empty()) return {};
    if (!tsumo) { const auto flags = furiten(player); if (contains(flags, true)) return {}; }
    const bool first = uninterrupted_ && discards_[player].empty() && !riichi_[player];
    const bool rinshan = tsumo && rinshan_;
    const bool last = live_wall_.empty() && !rinshan && !chankan && !rinshan_;
    WinContext ctx;
    ctx.tsumo = tsumo; ctx.seat_wind = 27 + (player - dealer_ + 4) % 4; ctx.round_wind = 27 + round_index_ / 4;
    ctx.riichi = riichi_[player]; ctx.double_riichi = double_riichi_[player]; ctx.ippatsu = ippatsu_[player];
    ctx.rinshan = rinshan; ctx.chankan = chankan; ctx.haitei = tsumo && last; ctx.houtei = !tsumo && last;
    ctx.tenhou = first && tsumo && player == dealer_; ctx.chiihou = first && tsumo && player != dealer_;
    ctx.renhou = first && !tsumo && player != dealer_ && draw_counts_[player] == 0;
    for (int i = 0; i < dora_count_; ++i) {
        ctx.dora_indicators.push_back(dead_wall_.at(4 + 2 * i) / 4);
        ctx.ura_indicators.push_back(dead_wall_.at(5 + 2 * i) / 4);
    }
    for (const auto& m : melds_[player]) hand.insert(hand.end(), m.tiles.begin(), m.tiles.end());
    try { return score_hand(counts, tile / 4, melds, ctx, rules_, honba < 0 ? honba_ : honba, 0, hand); }
    catch (const NoYakuError&) { return {}; }
}
std::vector<std::pair<int, std::string>> Game::kan_options(int player) const {
    if (kan_count_ >= 4 || live_wall_.empty() || phase_ != RQ_PHASE_DRAW) return {};
    const auto counts = to_counts(hands_[player]);
    std::vector<std::pair<int, std::string>> options;
    for (int t = 0; t < 34; ++t) if (counts[t] == 4) options.emplace_back(t, "closed");
    if (!riichi_[player]) {
        for (const auto& m : melds_[player]) if (m.kind == "pon" && counts[m.tiles[0] / 4])
            options.emplace_back(m.tiles[0] / 4, "added");
        return options;
    }
    const int drawn = drawn_tiles_[player];
    const auto old_melds = analysis_melds(player);
    const auto old_waits = waits(*before_draw_counts(player), old_melds);
    std::vector<std::pair<int, std::string>> allowed;
    for (const auto& [t, kind] : options) {
        if (drawn < 0 || drawn / 4 != t) continue;
        auto after = counts; after[t] -= 4;
        auto new_melds = old_melds; new_melds.emplace_back(std::vector<int>(4, t), "kan", false);
        if (!old_waits.empty() && waits(after, new_melds) == old_waits) allowed.emplace_back(t, kind);
    }
    return allowed;
}
std::array<int, 3> Game::chi_tiles(int action, int tile) {
    const int low = tile - (action - chi_low); return {low, low + 1, low + 2};
}
std::set<int> Game::call_forbidden(int action, int tile) {
    std::set<int> result{tile};
    if (action == chi_low && tile % 9 < 6) result.insert(tile + 3);
    if (action == chi_high && tile % 9 > 2) result.insert(tile - 3);
    return result;
}
std::vector<int> Game::legal_actions(int player) const {
    validate_player(player);
    if (!contains(actors(), player)) return {};
    if (legal_cache_[player]) return *legal_cache_[player];
    const auto counts = to_counts(hands_[player]);
    const auto melds = analysis_melds(player);
    std::vector<int> actions;
    if (phase_ == RQ_PHASE_CLAIM || phase_ == RQ_PHASE_KAN_CLAIM) {
        actions.push_back(pass);
        const int tile_id = phase_ == RQ_PHASE_CLAIM ? last_discard_->second : kan_tile(turn_, pending_kan_->tile);
        const int tile = tile_id / 4;
        const auto result = score(player, tile_id, false, phase_ == RQ_PHASE_KAN_CLAIM);
        if (result && (phase_ != RQ_PHASE_KAN_CLAIM || pending_kan_->kind != "closed" || result->decomposition.kind == "kokushi"))
            actions.push_back(ron);
        if (phase_ == RQ_PHASE_CLAIM && !riichi_[player] && !live_wall_.empty()) {
            bool other = false;
            for (int t = 0; t < 34; ++t) other = other || (counts[t] && t != tile);
            if (counts[tile] >= 2 && other) actions.push_back(pon);
            if (counts[tile] == 3 && kan_count_ < 4) actions.push_back(kan);
            if (player == (turn_ + 1) % 4 && tile < 27) for (int action = chi_low; action <= chi_high; ++action) {
                const auto seq = chi_tiles(action, tile);
                if (seq[0] < 0 || seq[2] >= 27 || seq[0] / 9 != seq[2] / 9) continue;
                auto after = counts; bool enough = true;
                for (int t : seq) if (t != tile) { enough = enough && counts[t] > 0; --after[t]; }
                if (!enough) continue;
                const auto forbidden = call_forbidden(action, tile);
                for (int t = 0; t < 34; ++t) if (after[t] && !forbidden.contains(t)) { actions.push_back(action); break; }
            }
        }
    } else {
        const int drawn = drawn_tiles_[player];
        if (riichi_[player]) actions.push_back(drawn / 4);
        else for (int t = 0; t < 34; ++t) if (counts[t] && !forbidden_discards_.contains(t)) actions.push_back(t);
        if (phase_ == RQ_PHASE_DRAW) {
            if (score(player, drawn, true)) actions.push_back(tsumo_action);
            if (!riichi_[player] && closed(player) && live_wall_.size() >= 4 && scores_[player] >= rules_.riichi_stick && player_shanten(player) <= 0) {
                const auto discards = actions;
                for (int t : discards) if (t < 34) {
                    auto after = counts; --after[t];
                    if (!waits(after, melds).empty()) actions.push_back(34 + t);
                }
            }
            const auto kans = kan_options(player);
            if (kans.size() == 1) actions.push_back(kan);
            else for (const auto& [t, kind] : kans) actions.push_back(kan_base + t);
            int orphans = 0;
            for (int t = 0; t < 34; ++t) orphans += counts[t] > 0 && orphan(t);
            if (rules_.kyuushu_kyuuhai && uninterrupted_ && discards_[player].empty() && orphans >= 9) actions.push_back(nine_terminals);
        }
    }
    std::sort(actions.begin(), actions.end()); legal_cache_[player] = actions; return actions;
}
std::vector<int> Game::take(int player, int tile, int count) {
    std::vector<int> choices;
    for (int t : hands_[player]) if (t / 4 == tile) choices.push_back(t);
    std::sort(choices.begin(), choices.end(), [&](int a, int b) {
        return std::pair(rules_.red_fives && red_id(a), a) < std::pair(rules_.red_fives && red_id(b), b);
    });
    require(static_cast<int>(choices.size()) >= count, "insufficient tiles");
    choices.resize(count);
    for (int t : choices) remove_tile(hands_[player], t);
    return choices;
}
int Game::kan_tile(int player, int tile) const {
    const int drawn = drawn_tiles_[player];
    if (drawn >= 0 && drawn / 4 == tile) return drawn;
    int result = 136;
    for (int t : hands_[player]) if (t / 4 == tile) result = std::min(result, t);
    require(result < 136, "insufficient tiles"); return result;
}
void Game::play_joint(const Joint& actions) {
    const auto current = actors();
    require(!terminal_, "joint actions must match the current actors");
    for (int p = 0; p < 4; ++p)
        require((actions[p] != -1) == contains(current, p), "joint actions must match the current actors");
    for (int p : current) if (!contains(legal_actions(p), actions[p]))
        throw std::invalid_argument("illegal action for player " + std::to_string(p) + ": " + std::to_string(actions[p]));
    Game next = *this;
    next.apply_joint(actions);
    *this = std::move(next);
}
void Game::apply_joint(const Joint& actions) {
    const int hand_before = hand_number_;
    if (phase_ == RQ_PHASE_CLAIM || phase_ == RQ_PHASE_KAN_CLAIM) resolve_claims(actions);
    else {
        const int action = actions[turn_];
        if (action < 68) discard(turn_, action % 34, action >= 34);
        else if (action == tsumo_action) win({turn_}, drawn_tiles_[turn_], true);
        else if (action == nine_terminals) finish_hand("kyuushu_kyuuhai", true, true);
        else {
            const auto options = kan_options(turn_);
            const int tile = action == kan ? options.front().first : action - kan_base;
            const auto it = std::find_if(options.begin(), options.end(), [tile](const auto& x) { return x.first == tile; });
            pending_kan_ = PendingKan{turn_, it->second, tile}; phase_ = RQ_PHASE_KAN_CLAIM; uninterrupted_ = false;
        }
    }
    moves_.push_back(actions);
    if (hand_number_ != hand_before) hand_start_ = static_cast<int>(moves_.size());
    clear_cache();
}
void Game::discard(int player, int tile, bool riichi) {
    const int drawn = drawn_tiles_[player];
    int physical;
    if (drawn >= 0 && drawn / 4 == tile) { physical = drawn; remove_tile(hands_[player], physical); }
    else physical = take(player, tile).front();
    if (riichi_[player]) ippatsu_[player] = false;
    discards_[player].push_back({physical, rules_.track_tsumogiri && physical == drawn, riichi, false});
    drawn_tiles_[player] = -1; last_discard_ = {player, physical}; pending_riichi_ = riichi ? player : -1;
    phase_ = RQ_PHASE_CLAIM; dora_count_ += pending_dora_; pending_dora_ = 0;
}
void Game::accept_riichi() {
    if (pending_riichi_ < 0) return;
    const int p = pending_riichi_;
    scores_[p] = checked_score(static_cast<std::int64_t>(scores_[p]) - rules_.riichi_stick);
    ++riichi_sticks_; riichi_[p] = true; double_riichi_[p] = uninterrupted_ && discards_[p].size() == 1;
    ippatsu_[p] = rules_.ippatsu; pending_riichi_ = -1;
}
void Game::resolve_claims(const Joint& actions) {
    const int source = turn_;
    const auto current = actors();
    std::vector<int> winners;
    for (int p : current) if (actions[p] == ron) winners.push_back(p);
    for (int p : current) if (actions[p] != ron && (phase_ == RQ_PHASE_CLAIM ?
            contains(player_waits(p), last_discard_->second / 4) : contains(legal_actions(p), ron))) {
        temporary_furiten_[p] = true; if (riichi_[p]) riichi_furiten_[p] = true;
    }
    if (!winners.empty()) {
        if (winners.size() == 3 && rules_.triple_ron_abort) {
            accept_riichi(); finish_hand("triple_ron", true, true); return;
        }
        if (rules_.atama_hane || !rules_.double_ron) winners.resize(1);
        const bool chankan = phase_ == RQ_PHASE_KAN_CLAIM;
        const int tile = chankan ? kan_tile(source, pending_kan_->tile) : last_discard_->second;
        win(winners, tile, false, chankan); return;
    }
    if (phase_ == RQ_PHASE_KAN_CLAIM) {
        const auto pending = *pending_kan_;
        complete_kan(pending.player, pending.kind, pending.tile); return;
    }
    accept_riichi();
    if (rules_.suucha_riichi && std::all_of(riichi_.begin(), riichi_.end(), [](bool b) { return b; })) {
        finish_hand("suucha_riichi", true, true); return;
    }
    if (rules_.suufon_renda && uninterrupted_ && std::all_of(discards_.begin(), discards_.end(), [](const auto& d) { return d.size() == 1; })) {
        const int first = discards_[0][0].tile / 4;
        if (first >= 27 && first <= 30 && std::all_of(discards_.begin(), discards_.end(), [first](const auto& d) { return d[0].tile / 4 == first; })) {
            finish_hand("suufon_renda", true, true); return;
        }
    }
    if (pending_four_kans_) { finish_hand("suukaikan", true, true); return; }
    for (int p : current) if (actions[p] == pon || actions[p] == kan) { call(p, actions[p]); return; }
    for (int p : current) if (actions[p] >= chi_low && actions[p] <= chi_high) { call(p, actions[p]); return; }
    draw((source + 1) % 4);
}
void Game::call(int player, int action) {
    const auto [source, physical] = *last_discard_;
    const int tile = physical / 4;
    discards_[source].back().called = true; ippatsu_.fill(false); uninterrupted_ = false;
    if (action == kan) {
        auto owned = take(player, tile, 3); owned.push_back(physical);
        melds_[player].push_back({std::move(owned), "kan", true, source, physical}); after_kan(player, true); return;
    }
    std::vector<int> owned;
    if (action == pon) owned = take(player, tile, 2);
    else for (int t : chi_tiles(action, tile)) if (t != tile) owned.push_back(take(player, t).front());
    owned.push_back(physical);
    melds_[player].push_back({std::move(owned), action == pon ? "pon" : "chi", true, source, physical});
    turn_ = player; phase_ = RQ_PHASE_DISCARD; drawn_tiles_[player] = -1;
    forbidden_discards_ = call_forbidden(action, tile);
}
void Game::complete_kan(int player, const std::string& kind, int tile) {
    if (kind == "closed") melds_[player].push_back({take(player, tile, 4), "kan", false, player, {}});
    else {
        const int physical = kan_tile(player, tile); remove_tile(hands_[player], physical);
        auto it = std::find_if(melds_[player].begin(), melds_[player].end(), [tile](const auto& m) { return m.kind == "pon" && m.tiles[0] / 4 == tile; });
        it->tiles.push_back(physical); it->kind = "kan";
    }
    pending_kan_.reset(); after_kan(player, kind == "added");
}
void Game::after_kan(int player, bool opened) {
    ippatsu_.fill(false); ++kan_count_;
    if (rules_.kan_dora) {
        const bool immediate = opened ? !rules_.open_kan_dora_after_discard : rules_.closed_kan_dora_immediate;
        if (immediate) ++dora_count_; else ++pending_dora_;
    }
    int owners = 0;
    for (const auto& melds : melds_) owners += std::any_of(melds.begin(), melds.end(), [](const auto& m) { return m.kind == "kan"; });
    pending_four_kans_ = rules_.suukaikan && kan_count_ == 4 && owners > 1;
    draw(player, true);
}
void Game::win(const std::vector<int>& winners, int tile, bool tsumo, bool chankan) {
    std::vector<Score> results;
    for (std::size_t i = 0; i < winners.size(); ++i) {
        const auto result = score(winners[i], tile, tsumo, chankan, i == 0 ? honba_ : 0);
        require(result.has_value(), "win no longer eligible"); results.push_back(*result);
    }
    for (std::size_t i = 0; i < winners.size(); ++i) {
        const auto delta = settle_payment(results[i].payment, winners[i], dealer_, tsumo ? std::optional<int>{} : turn_);
        for (int p = 0; p < 4; ++p) scores_[p] = checked_score(scores_[p] + delta[p]);
    }
    scores_[winners[0]] = checked_score(scores_[winners[0]] + static_cast<std::int64_t>(riichi_sticks_) * rules_.riichi_stick);
    riichi_sticks_ = 0; pending_riichi_ = -1;
    finish_hand(tsumo ? "tsumo" : chankan ? "chankan" : "ron", contains(winners, dealer_), false, winners, {}, tile, tsumo ? -1 : turn_);
}
void Game::exhaustive_draw() {
    if (rules_.nagashi_mangan) {
        std::vector<int> winners;
        for (int p = 0; p < 4; ++p) {
            std::vector<int> tiles; bool called = false;
            for (const auto& d : discards_[p]) { tiles.push_back(d.tile / 4); called = called || d.called; }
            if (nagashi_eligible(tiles, called, !closed(p), rules_)) winners.push_back(p);
        }
        if (!winners.empty()) {
            for (int p : winners) {
                const auto pay = payments(2000, p == dealer_, true, honba_, 0, rules_);
                const auto delta = settle_payment(pay, p, dealer_);
                for (int seat = 0; seat < 4; ++seat) scores_[seat] = checked_score(scores_[seat] + delta[seat]);
            }
            scores_[winners[0]] = checked_score(scores_[winners[0]] + static_cast<std::int64_t>(riichi_sticks_) * rules_.riichi_stick);
            riichi_sticks_ = 0; finish_hand("nagashi_mangan", contains(winners, dealer_), false, winners); return;
        }
    }
    std::array<bool, 4> tenpai{};
    for (int p = 0; p < 4; ++p) tenpai[p] = !waits(to_counts(hands_[p]), analysis_melds(p)).empty();
    const auto delta = tenpai_payments(tenpai, rules_);
    for (int p = 0; p < 4; ++p) scores_[p] = checked_score(scores_[p] + delta[p]);
    finish_hand("exhaustive_draw", tenpai[dealer_], false, {}, tenpai);
}
std::array<int, 4> Game::ranking() const {
    std::array<int, 4> result{0, 1, 2, 3};
    std::sort(result.begin(), result.end(), [&](int a, int b) { return scores_[a] == scores_[b] ? a < b : scores_[a] > scores_[b]; });
    return result;
}
std::array<int, 4> Game::placement() const {
    std::array<int, 4> result{}; const auto order = ranking();
    for (int i = 0; i < 4; ++i) result[order[i]] = i + 1;
    return result;
}
void Game::finish_hand(const std::string& reason, bool repeat, bool abort, const std::vector<int>& winners,
                       std::optional<std::array<bool, 4>> tenpai, int winning_tile, int discarder) {
    clear_cache();
    json::Object summary;
    summary.add("reason", reason); summary.list("winners", winners);
    summary.add("dealer", dealer_); summary.add("round_index", round_index_); summary.add("honba", honba_);
    summary.add("riichi_sticks", riichi_sticks_); summary.add("hand_number", hand_number_);
    summary.add("winning_tile", winning_tile < 0 ? std::optional<int>{} : winning_tile);
    summary.add("discarder", discarder < 0 ? std::optional<int>{} : discarder);
    std::array<bool, 4> ready{};
    std::vector<std::string> all_waits, all_counts, all_hands, all_melds;
    for (int p = 0; p < 4; ++p) {
        Counts wait_mask{}; for (int t : player_waits(p)) wait_mask[t] = 1;
        ready[p] = !player_waits(p).empty();
        all_waits.push_back(json::array(wait_mask)); all_counts.push_back(json::array(to_counts(hands_[p])));
        auto hand = hands_[p]; std::sort(hand.begin(), hand.end()); all_hands.push_back(json::array(hand));
        std::vector<std::string> melds; for (const auto& m : melds_[p]) melds.push_back(meld_json(m));
        all_melds.push_back(json::objects(melds));
    }
    summary.list("tenpai", tenpai.value_or(ready)); summary.raw("waits", json::objects(all_waits));
    summary.raw("hand_counts", json::objects(all_counts)); summary.raw("hands", json::objects(all_hands));
    summary.raw("melds", json::objects(all_melds)); summary.list("riichi", riichi_); summary.list("dora_indicators", dora_indicators());
    const int limit = rules_.rounds == RQ_ROUNDS_TONPUU ? 4 : 8;
    int end = RQ_END_ONGOING;
    if (rules_.tobi && *std::min_element(scores_.begin(), scores_.end()) < 0) end = RQ_END_TOBI;
    else if (rules_.agari_yame && round_index_ == limit - 1 && contains(winners, dealer_) && ranking()[0] == dealer_ && scores_[dealer_] >= rules_.return_points)
        end = RQ_END_AGARI_YAME;
    honba_ = repeat || winners.empty() ? honba_ + 1 : 0;
    if (!repeat && !abort) { ++round_index_; dealer_ = (dealer_ + 1) % 4; }
    if (round_index_ >= limit && !end) end = RQ_END_ROUNDS_COMPLETE;
    if (end) {
        const int top = ranking()[0];
        scores_[top] = checked_score(scores_[top] + static_cast<std::int64_t>(riichi_sticks_) * rules_.riichi_stick);
        riichi_sticks_ = 0; final_points_ = placement_points(scores_, rules_);
        terminal_ = true; winner_ = ranking()[0]; reason_ = end; phase_ = RQ_PHASE_TERMINAL;
        pending_riichi_ = -1; pending_kan_.reset(); clear_cache();
    }
    summary.list("placement", placement()); summary.list("scores", scores_); last_hand_ = summary.str();
    if (!end) start_hand();
}
rq_state Game::state() const {
    rq_state out{};
    out.terminal = terminal_; out.winner = winner_; out.reason = reason_; out.phase = phase_; out.turn = turn_;
    out.dealer = dealer_; out.round_index = round_index_; out.honba = honba_; out.riichi_sticks = riichi_sticks_;
    out.hand_number = hand_number_; out.hand_start = hand_start_; out.num_moves = static_cast<int>(moves_.size());
    out.remaining_wall = static_cast<int>(live_wall_.size()); out.seed = seed_;
    std::copy(scores_.begin(), scores_.end(), out.scores); std::copy(final_points_.begin(), final_points_.end(), out.final_points);
    return out;
}
std::string Game::observation_json(int player) const {
    validate_player(player);
    json::Object out;
    out.add("player", player); out.list("hand", hands_[player]); out.list("hand_counts", to_counts(hands_[player]));
    out.add("drawn_tile", drawn_tiles_[player] < 0 ? std::optional<int>{} : drawn_tiles_[player]);
    std::vector<std::string> discards, melds;
    std::array<bool, 4> ready{}, doubles{}, ippatsu{};
    std::array<int, 4> scores{}, places{}, winds{};
    const auto standing = placement();
    for (int row = 0; row < 4; ++row) {
        const int p = (player + row) % 4;
        std::vector<std::string> river, called;
        for (const auto& d : discards_[p]) {
            json::Object item; item.add("tile", d.tile); item.add("tsumogiri", d.tsumogiri);
            item.add("riichi", d.riichi); item.add("called", d.called); river.push_back(item.str());
        }
        for (const auto& m : melds_[p]) called.push_back(meld_json(m, player));
        discards.push_back(json::objects(river)); melds.push_back(json::objects(called));
        ready[row] = riichi_[p]; doubles[row] = double_riichi_[p]; ippatsu[row] = ippatsu_[p];
        scores[row] = scores_[p]; places[row] = standing[p]; winds[row] = 27 + (p - dealer_ + 4) % 4;
    }
    out.raw("discards", json::objects(discards)); out.raw("melds", json::objects(melds));
    out.list("riichi", ready); out.list("double_riichi", doubles); out.list("ippatsu", ippatsu);
    out.list("dora_indicators", dora_indicators()); out.add("red_fives", bool(rules_.red_fives));
    out.add("remaining_wall", static_cast<int>(live_wall_.size())); out.list("scores", scores); out.list("placement", places);
    out.add("round_wind", 27 + std::min(round_index_ / 4, 3)); out.list("seat_winds", winds);
    out.add("round_index", round_index_); out.add("dealer", (dealer_ - player + 4) % 4);
    out.add("honba", honba_); out.add("riichi_sticks", riichi_sticks_); out.list("furiten", furiten(player));
    out.add("shanten", player_shanten(player)); out.list("waits", player_waits(player));
    out.add("turn", (turn_ - player + 4) % 4); out.add("phase", std::string_view(phases[phase_]));
    out.raw("last_discard", last_discard_ ? json::array(std::array<int, 2>{(last_discard_->first - player + 4) % 4, last_discard_->second}) : "null");
    out.raw("pending_kan", pending_kan_ ? "[" + std::to_string((pending_kan_->player - player + 4) % 4) + ',' + json::value(pending_kan_->kind) + ',' + std::to_string(pending_kan_->tile) + ']' : "null");
    out.list("legal_actions", legal_actions(player)); out.add("terminal", terminal_);
    return out.str();
}
void Game::redeal(const rq_hidden& hidden) {
    Game next = *this; next.replace_hidden(hidden); *this = std::move(next);
}
void Game::replace_hidden(const rq_hidden& h) {
    require(!terminal_, "cannot redeal a terminal game");
    std::vector<int> old, tiles;
    std::array<std::vector<int>, 4> hands;
    for (int p = 0; p < 4; ++p) {
        require(h.hand_sizes[p] >= 0 && h.hand_sizes[p] <= RQ_MAX_HAND, "invalid hand size");
        require((h.temporary_furiten[p] == 0 || h.temporary_furiten[p] == 1) && (h.riichi_furiten[p] == 0 || h.riichi_furiten[p] == 1), "furiten flags must be booleans");
        hands[p].assign(h.hands[p], h.hands[p] + h.hand_sizes[p]);
        tiles.insert(tiles.end(), hands[p].begin(), hands[p].end());
        old.insert(old.end(), hands_[p].begin(), hands_[p].end());
    }
    require(h.live_count >= 0 && h.live_count <= RQ_LIVE_WALL, "invalid live wall length");
    tiles.insert(tiles.end(), h.live_wall, h.live_wall + h.live_count); tiles.insert(tiles.end(), std::begin(h.dead_wall), std::end(h.dead_wall));
    old.insert(old.end(), live_wall_.begin(), live_wall_.end()); old.insert(old.end(), dead_wall_.begin(), dead_wall_.end());
    std::sort(tiles.begin(), tiles.end()); std::sort(old.begin(), old.end());
    require(tiles == old, "redeal must permute exactly the hidden tiles");
    for (int p = 0; p < 4; ++p) require(hands[p].size() == hands_[p].size(), "hand sizes and wall lengths are public and must not change");
    require(h.live_count == static_cast<int>(live_wall_.size()), "hand sizes and wall lengths are public and must not change");
    for (int i = 0; i < dora_count_; ++i) require(h.dead_wall[4 + 2 * i] == dead_wall_[4 + 2 * i], "revealed dora indicators must not change");
    for (int p = 0; p < 4; ++p) {
        require((h.drawn_tiles[p] == -1) == (drawn_tiles_[p] == -1), "which seats hold a drawn tile is public");
        require(h.drawn_tiles[p] == -1 || contains(hands[p], h.drawn_tiles[p]), "a drawn tile must be in its seat hand");
        std::sort(hands[p].begin(), hands[p].end());
    }
    hands_ = std::move(hands); live_wall_.assign(h.live_wall, h.live_wall + h.live_count);
    std::copy_n(h.dead_wall, 14, dead_wall_.begin()); std::copy_n(h.drawn_tiles, 4, drawn_tiles_.begin());
    std::copy_n(h.temporary_furiten, 4, temporary_furiten_.begin()); std::copy_n(h.riichi_furiten, 4, riichi_furiten_.begin());
    seed_ = h.seed; rng_.reseed(seed_); clear_cache();
}
}
