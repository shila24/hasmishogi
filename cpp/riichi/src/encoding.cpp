#include "riichi/encoding.hpp"
#include <algorithm>

namespace riichi {
void Game::encode(int player, std::span<float> output) const {
    validate_player(player);
    if (output.size() != RQ_PLANES_SIZE) throw std::invalid_argument("invalid planes size");
    std::fill(output.begin(), output.end(), 0.0F);
    const auto cell = [&](int plane, int row, int tile) -> float& { return output[(plane * 4 + row) * 34 + tile]; };
    const auto row_constant = [&](int plane, int row, double value) {
        std::fill_n(output.begin() + (plane * 4 + row) * 34, 34, static_cast<float>(value));
    };
    const auto constant = [&](int plane, double value) { for (int row = 0; row < 4; ++row) row_constant(plane, row, value); };
    const auto counts = to_counts(hands_[player]);
    for (int t = 0; t < 34; ++t) for (int level = 0; level < 4; ++level) cell(level, 0, t) = counts[t] > level;
    for (int tile : hands_[player]) if (rules_.red_fives && red_id(tile)) cell(4, 0, tile / 4) = 1.0F;
    if (drawn_tiles_[player] >= 0) cell(5, 0, drawn_tiles_[player] / 4) = 1.0F;
    const auto places = placement();
    for (int row = 0; row < 4; ++row) {
        const int p = (player + row) % 4;
        const auto& discards = discards_[p];
        for (const auto& d : discards) {
            const int t = d.tile / 4;
            cell(6, row, t) += 0.25F; cell(7, row, t) += 0.25F * d.called;
            if (d.riichi) cell(91, row, t) = 1.0F;
            if (rules_.red_fives && red_id(d.tile)) cell(97, row, t) = 1.0F;
        }
        const int size = static_cast<int>(discards.size());
        const int begin = std::max(0, size - 32);
        for (int i = begin; i < size; ++i) {
            const int index = 32 - size + i;
            cell(8 + index, row, discards[i].tile / 4) = 1.0F;
            cell(40 + index, row, discards[i].tile / 4) = discards[i].tsumogiri;
        }
        for (int i = 0; i < static_cast<int>(melds_[p].size()); ++i) {
            const auto& m = melds_[p][i];
            for (int physical : m.tiles) {
                const int t = physical / 4;
                cell(72 + i, row, t) += 0.25F; cell(76 + i, row, t) = m.open; cell(80 + i, row, t) = m.kind == "kan";
                if (rules_.red_fives && red_id(physical)) cell(84 + i, row, t) = 1.0F;
            }
        }
        row_constant(88, row, riichi_[p]); row_constant(89, row, double_riichi_[p]); row_constant(90, row, ippatsu_[p]);
        row_constant(99, row, scores_[p] / 100000.0); row_constant(100, row, places[p] / 4.0);
        for (int i = 0; i < dora_count_; ++i) cell(92 + i, row, dead_wall_[4 + 2 * i] / 4) = 1.0F;
        cell(101, row, 27 + std::min(round_index_ / 4, 3)) = 1.0F;
        cell(102, row, 27 + (p - dealer_ + 4) % 4) = 1.0F;
    }
    constant(98, live_wall_.size() / 70.0); constant(103, honba_ / 10.0); constant(104, riichi_sticks_ / 10.0);
    const auto flags = furiten(player);
    for (int i = 0; i < 3; ++i) row_constant(105 + i, 0, flags[i]);
    row_constant(108, 0, player_shanten(player) / 8.0);
    for (int t : player_waits(player)) cell(109, 0, t) = 1.0F;
    row_constant(110, (dealer_ - player + 4) % 4, 1); row_constant(111, (turn_ - player + 4) % 4, 1);
    constant(112 + phase_, 1);
    if (last_discard_) cell(117, (last_discard_->first - player + 4) % 4, last_discard_->second / 4) = 1.0F;
    if (pending_kan_) cell(118, (pending_kan_->player - player + 4) % 4, pending_kan_->tile) = 1.0F;
    constant(119, 1);
}
void Game::legal_mask(int player, std::span<float> output) const {
    validate_player(player);
    if (output.size() != RQ_ACTION_SIZE) throw std::invalid_argument("invalid mask size");
    const auto legal = legal_actions(player);
    std::fill(output.begin(), output.end(), 0.0F);
    for (int action : legal) output[action] = 1.0F;
}
}
