// SPDX-License-Identifier: MIT
// Generated from hasami_learn/nnue/layout.py by features.generated_header().
#ifndef SHOGI_SEARCH_NNUE_LAYOUT_HPP
#define SHOGI_SEARCH_NNUE_LAYOUT_HPP
#include <array>
namespace shogi_search::nnue {
inline constexpr char feature_set[] = "hlnn-kp-v1";
inline constexpr std::array<int, 13> board_types{1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14};
inline constexpr std::array<int, 7> hand_limits{18, 4, 4, 4, 4, 2, 2};
inline constexpr int max_active = 39;
inline constexpr int king_stride = 2263;
inline constexpr int feature_count = 183303;
inline constexpr float output_scale = 600.0F;
}
#endif
