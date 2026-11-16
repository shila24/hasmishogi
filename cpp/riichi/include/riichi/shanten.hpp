#pragma once
#include "riichi/agari.hpp"
namespace riichi {
// Results are overall, standard, seven pairs, thirteen orphans, in ABI order.
std::array<int,4> shanten_all(const Counts& counts, const std::vector<Meld>& melds = {});
int shanten(const Counts& counts, const std::vector<Meld>& melds = {});
int standard_shanten(const Counts& counts, const std::vector<Meld>& melds = {});
int chiitoitsu_shanten(const Counts& counts, const std::vector<Meld>& melds = {});
int kokushi_shanten(const Counts& counts, const std::vector<Meld>& melds = {});
std::vector<int> waits(const Counts& counts, const std::vector<Meld>& melds = {});
}
