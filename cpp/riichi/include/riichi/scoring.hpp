#pragma once
#include "riichi/yaku.hpp"
#include <cstdint>
#include <stdexcept>
namespace riichi {
class NoYakuError : public std::invalid_argument { public: using std::invalid_argument::invalid_argument; };
struct Payment {
    std::int64_t ron=0, tsumo_dealer=0, tsumo_non_dealer=0, riichi_bonus=0, total=0;
};
struct Score {
    int han=0, fu=0, yakuman=0;
    std::int64_t base_points=0;
    std::vector<Yaku> yaku{};
    Payment payment{};
    Decomposition decomposition{};
};
int calculate_fu(const Decomposition& d,const WinContext& context,const Rules& rules = {});
std::int64_t base_points(int han,int fu,int yakuman=0,const Rules& rules = {});
Payment payments(std::int64_t base,bool dealer=false,bool tsumo=false,int honba=0,int riichi_sticks=0,const Rules& rules = {});
Score score_hand(const Counts& counts,int winning_tile,const std::vector<Meld>& melds = {},
                 WinContext context = {},const Rules& rules = {},int honba=0,int riichi_sticks=0,
                 std::optional<std::vector<int>> tile_ids = {});
std::array<std::int64_t,4> tenpai_payments(const std::array<bool,4>& tenpai,const Rules& rules = {});
std::array<double,4> placement_points(const std::array<int,4>& scores,const Rules& rules = {},std::array<int,4> tie_order = {0,1,2,3});
std::array<std::int64_t,4> settle_payment(const Payment& payment,int winner,int dealer,std::optional<int> discarder = {});
std::string score_json(const Score& score);
}
