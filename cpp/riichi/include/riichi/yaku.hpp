#pragma once
#include "riichi/agari.hpp"
#include "riichi/rules.hpp"
namespace riichi {
struct WinContext {
    bool tsumo=false;
    int seat_wind=28, round_wind=27;
    bool riichi=false, double_riichi=false, ippatsu=false, rinshan=false, chankan=false;
    bool haitei=false, houtei=false, tenhou=false, chiihou=false, renhou=false;
    std::vector<int> dora_indicators{}, ura_indicators{};
    int red_dora=0;
    std::optional<bool> yaku_established{};
    bool dealer() const { return seat_wind==27; }
    void validate() const;
};
struct Yaku { std::string name; int han=0, yakuman=0; };
int pair_fu(std::optional<int> pair,const WinContext& context,const Rules& rules);
bool concealed_triplet(const Group& g,int index,const Decomposition& d,const WinContext& context);
void validate_context(const Decomposition& d,const WinContext& context,const Rules& rules);
std::vector<Yaku> detect_yaku(const Decomposition& d,const Counts& total,const WinContext& context,const Rules& rules = {});
bool nagashi_eligible(std::span<const int> discards,bool any_discard_called=false,bool has_open_meld=false,const Rules& rules = {});
}
