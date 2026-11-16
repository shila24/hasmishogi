#include "riichi/scoring.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <tuple>
#include <utility>
namespace riichi {
namespace {
void nonnegative(std::int64_t n,const char* name) {
    if(n<0) throw std::invalid_argument(std::string(name)+" must be a nonnegative integer");
}
std::int64_t checked_add(std::int64_t a,std::int64_t b) {
    if(b>std::numeric_limits<std::int64_t>::max()-a) throw std::invalid_argument("payment exceeds native integer range");
    return a+b;
}
}
int calculate_fu(const Decomposition& d,const WinContext& c,const Rules& r) {
    if(d.kind=="chiitoitsu") return 25;
    if(d.kind=="kokushi") return 0;
    if(!d.winning_tile) throw std::invalid_argument("fu requires a winning-tile assignment");
    const auto wait=d.wait();
    if(d.menzen() && std::all_of(d.groups.begin(),d.groups.end(),[](const Group& g){return g.sequence();}) && !pair_fu(d.pair,c,r) && wait=="ryanmen") return c.tsumo?20:30;
    int fu=20+pair_fu(d.pair,c,r);
    if(c.tsumo) fu+=2; else if(d.menzen()) fu+=10;
    if(wait=="tanki" || wait=="kanchan" || wait=="penchan") fu+=2;
    for(std::size_t i=0;i<d.groups.size();++i) {
        const auto& g=d.groups[i]; if(g.sequence()) continue;
        int value=g.kan()?8:2;
        if(concealed_triplet(g,static_cast<int>(i),d,c)) value*=2;
        if(orphan(g.tiles[0])) value*=2;
        fu+=value;
    }
    return std::max(30,(fu+9)/10*10);
}
std::int64_t base_points(int han,int fu,int yakuman,const Rules& r) {
    nonnegative(han,"han"); nonnegative(fu,"fu"); nonnegative(yakuman,"yakuman");
    if(yakuman) return 8000LL*yakuman;
    if(han<1 || !(fu==20 || fu==25 || (fu>=30 && fu<=160 && fu%10==0))) throw std::invalid_argument("positive han and standard fu required");
    if(han>=13 && r.kazoe_yakuman) return 8000;
    if(han>=11) return 6000;
    if(han>=8) return 4000;
    if(han>=6) return 3000;
    if(han>=5 || (r.kiriage_mangan && ((han==4 && fu==30) || (han==3 && fu==60)))) return 2000;
    return std::min(2000,fu*(1<<(han+2)));
}
Payment payments(std::int64_t base,bool dealer,bool tsumo,int honba,int sticks,const Rules& r) {
    nonnegative(base,"base"); nonnegative(honba,"honba"); nonnegative(sticks,"riichi_sticks");
    if(!base) throw std::invalid_argument("positive base and boolean dealer/tsumo required");
    if(base>(std::numeric_limits<std::int64_t>::max()-99)/6) throw std::invalid_argument("base exceeds native integer range");
    const std::int64_t bonus=static_cast<std::int64_t>(r.riichi_stick)*sticks;
    const std::int64_t extra=static_cast<std::int64_t>(honba)*r.honba;
    const auto rounded=[](std::int64_t n){return (n+99)/100*100;};
    Payment pay; pay.riichi_bonus=bonus;
    if(!tsumo) { pay.ron=checked_add(rounded(base*(dealer?6:4)),extra); pay.total=checked_add(pay.ron,bonus); }
    else {
        pay.tsumo_non_dealer=checked_add(rounded(base*(dealer?2:1)),extra/3);
        pay.tsumo_dealer=dealer?0:checked_add(rounded(base*2),extra/3);
        auto sum=checked_add(pay.tsumo_non_dealer,pay.tsumo_non_dealer);
        sum=checked_add(sum,dealer?pay.tsumo_non_dealer:pay.tsumo_dealer);
        pay.total=checked_add(sum,bonus);
    }
    return pay;
}
Score score_hand(const Counts& c,int win,const std::vector<Meld>& melds,WinContext context,const Rules& r,int honba,int sticks,std::optional<std::vector<int>> ids) {
    r.validate(); context.validate(); const auto total=hand_context(c,melds);
    if(ids) {
        if(to_counts(*ids)!=total) throw std::invalid_argument("tile_ids do not match the hand and melds");
        context.red_dora=r.red_fives?static_cast<int>(std::count_if(ids->begin(),ids->end(),red_id)):0;
    }
    if(context.red_dora>static_cast<int>(total[4]>0)+(total[13]>0)+(total[22]>0)) throw std::invalid_argument("red_dora exceeds available five types");
    const auto decompositions=decompose(c,melds,win);
    if(decompositions.empty()) throw std::invalid_argument("hand is not complete");
    std::optional<Score> best;
    const auto key=[](const Score& s){return std::make_tuple(s.payment.total,s.yakuman,s.han,s.fu);};
    for(const auto& d:decompositions) {
        auto y=detect_yaku(d,total,context,r); if(y.empty()) continue;
        int han=0,units=0,maximum=0;
        for(const auto& item:y) { han+=item.han; units+=item.yakuman; maximum=std::max(maximum,item.yakuman); }
        if(units && !r.multiple_yakuman) units=maximum;
        const int fu=calculate_fu(d,context,r); const auto base=base_points(han,fu,units,r);
        Score score{han,fu,units,base,std::move(y),payments(base,context.dealer(),context.tsumo,honba,sticks,r),d};
        if(!best || key(score)>key(*best)) best=std::move(score);
    }
    if(!best) throw NoYakuError("no yaku; dora alone cannot qualify a hand");
    return *best;
}
std::array<std::int64_t,4> tenpai_payments(const std::array<bool,4>& tenpai,const Rules& r) {
    const int n=std::count(tenpai.begin(),tenpai.end(),true); std::array<std::int64_t,4> result{};
    if(n && n<4) for(int i=0;i<4;++i) result[i]=tenpai[i]?r.tenpai_payment/n:-r.tenpai_payment/(4-n);
    return result;
}
std::array<double,4> placement_points(const std::array<int,4>& scores,const Rules& r,std::array<int,4> order) {
    auto sorted=order; std::sort(sorted.begin(),sorted.end());
    if(sorted!=std::array<int,4>{0,1,2,3}) throw std::invalid_argument("tie_order must be a seat permutation");
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b];});
    std::array<double,4> result{};
    for(int rank=0;rank<4;++rank) result[order[rank]]=(static_cast<std::int64_t>(scores[order[rank]])-r.return_points)/1000.0+r.uma[rank];
    result[order[0]]+=4LL*(static_cast<std::int64_t>(r.return_points)-r.start_points)/1000.0;
    return result;
}
std::array<std::int64_t,4> settle_payment(const Payment& p,int winner,int dealer,std::optional<int> discarder) {
    if(winner<0 || winner>=4 || dealer<0 || dealer>=4) throw std::invalid_argument("invalid seat");
    std::array<std::int64_t,4> delta{};
    if(p.ron) {
        if(!discarder || *discarder<0 || *discarder>=4 || *discarder==winner) throw std::invalid_argument("ron requires another discarder seat");
        delta[*discarder]=-p.ron;
    } else {
        if(discarder) throw std::invalid_argument("tsumo has no discarder");
        if((winner==dealer)!=(p.tsumo_dealer==0)) throw std::invalid_argument("payment dealer status does not match seats");
        for(int seat=0;seat<4;++seat) if(seat!=winner) delta[seat]=-(seat==dealer?p.tsumo_dealer:p.tsumo_non_dealer);
    }
    delta[winner]=p.total; return delta;
}
std::string score_json(const Score& s) {
    std::ostringstream out;
    out << "{\"han\":" << s.han << ",\"fu\":" << s.fu << ",\"yakuman\":" << s.yakuman << ",\"base_points\":" << s.base_points
        << ",\"payment\":{\"ron\":" << s.payment.ron << ",\"tsumo_dealer\":" << s.payment.tsumo_dealer
        << ",\"tsumo_non_dealer\":" << s.payment.tsumo_non_dealer << ",\"riichi_bonus\":" << s.payment.riichi_bonus
        << ",\"total\":" << s.payment.total << "},\"yaku\":[";
    bool first=true;
    for(const auto& y:s.yaku) { if(!first) out << ','; first=false; out << "[\"" << y.name << "\"," << y.han << ',' << y.yakuman << ']'; }
    out << "]}"; return out.str();
}
}
