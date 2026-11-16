#include "riichi/yaku.hpp"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
namespace riichi {
void WinContext::validate() const {
    if(seat_wind<27 || seat_wind>30 || round_wind<27 || round_wind>30) throw std::invalid_argument("winds must be tile types 27..30");
    for(const auto* values:{&dora_indicators,&ura_indicators}) {
        if(values->size()>5) throw std::invalid_argument("at most five indicators");
        for(int t:*values) dora_type(t);
    }
    if(red_dora<0 || red_dora>3) throw std::invalid_argument("red_dora must be an integer in 0..3");
    if(ippatsu && !(riichi || double_riichi)) throw std::invalid_argument("ippatsu requires riichi");
    if((rinshan || haitei || tenhou || chiihou) && !tsumo) throw std::invalid_argument("tsumo-only context on ron");
    if((chankan || houtei || renhou) && tsumo) throw std::invalid_argument("ron-only context on tsumo");
    if(rinshan && ippatsu) throw std::invalid_argument("kan cancels ippatsu before a rinshan draw");
    if((rinshan && haitei) || (chankan && houtei)) throw std::invalid_argument("last-wall and kan events are mutually exclusive");
    if(static_cast<int>(tenhou)+chiihou+renhou>1) throw std::invalid_argument("first-turn wins are mutually exclusive");
    if((tenhou && seat_wind!=27) || ((chiihou || renhou) && seat_wind==27)) throw std::invalid_argument("first-turn win has wrong seat wind");
    if((tenhou || chiihou || renhou) && (riichi || double_riichi || ippatsu || rinshan || chankan || haitei || houtei)) throw std::invalid_argument("inconsistent first-turn context");
}
int pair_fu(std::optional<int> pair,const WinContext& c,const Rules& r) {
    if(!pair) return 0;
    if(*pair>=31) return 2;
    const int values=(*pair==c.seat_wind)+(*pair==c.round_wind);
    return values==2?r.double_wind_pair_fu:values*2;
}
bool concealed_triplet(const Group& g,int index,const Decomposition& d,const WinContext& c) {
    return !g.sequence() && !g.open && (c.tsumo || d.winning_group!=index);
}
void validate_context(const Decomposition& d,const WinContext& c,const Rules& r) {
    if(!d.menzen() && (c.riichi || c.double_riichi || c.ippatsu)) throw std::invalid_argument("riichi requires a closed hand");
    if((c.tenhou || c.chiihou || c.renhou) && std::any_of(d.groups.begin(),d.groups.end(),[](const Group& g){return g.declared;})) throw std::invalid_argument("first-turn win cannot contain declared melds");
    if(c.rinshan && std::none_of(d.groups.begin(),d.groups.end(),[](const Group& g){return g.kan();})) throw std::invalid_argument("rinshan requires a kan");
    if(!r.atozuke && !d.menzen() && !c.yaku_established) throw std::invalid_argument("atozuke-disabled open scoring requires yaku_established history");
}
std::vector<Yaku> detect_yaku(const Decomposition& d,const Counts& counts,const WinContext& c,const Rules& r) {
    validate_context(d,c,r);
    if(!d.winning_tile) throw std::invalid_argument("yaku evaluation requires a winning-tile assignment");
    std::set<int> tiles,suits,triplets; std::vector<int> sequences;
    int concealed=0,kans=0;
    for(int t=0;t<34;++t) if(counts[t]) { tiles.insert(t); if(t<27) suits.insert(t/9); }
    for(std::size_t i=0;i<d.groups.size();++i) {
        const auto& g=d.groups[i];
        if(g.sequence()) sequences.push_back(g.tiles[0]); else triplets.insert(g.tiles[0]);
        concealed+=concealed_triplet(g,static_cast<int>(i),d,c); kans+=g.kan();
    }
    const auto all_tiles=[&](auto predicate){return std::all_of(tiles.begin(),tiles.end(),predicate);};
    const bool honors=std::any_of(tiles.begin(),tiles.end(),[](int t){return t>=27;});
    const bool has_orphans=std::any_of(tiles.begin(),tiles.end(),orphan);
    const auto seq=[&](int t){return std::find(sequences.begin(),sequences.end(),t)!=sequences.end();};
    std::vector<Yaku> y;
    const auto yak=[&](const char* name,bool condition,bool twice=false){if(condition) y.push_back({name,0,twice && r.double_yakuman?2:1});};
    yak("tenhou",c.tenhou); yak("chiihou",c.chiihou); yak("renhou",c.renhou && r.renhou==2);
    if(d.kind=="kokushi") { const bool thirteen=counts[*d.winning_tile]==2; yak(thirteen?"kokushi_13":"kokushi",true,thirteen); }
    yak("daisangen",triplets.contains(31) && triplets.contains(32) && triplets.contains(33));
    int winds=0; for(int t=27;t<=30;++t) winds+=triplets.contains(t);
    yak("daisuushii",winds==4,true); yak("shousuushii",winds==3 && d.pair && *d.pair>=27 && *d.pair<=30);
    yak("tsuuiisou",!tiles.empty() && all_tiles([](int t){return t>=27;}));
    yak("chinroutou",!tiles.empty() && all_tiles(terminal));
    yak("ryuuiisou",!tiles.empty() && all_tiles([](int t){return t==19 || t==20 || t==21 || t==23 || t==25 || t==32;}));
    yak("suukantsu",kans==4);
    if(concealed==4) { const bool tanki=d.wait()=="tanki"; yak(tanki?"suuankou_tanki":"suuankou",true,tanki); }
    const bool declared=std::any_of(d.groups.begin(),d.groups.end(),[](const Group& g){return g.declared;});
    if(d.menzen() && !declared && suits.size()==1 && !honors) {
        const int offset=*suits.begin()*9; const std::array<int,9> base{3,1,1,1,1,1,1,1,3}; bool shape=true,pure=true;
        for(int i=0;i<9;++i) { shape &= counts[offset+i]>=base[i]; pure &= counts[offset+i]-(*d.winning_tile%9==i)==base[i]; }
        if(shape) yak(pure?"junsei_chuuren":"chuuren",true,pure);
    }
    if(!y.empty()) return y;
    if(!r.atozuke && !d.menzen() && c.yaku_established==false) return {};
    const auto add=[&](const char* name,bool condition,int closed,int opened=-1) {
        const int han=d.menzen() || opened==-1?closed:opened;
        if(condition && han) y.push_back({name,han,0});
    };
    add(c.double_riichi?"double_riichi":"riichi",c.riichi || c.double_riichi,c.double_riichi?2:1,0);
    add("ippatsu",c.ippatsu && r.ippatsu,1,0); add("menzen_tsumo",c.tsumo,1,0);
    add("rinshan",c.rinshan,1); add("chankan",c.chankan,1); add("haitei",c.haitei,1); add("houtei",c.houtei,1);
    add("tanyao",!has_orphans && (d.menzen() || r.kuitan),1); add("chiitoitsu",d.kind=="chiitoitsu",2,0);
    add("honroutou",all_tiles(orphan),2);
    if(suits.size()==1) add(honors?"honitsu":"chinitsu",true,honors?3:6,honors?2:5);
    if(d.kind=="standard") {
        add("pinfu",sequences.size()==4 && !pair_fu(d.pair,c,r) && d.wait()=="ryanmen",1,0);
        std::map<int,int> frequency; for(int t:sequences) ++frequency[t]; int repeated=0;
        for(auto [t,n]:frequency) { (void)t; repeated+=n/2; }
        add(repeated==2?"ryanpeikou":"iipeikou",repeated>0,repeated==2?3:1,0);
        add("toitoi",sequences.empty(),2); add("sanankou",concealed>=3,2); add("sankantsu",kans==3,2);
        const int dragons=static_cast<int>(triplets.contains(31))+triplets.contains(32)+triplets.contains(33);
        add("shousangen",dragons==2 && d.pair && *d.pair>=31,2);
        add("haku",triplets.contains(31),1); add("hatsu",triplets.contains(32),1); add("chun",triplets.contains(33),1);
        add("seat_wind",triplets.contains(c.seat_wind),1); add("round_wind",triplets.contains(c.round_wind),1);
        bool doujun=false,doukou=false,ittsuu=false;
        for(int n=0;n<7;++n) doujun |= seq(n) && seq(n+9) && seq(n+18);
        for(int n=0;n<9;++n) doukou |= triplets.contains(n) && triplets.contains(n+9) && triplets.contains(n+18);
        for(int s=0;s<3;++s) ittsuu |= seq(s*9) && seq(s*9+3) && seq(s*9+6);
        add("sanshoku_doujun",doujun,2,1); add("sanshoku_doukou",doukou,2); add("ittsuu",ittsuu,2,1);
        const bool outside=d.pair && orphan(*d.pair) && std::all_of(d.groups.begin(),d.groups.end(),[](const Group& g){return std::any_of(g.tiles.begin(),g.tiles.end(),orphan);});
        add(honors?"chanta":"junchan",outside && !sequences.empty(),honors?2:3,honors?1:2);
    }
    if(y.empty() && !(c.renhou && r.renhou==1)) return {};
    const bool ordinary=!y.empty();
    const auto bonus=[&](const std::vector<int>& indicators) {
        int n=0; for(std::size_t i=0;i<indicators.size() && (r.kan_dora || i==0);++i) n+=counts[dora_type(indicators[i])]; return n;
    };
    add("dora",true,bonus(c.dora_indicators)); add("aka_dora",r.red_fives,c.red_dora);
    add("ura_dora",r.ura_dora && (c.riichi || c.double_riichi),bonus(c.ura_indicators));
    int han=0; for(const auto& item:y) han+=item.han;
    if(c.renhou && r.renhou==1 && (!ordinary || han<5)) return {{"renhou",5,0}};
    return y;
}
bool nagashi_eligible(std::span<const int> discards,bool called,bool opened,const Rules& rules) {
    for(int t:discards) if(t<0 || t>=34) throw std::invalid_argument("discards must be tile types");
    return rules.nagashi_mangan && !discards.empty() && !called && !opened && std::all_of(discards.begin(),discards.end(),orphan);
}
}
