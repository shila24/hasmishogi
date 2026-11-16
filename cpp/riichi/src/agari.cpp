#include "riichi/agari.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>
namespace riichi {
Meld::Meld(std::vector<int> values, std::string name, bool opened)
    : tiles(std::move(values)), kind(std::move(name)), open(opened) {
    std::sort(tiles.begin(), tiles.end()); validate();
}
void Meld::validate() const {
    for (int t : tiles) if (t < 0 || t >= 34) throw std::invalid_argument("invalid meld tiles or open flag");
    bool valid = false;
    if (kind == "chi") valid = tiles.size()==3 && tiles[0]<27 && tiles[0]/9==tiles[2]/9 && tiles[1]==tiles[0]+1 && tiles[2]==tiles[0]+2 && open;
    if (kind == "pon" || kind == "kan") valid = tiles.size()==(kind=="kan" ? 4U : 3U) &&
        std::all_of(tiles.begin(),tiles.end(),[&](int t){return t==tiles[0];}) && (open || kind=="kan");
    if (!valid) throw std::invalid_argument("invalid declared meld");
}
bool Decomposition::menzen() const {
    return std::none_of(groups.begin(), groups.end(), [](const Group& g){ return g.open; });
}
std::string Decomposition::wait() const {
    if (kind=="kokushi") return "kokushi";
    if (kind=="chiitoitsu" || winning_group == -1) return "tanki";
    if (!winning_group) return "unknown";
    const auto& g = groups.at(*winning_group);
    if (!g.sequence()) return "shanpon";
    const int low = g.tiles[0];
    if (winning_tile==low+1) return "kanchan";
    if ((low%9==0 && winning_tile==low+2) || (low%9==6 && winning_tile==low)) return "penchan";
    return "ryanmen";
}
Counts hand_context(const Counts& counts, const std::vector<Meld>& melds) {
    validate_counts(counts);
    if (melds.size()>4) throw std::invalid_argument("expected at most four Meld objects");
    Counts total=counts;
    for (const auto& m : melds) { m.validate(); for (int t : m.tiles) ++total[t]; }
    validate_counts(total);
    if (std::accumulate(counts.begin(),counts.end(),0)>14-3*static_cast<int>(melds.size()))
        throw std::invalid_argument("too many concealed tiles for meld context");
    return total;
}
namespace {
using Partition = std::vector<std::vector<int>>;
void sets(Counts& c, Partition& prefix, std::vector<Partition>& result) {
    int t=0; while (t<34 && !c[t]) ++t;
    if (t==34) { auto sorted=prefix; std::sort(sorted.begin(),sorted.end());
        if (std::find(result.begin(),result.end(),sorted)==result.end()) result.push_back(std::move(sorted));
        return;
    }
    if (c[t]>=3) { c[t]-=3; prefix.push_back({t,t,t}); sets(c,prefix,result); prefix.pop_back(); c[t]+=3; }
    if (t<27 && t%9<=6 && c[t+1] && c[t+2]) {
        --c[t]; --c[t+1]; --c[t+2]; prefix.push_back({t,t+1,t+2}); sets(c,prefix,result);
        prefix.pop_back(); ++c[t]; ++c[t+1]; ++c[t+2];
    }
}
}
std::vector<Decomposition> decompose(const Counts& c, const std::vector<Meld>& melds, std::optional<int> win) {
    hand_context(c,melds);
    if (win && (*win<0 || *win>=34 || !c[*win])) throw std::invalid_argument("winning tile must occur in the concealed hand");
    if (std::accumulate(c.begin(),c.end(),0)!=14-3*static_cast<int>(melds.size())) return {};
    std::vector<Decomposition> result;
    if (melds.empty()) {
        if (std::count(c.begin(),c.end(),2)==7) {
            Decomposition d{"chiitoitsu"}; d.winning_tile=win;
            for(int t=0;t<34;++t) if(c[t]==2) d.special_pairs.push_back(t);
            result.push_back(d);
        }
        int unique=0,total=0,pair=-1;
        for(int t=0;t<34;++t) { if(orphan(t)) { unique+=c[t]>0; total+=c[t]; } if(c[t]==2) pair=t; }
        if(unique==13 && total==14) { Decomposition d{"kokushi"}; d.pair=pair; d.winning_tile=win; result.push_back(d); }
    }
    for(int pair=0;pair<34;++pair) if(c[pair]>=2) {
        auto work=c; work[pair]-=2; Partition prefix; std::vector<Partition> partitions; sets(work,prefix,partitions);
        for(const auto& partition:partitions) {
            Decomposition d{"standard"}; d.pair=pair; d.winning_tile=win;
            for(const auto& g:partition) d.groups.push_back({g,false,false});
            for(const auto& m:melds) d.groups.push_back({m.tiles,m.open,true});
            if(!win) { result.push_back(d); continue; }
            if(pair==win) { d.winning_group=-1; result.push_back(d); }
            std::set<std::vector<int>> seen;
            for(std::size_t i=0;i<partition.size();++i) {
                const auto& g=partition[i];
                if(std::find(g.begin(),g.end(),*win)!=g.end() && seen.insert(g).second) {
                    d.winning_group=static_cast<int>(i); result.push_back(d);
                }
            }
        }
    }
    return result;
}
bool is_agari(const Counts& c,const std::vector<Meld>& m) { return !decompose(c,m).empty(); }
}
