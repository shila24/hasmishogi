#include "riichi/shanten.hpp"
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_map>
namespace riichi {
namespace {
using Suit = std::array<int,9>;
using Frontier = std::array<std::array<int,2>,5>;
Frontier empty_frontier(int value) {
    Frontier f{}; for(auto& row:f) row.fill(value); return f;
}
std::uint32_t suit_key(const Suit& c, bool honors) {
    std::uint32_t key=honors;
    for(int n:c) key=key*5+static_cast<unsigned>(n);
    return key;
}
Frontier suit_table(Suit c, bool honors) {
    // Thread-local bounded caches keep independent ABI callers race-free.
    thread_local std::unordered_map<std::uint32_t,Frontier> cache;
    const auto key=suit_key(c,honors);
    if(auto it=cache.find(key);it!=cache.end()) return it->second;
    int first=0; while(first<9 && !c[first]) ++first;
    auto result=empty_frontier(-1);
    if(first==9) { result[0][0]=0; return result; }
    --c[first]; result=suit_table(c,honors);
    const auto take=[&](std::initializer_list<int> tiles,int sets,int partial,int head) {
        for(int t:tiles) --c[t];
        const auto tail=suit_table(c,honors);
        for(int t:tiles) ++c[t];
        for(int m=0;m+sets<=4;++m) for(int p=0;p+head<=1;++p) if(tail[m][p]>=0)
            result[m+sets][p+head]=std::max(result[m+sets][p+head],std::min(4,tail[m][p]+partial));
    };
    if(c[first]>=2) take({first,first},1,0,0);
    if(c[first]) { take({first},0,1,0); take({first},0,0,1); }
    if(!honors) {
        if(first<=6 && c[first+1] && c[first+2]) take({first+1,first+2},1,0,0);
        for(int d:{1,2}) if(first+d<9 && c[first+d]) take({first+d},0,1,0);
    }
    if(cache.size()>=200000) cache.clear();
    cache.emplace(key,result); return result;
}
int structural(const Counts& c,int meld_count) {
    auto f=empty_frontier(-1); f[meld_count][0]=0;
    for(int start:{0,9,18,27}) {
        Suit suit{}; std::copy_n(c.begin()+start,start==27?7:9,suit.begin());
        const auto table=suit_table(suit,start==27); auto next=empty_frontier(-1);
        for(int m=0;m<=4;++m) for(int p=0;p<=1;++p) if(f[m][p]>=0)
            for(int sm=0;m+sm<=4;++sm) for(int sp=0;p+sp<=1;++sp) if(table[sm][sp]>=0)
                next[m+sm][p+sp]=std::max(next[m+sm][p+sp],std::min(4-m-sm,f[m][p]+table[sm][sp]));
        f=next;
    }
    int best=99;
    for(int m=0;m<=4;++m) for(int p=0;p<=1;++p) if(f[m][p]>=0) best=std::min(best,8-2*m-f[m][p]-p);
    return best;
}
using Targets=std::array<std::array<std::vector<Suit>,2>,5>;
Targets make_targets(bool honors) {
    Targets targets{}; const int size=honors?7:9;
    std::vector<std::array<int,3>> groups;
    for(int t=0;t<size;++t) groups.push_back({t,t,t});
    if(!honors) for(int t=0;t<7;++t) groups.push_back({t,t+1,t+2});
    for(int sets=0;sets<=4;++sets) {
        std::array<std::set<Suit>,2> patterns;
        Suit c{};
        std::function<void(int,int)> visit=[&](int begin,int left) {
            if(!left) {
                patterns[0].insert(c);
                for(int t=0;t<size;++t) if(c[t]<=2) { c[t]+=2; patterns[1].insert(c); c[t]-=2; }
                return;
            }
            for(int i=begin;i<static_cast<int>(groups.size());++i) {
                for(int t:groups[i]) ++c[t];
                if(*std::max_element(c.begin(),c.end())<=4) visit(i,left-1);
                for(int t:groups[i]) --c[t];
            }
        };
        visit(0,sets);
        for(int p=0;p<2;++p) targets[sets][p].assign(patterns[p].begin(),patterns[p].end());
    }
    return targets;
}
Frontier bounded_costs(const Suit& c,const Suit& cap,bool honors,int maximum) {
    static const Targets suited=make_targets(false), honor=make_targets(true);
    const auto& targets=honors?honor:suited;
    thread_local std::map<std::tuple<Suit,Suit,bool,int>,Frontier> cache;
    const auto key=std::make_tuple(c,cap,honors,maximum);
    if(auto it=cache.find(key);it!=cache.end()) return it->second;
    auto costs=empty_frontier(99); const int count=std::accumulate(c.begin(),c.end(),0);
    for(int m=0;m<=maximum;++m) for(int p=0;p<2;++p) {
        int best=99;
        for(const auto& target:targets[m][p]) {
            int cost=0;
            for(int t=0;t<9;++t) {
                if(target[t]>cap[t]) { cost=99; break; }
                cost+=std::max(0,target[t]-c[t]);
                if(cost>=best) break;
            }
            best=std::min(best,cost);
            if(best==std::max(0,3*m+2*p-count)) break;
        }
        costs[m][p]=best;
    }
    if(cache.size()>=16384) cache.clear();
    cache.emplace(key,costs); return costs;
}
int bounded(const Counts& c,const Counts& total,int meld_count) {
    const int maximum=4-meld_count;
    auto f=empty_frontier(99); f[0][0]=0;
    for(int start:{0,9,18,27}) {
        Suit suit{},cap{};
        for(int i=0;i<(start==27?7:9);++i) { suit[i]=c[start+i]; cap[i]=4-total[start+i]+c[start+i]; }
        const auto costs=bounded_costs(suit,cap,start==27,maximum); auto next=empty_frontier(99);
        for(int m=0;m<=maximum;++m) for(int p=0;p<2;++p)
            for(int sm=0;m+sm<=maximum;++sm) for(int sp=0;p+sp<2;++sp)
                next[m+sm][p+sp]=std::min(next[m+sm][p+sp],f[m][p]+costs[sm][sp]);
        f=next;
    }
    return f[maximum][1]-1;
}
}
std::array<int,4> shanten_all(const Counts& c,const std::vector<Meld>& melds) {
    const auto total=hand_context(c,melds); const int n=static_cast<int>(melds.size());
    const int count=std::accumulate(c.begin(),c.end(),0);
    const bool capacity=(!melds.empty() || std::find(total.begin(),total.end(),4)!=total.end()) && (count==13-3*n || count==14-3*n);
    const int standard=capacity?bounded(c,total,n):structural(c,n);
    int chiitoi=99,kokushi=99;
    if(!n) {
        int pairs=0,unique=0,orphans=0; bool orphan_pair=false;
        for(int t=0;t<34;++t) { pairs+=c[t]>=2; unique+=c[t]>0; if(orphan(t)) { orphans+=c[t]>0; orphan_pair|=c[t]>=2; } }
        chiitoi=6-pairs+std::max(0,7-unique); kokushi=13-orphans-orphan_pair;
    }
    return {std::min({standard,chiitoi,kokushi}),standard,chiitoi,kokushi};
}
int shanten(const Counts& c,const std::vector<Meld>& m) { return shanten_all(c,m)[0]; }
int standard_shanten(const Counts& c,const std::vector<Meld>& m) { return shanten_all(c,m)[1]; }
int chiitoitsu_shanten(const Counts& c,const std::vector<Meld>& m) { return shanten_all(c,m)[2]; }
int kokushi_shanten(const Counts& c,const std::vector<Meld>& m) { return shanten_all(c,m)[3]; }
std::vector<int> waits(const Counts& c,const std::vector<Meld>& m) {
    const auto total=hand_context(c,m);
    if(std::accumulate(c.begin(),c.end(),0)!=13-3*static_cast<int>(m.size())) throw std::invalid_argument("waits requires a hand before the winning draw");
    std::vector<int> result;
    for(int t=0;t<34;++t) if(total[t]<4) { auto work=c; ++work[t]; if(is_agari(work,m)) result.push_back(t); }
    return result;
}
}
