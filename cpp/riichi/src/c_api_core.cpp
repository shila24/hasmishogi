#include "riichi/core.hpp"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <exception>
namespace {
// Fixed storage ensures even allocation failures can be reported without throwing.
thread_local char last_error[512]{};
using riichi::abi::guarded;
void require(bool condition,const char* message) { if(!condition) throw std::invalid_argument(message); }
riichi::Counts counts_from(const int32_t* ptr) {
    require(ptr,"counts must not be NULL"); riichi::Counts result{}; std::copy_n(ptr,34,result.begin()); return result;
}
std::vector<riichi::Meld> melds_from(const rq_meld* ptr,int n) {
    require(n>=0 && n<=4,"expected at most four Meld objects"); require(ptr || n==0,"melds must not be NULL");
    std::vector<riichi::Meld> result;
    for(int i=0;i<n;++i) {
        const auto& m=ptr[i]; require(m.kind>=0 && m.kind<=2,"invalid declared meld"); require(m.open==0 || m.open==1,"invalid meld open flag");
        result.emplace_back(std::vector<int>(m.tiles,m.tiles+(m.kind==2?4:3)),m.kind==0?"chi":m.kind==1?"pon":"kan",m.open!=0);
    }
    return result;
}
riichi::WinContext context_from(const rq_win_context* ptr) {
    if(!ptr) return {};
    const auto& x=*ptr; riichi::WinContext c;
#define FLAG(name) require(x.name==0 || x.name==1,#name " must be boolean"); c.name=x.name!=0
    FLAG(tsumo); FLAG(riichi); FLAG(double_riichi); FLAG(ippatsu); FLAG(rinshan); FLAG(chankan);
    FLAG(haitei); FLAG(houtei); FLAG(tenhou); FLAG(chiihou); FLAG(renhou);
#undef FLAG
    c.seat_wind=x.seat_wind; c.round_wind=x.round_wind; c.red_dora=x.red_dora;
    require(x.n_dora>=0 && x.n_dora<=5 && x.n_ura>=0 && x.n_ura<=5,"at most five indicators");
    c.dora_indicators.assign(x.dora_indicators,x.dora_indicators+x.n_dora);
    c.ura_indicators.assign(x.ura_indicators,x.ura_indicators+x.n_ura);
    require(x.yaku_established>=-1 && x.yaku_established<=1,"yaku_established must be a boolean or None");
    if(x.yaku_established!=-1) c.yaku_established=x.yaku_established!=0;
    c.validate(); return c;
}
}
void riichi::abi::set_error(const char* message) noexcept {
    std::snprintf(last_error, sizeof(last_error), "%s", message);
}
extern "C" {
int32_t rq_abi_version() { return RQ_ABI_VERSION; }
int32_t rq_sizeof_rules() { return sizeof(rq_rules); }
int32_t rq_sizeof_meld() { return sizeof(rq_meld); }
int32_t rq_sizeof_win_context() { return sizeof(rq_win_context); }
int32_t rq_sizeof_state() { return sizeof(rq_state); }
int32_t rq_sizeof_hidden() { return sizeof(rq_hidden); }
const char* rq_last_error() { return last_error; }
int32_t rq_default_rules(rq_rules* out) { return guarded([&]{require(out,"rules output must not be NULL"); *out=riichi::Rules{}; return 0;}); }
int32_t rq_walls(uint64_t seed,int32_t count,int32_t* out,int32_t capacity) {
    return guarded([&]{
        require(count>=0 && count<=INT32_MAX/136,"invalid wall count"); require(capacity>=0,"invalid capacity");
        const int length=count*136;
        if(!out && capacity==0) return length;
        require(out && capacity>=length,"wall output capacity too small");
        riichi::Random rng(seed);
        for(int i=0;i<count;++i) { const auto w=riichi::build_wall(rng); std::copy(w.live.begin(),w.live.end(),out+i*136); std::copy(w.dead.begin(),w.dead.end(),out+i*136+122); }
        return length;
    });
}
int32_t rq_shanten(const int32_t* c,const rq_meld* m,int32_t n,int32_t* out) {
    return guarded([&]{require(out,"shanten output must not be NULL"); const auto result=riichi::shanten_all(counts_from(c),melds_from(m,n)); std::copy(result.begin(),result.end(),out); return 0;});
}
int32_t rq_waits(const int32_t* c,const rq_meld* m,int32_t n,int32_t* out) {
    return guarded([&]{require(out,"waits output must not be NULL"); const auto result=riichi::waits(counts_from(c),melds_from(m,n)); std::fill_n(out,34,0); for(int t:result) out[t]=1; return static_cast<int>(result.size());});
}
int32_t rq_is_agari(const int32_t* c,const rq_meld* m,int32_t n) {
    return guarded([&]{return static_cast<int>(riichi::is_agari(counts_from(c),melds_from(m,n)));});
}
int32_t rq_score_hand_json(const int32_t* c,int32_t win,const rq_meld* m,int32_t n,const rq_win_context* ctx,const rq_rules* rules,int32_t honba,int32_t sticks,const int32_t* ids,int32_t n_ids,char* out,int32_t capacity) {
    return guarded([&]{
        require(n_ids>=0 && n_ids<=136,"invalid tile ID count"); require(ids || n_ids==0,"tile IDs must not be NULL");
        require(capacity>=0,"invalid capacity");
        std::optional<std::vector<int>> tiles;
        if(ids) tiles=std::vector<int>(ids,ids+n_ids);
        const auto result=riichi::score_json(riichi::score_hand(counts_from(c),win,melds_from(m,n),context_from(ctx),rules?riichi::Rules(*rules):riichi::Rules{},honba,sticks,tiles));
        const int length=static_cast<int>(result.size());
        if(!out && capacity==0) return length;
        require(out && capacity>length,"score output capacity too small"); std::memcpy(out,result.c_str(),result.size()+1); return length;
    });
}
int32_t rq_placement_points(const int32_t* scores,const rq_rules* rules,double* out) {
    return guarded([&]{require(scores && out,"scores and output must not be NULL"); std::array<int,4> values{}; std::copy_n(scores,4,values.begin());
        const auto result=riichi::placement_points(values,rules?riichi::Rules(*rules):riichi::Rules{}); std::copy(result.begin(),result.end(),out); return 0;});
}
}
