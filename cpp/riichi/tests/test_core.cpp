#include "riichi/core.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
void check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
int main() { try {
    { riichi::Random rng(0ULL);
      check(rng.getrandbits(1)==1U,"CPython RNG vector");
      check(rng.getrandbits(5)==12U,"CPython RNG vector");
      check(rng.getrandbits(32)==3255389356U,"CPython RNG vector");
      check(rng.getrandbits(17)==116686U,"CPython RNG vector");
      check(rng.getrandbits(32)==1806341205U,"CPython RNG vector");
    }
    { riichi::Random rng(1ULL);
      check(rng.getrandbits(1)==0U,"CPython RNG vector");
      check(rng.getrandbits(5)==18U,"CPython RNG vector");
      check(rng.getrandbits(32)==3639700191U,"CPython RNG vector");
      check(rng.getrandbits(17)==105154U,"CPython RNG vector");
      check(rng.getrandbits(32)==3280387012U,"CPython RNG vector");
    }
    { riichi::Random rng(4294967296ULL);
      check(rng.getrandbits(1)==0U,"CPython RNG vector");
      check(rng.getrandbits(5)==11U,"CPython RNG vector");
      check(rng.getrandbits(32)==1794561286U,"CPython RNG vector");
      check(rng.getrandbits(17)==122515U,"CPython RNG vector");
      check(rng.getrandbits(32)==71624475U,"CPython RNG vector");
    }
    { riichi::Random rng(9223372036854775807ULL);
      check(rng.getrandbits(1)==0U,"CPython RNG vector");
      check(rng.getrandbits(5)==21U,"CPython RNG vector");
      check(rng.getrandbits(32)==2711238091U,"CPython RNG vector");
      check(rng.getrandbits(17)==17391U,"CPython RNG vector");
      check(rng.getrandbits(32)==3451239204U,"CPython RNG vector");
    }
    { riichi::Random rng(18446744073709551615ULL);
      check(rng.getrandbits(1)==0U,"CPython RNG vector");
      check(rng.getrandbits(5)==7U,"CPython RNG vector");
      check(rng.getrandbits(32)==1452108352U,"CPython RNG vector");
      check(rng.getrandbits(17)==81100U,"CPython RNG vector");
      check(rng.getrandbits(32)==910393425U,"CPython RNG vector");
    }
    check(riichi::shanten(riichi::parse_counts("19m19p19s11234567z"))==-1,"kokushi");
    check(riichi::shanten(riichi::parse_counts("1111m222p333s444z"))==1,"fifth copy");
    check(riichi::waits(riichi::parse_counts("123m456p789s1122z"))==std::vector<int>({27,28}),"waits");
    check(riichi::decompose(riichi::parse_counts("11223344556677m")).size()==4,"partition order");
    check(riichi::decompose(riichi::parse_counts("111123m456p789s22z"),{},0).size()==2,"duplicate partitions");
    check(riichi::shanten(riichi::parse_counts("6889m112z"),{{{6,6,6},"pon"},{{9,9,9},"pon"}})==2,"meld capacities");
    for(const auto* code:{"123m4p0s7z","0555m123p777z","19m19p19s1234567z",""}) {
        const auto ids=riichi::parse_tiles(code);
        check(riichi::to_counts(ids)==riichi::parse_counts(code),"tile counts");
        check(riichi::format_tiles(riichi::parse_tiles(riichi::format_tiles(ids)))==riichi::format_tiles(ids),"tile roundtrip");
    }
    for(const auto* code:{"0z","8z","10z","123","m","11111m","00p","5555s","1x"}) {
        bool rejected=false; try { riichi::parse_tiles(code); } catch(const std::invalid_argument&) { rejected=true; }
        check(rejected,"invalid notation");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==2 && score.fu==30 && score.yakuman==0 && score.payment.total==2000,"riichi score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.double_riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==3 && score.fu==30 && score.yakuman==0 && score.payment.total==3900,"double_riichi score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.ippatsu=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==3 && score.fu==30 && score.yakuman==0 && score.payment.total==3900,"ippatsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.tsumo=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==3 && score.fu==20 && score.yakuman==0 && score.payment.total==2700,"menzen_tsumo score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==2 && score.fu==30 && score.yakuman==0 && score.payment.total==2000,"pinfu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("234456m23455p678s"),1,{},context);
      check(score.han==3 && score.fu==30 && score.yakuman==0 && score.payment.total==3900,"tanyao score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("456m23455p678s"),3,{{{1,2,3},"chi",true}},context);
      check(score.han==1 && score.fu==30 && score.yakuman==0 && score.payment.total==1000,"tanyao score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("112233m456p678s55z"),0,{},context);
      check(score.han==2 && score.fu==40 && score.yakuman==0 && score.payment.total==2600,"iipeikou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("112233m445566p77s"),0,{},context);
      check(score.han==5 && score.fu==30 && score.yakuman==0 && score.payment.total==8000,"ryanpeikou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123m123p123789s55z"),0,{},context);
      check(score.han==5 && score.fu==40 && score.yakuman==0 && score.payment.total==8000,"sanshoku_doujun score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("123p123789s55z"),9,{{{0,1,2},"chi",true}},context);
      check(score.han==2 && score.fu==30 && score.yakuman==0 && score.payment.total==2000,"sanshoku_doujun score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456789m234p55z"),0,{},context);
      check(score.han==3 && score.fu==40 && score.yakuman==0 && score.payment.total==5200,"ittsuu score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("456789m234p55z"),3,{{{0,1,2},"chi",true}},context);
      check(score.han==1 && score.fu==30 && score.yakuman==0 && score.payment.total==1000,"ittsuu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123789m111p789s55z"),0,{},context);
      check(score.han==3 && score.fu==40 && score.yakuman==0 && score.payment.total==5200,"chanta score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("789m111p789s55z"),6,{{{0,1,2},"chi",true}},context);
      check(score.han==1 && score.fu==40 && score.yakuman==0 && score.payment.total==1300,"chanta score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123789m111p789s99p"),0,{},context);
      check(score.han==4 && score.fu==40 && score.yakuman==0 && score.payment.total==8000,"junchan score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("789m111p789s99p"),6,{{{0,1,2},"chi",true}},context);
      check(score.han==2 && score.fu==30 && score.yakuman==0 && score.payment.total==2000,"junchan score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111999m111p999s55z"),0,{},context);
      check(score.han==7 && score.fu==60 && score.yakuman==0 && score.payment.total==12000,"honroutou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("1199m1199p1199s55z"),0,{},context);
      check(score.han==5 && score.fu==25 && score.yakuman==0 && score.payment.total==8000,"honroutou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111m222p333444s55z"),21,{},context);
      check(score.han==5 && score.fu==50 && score.yakuman==0 && score.payment.total==8000,"toitoi score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111m222p333456s55z"),23,{},context);
      check(score.han==3 && score.fu==50 && score.yakuman==0 && score.payment.total==6400,"sanankou score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("678s55z"),23,{{{0,0,0,0},"kan",true},{{10,10,10,10},"kan",true},{{20,20,20,20},"kan",true}},context);
      check(score.han==2 && score.fu==60 && score.yakuman==0 && score.payment.total==3900,"sankantsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123m456p55566677z"),0,{},context);
      check(score.han==5 && score.fu==50 && score.yakuman==0 && score.payment.total==8000,"shousangen score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456789m11122z"),0,{},context);
      check(score.han==7 && score.fu==40 && score.yakuman==0 && score.payment.total==12000,"honitsu score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("456789m11122z"),3,{{{0,1,2},"chi",true}},context);
      check(score.han==4 && score.fu==30 && score.yakuman==0 && score.payment.total==7700,"honitsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("12334556778922m"),0,{},context);
      check(score.han==8 && score.fu==30 && score.yakuman==0 && score.payment.total==16000,"chinitsu score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("34556778922m"),2,{{{0,1,2},"chi",true}},context);
      check(score.han==5 && score.fu==30 && score.yakuman==0 && score.payment.total==8000,"chinitsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("11m22p334455s66z77z"),0,{},context);
      check(score.han==3 && score.fu==25 && score.yakuman==0 && score.payment.total==3200,"chiitoitsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m789p55566z"),0,{},context);
      check(score.han==2 && score.fu==40 && score.yakuman==0 && score.payment.total==2600,"haku score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m789p66677z"),0,{},context);
      check(score.han==2 && score.fu==40 && score.yakuman==0 && score.payment.total==2600,"hatsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m789p77755z"),0,{},context);
      check(score.han==2 && score.fu==40 && score.yakuman==0 && score.payment.total==2600,"chun score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m789p22255z"),0,{},context);
      check(score.han==2 && score.fu==40 && score.yakuman==0 && score.payment.total==2600,"seat_wind score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m789p11155z"),0,{},context);
      check(score.han==2 && score.fu==40 && score.yakuman==0 && score.payment.total==2600,"round_wind score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.tsumo=true;
      context.haitei=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==4 && score.fu==20 && score.yakuman==0 && score.payment.total==5200,"haitei score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.houtei=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==3 && score.fu==30 && score.yakuman==0 && score.payment.total==3900,"houtei score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.tsumo=true;
      context.rinshan=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123m456p789s55z"),0,{{{28,28,28,28},"kan",false}},context);
      check(score.han==4 && score.fu==60 && score.yakuman==0 && score.payment.total==8000,"rinshan score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.chankan=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==3 && score.fu==30 && score.yakuman==0 && score.payment.total==3900,"chankan score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111m111p111789s55z"),0,{},context);
      check(score.han==5 && score.fu==60 && score.yakuman==0 && score.payment.total==8000,"sanshoku_doukou score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      context.tsumo=true;
      context.tenhou=true;
      context.seat_wind=27;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==0 && score.fu==20 && score.yakuman==1 && score.payment.total==48000,"tenhou score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      context.tsumo=true;
      context.chiihou=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123456m23455p678s"),0,{},context);
      check(score.han==0 && score.fu==20 && score.yakuman==1 && score.payment.total==32000,"chiihou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("19m19p19s11234567z"),8,{},context);
      check(score.han==0 && score.fu==0 && score.yakuman==1 && score.payment.total==32000,"kokushi score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("19m19p19s11234567z"),27,{},context);
      check(score.han==0 && score.fu==0 && score.yakuman==1 && score.payment.total==32000,"kokushi_13 score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      context.tsumo=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111m222p333444s55z"),21,{},context);
      check(score.han==0 && score.fu==50 && score.yakuman==1 && score.payment.total==32000,"suuankou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111m222p333444s55z"),31,{},context);
      check(score.han==0 && score.fu==60 && score.yakuman==1 && score.payment.total==32000,"suuankou_tanki score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123m55p555666777z"),0,{},context);
      check(score.han==0 && score.fu==60 && score.yakuman==1 && score.payment.total==32000,"daisangen score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("123m11122233344z"),0,{},context);
      check(score.han==0 && score.fu==60 && score.yakuman==1 && score.payment.total==32000,"shousuushii score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("55m111222333444z"),27,{},context);
      check(score.han==0 && score.fu==60 && score.yakuman==1 && score.payment.total==32000,"daisuushii score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("11122233355566z"),27,{},context);
      check(score.han==0 && score.fu==60 && score.yakuman==1 && score.payment.total==32000,"tsuuiisou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("11223344556677z"),27,{},context);
      check(score.han==0 && score.fu==25 && score.yakuman==1 && score.payment.total==32000,"tsuuiisou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("111999m111p999s99p"),0,{},context);
      check(score.han==0 && score.fu==60 && score.yakuman==1 && score.payment.total==32000,"chinroutou score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("22233344466688s"),19,{},context);
      check(score.han==0 && score.fu==50 && score.yakuman==1 && score.payment.total==32000,"ryuuiisou score");
    }
    { riichi::WinContext context;
      context.riichi=false;
      const auto score=riichi::score_hand(riichi::parse_counts("55z"),31,{{{0,0,0,0},"kan",true},{{10,10,10,10},"kan",true},{{20,20,20,20},"kan",true},{{21,21,21,21},"kan",true}},context);
      check(score.han==0 && score.fu==70 && score.yakuman==1 && score.payment.total==32000,"suukantsu score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("11123455678999m"),0,{},context);
      check(score.han==0 && score.fu==50 && score.yakuman==1 && score.payment.total==32000,"chuuren score");
    }
    { riichi::WinContext context;
      context.riichi=true;
      const auto score=riichi::score_hand(riichi::parse_counts("11123455678999m"),4,{},context);
      check(score.han==0 && score.fu==50 && score.yakuman==1 && score.payment.total==32000,"junsei_chuuren score");
    }
    const std::array<std::array<int,4>,3> ron_non{{{1000,2000,3900,7700},{1300,2600,5200,8000},{1600,3200,6400,8000}}};
    const std::array<std::array<int,4>,3> ron_dealer{{{1500,2900,5800,11600},{2000,3900,7700,12000},{2400,4800,9600,12000}}};
    for(int row=0;row<3;++row) for(int han=1;han<=4;++han) {
        const auto base=riichi::base_points(han,30+10*row);
        check(riichi::payments(base).ron==ron_non[row][han-1],"nondealer ron grid");
        check(riichi::payments(base,true).ron==ron_dealer[row][han-1],"dealer ron grid");
    }
    for(const auto& row:std::vector<std::array<int,3>>{{3,70,2000},{4,40,2000},{5,30,2000},{6,30,3000},{8,30,4000},{11,30,6000},{13,30,8000}})
        check(riichi::base_points(row[0],row[1])==row[2],"limit points");
    check(riichi::base_points(0,0,2)==16000,"multiple yakuman base");
    {
        riichi::Rules rules; rules.double_wind_pair_fu=2;
        riichi::WinContext context; context.seat_wind=27;
        check(riichi::pair_fu(27,context,rules)==2,"double wind pair override");
        const auto score=riichi::score_hand(riichi::parse_counts("999s55z"),26,
            {{{0,0,0,0},"kan",false},{{8,8,8,8},"kan",false},{{9,9,9,9},"kan",false}});
        check(score.fu==140 && score.base_points==4000,"high fu");
    }
    for(int winner=0;winner<4;++winner) for(int dealer=0;dealer<4;++dealer) for(bool tsumo:{false,true}) {
        const auto pay=riichi::payments(2000,winner==dealer,tsumo,2,1);
        const auto delta=riichi::settle_payment(pay,winner,dealer,tsumo?std::nullopt:std::optional<int>((winner+1)%4));
        check(delta[0]+delta[1]+delta[2]+delta[3]==1000,"settlement");
    }
    check(riichi::tenpai_payments({true,false,false,false})==std::array<std::int64_t,4>{3000,-1000,-1000,-1000},"tenpai");
    check(riichi::placement_points({25000,25000,25000,25000})==std::array<double,4>{35,5,-15,-25},"placement ties");
    std::cout << "Core vectors passed\n";
    return 0;
} catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
