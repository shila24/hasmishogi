#include "riichi/rules.hpp"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
namespace riichi {
Rules::Rules() : rq_rules{} {
    rounds = 0;
    start_points = 25000;
    return_points = 30000;
    uma[0] = 20;
    uma[1] = 10;
    uma[2] = -10;
    uma[3] = -20;
    red_fives = 1;
    kuitan = 1;
    atozuke = 1;
    ippatsu = 1;
    ura_dora = 1;
    kan_dora = 1;
    open_kan_dora_after_discard = 1;
    closed_kan_dora_immediate = 1;
    double_ron = 1;
    atama_hane = 0;
    triple_ron_abort = 1;
    kyuushu_kyuuhai = 1;
    suufon_renda = 1;
    suucha_riichi = 1;
    suukaikan = 1;
    tenpai_payment = 3000;
    riichi_stick = 1000;
    honba = 300;
    tobi = 1;
    agari_yame = 1;
    kazoe_yakuman = 1;
    double_yakuman = 0;
    multiple_yakuman = 1;
    pao = 0;
    renhou = 0;
    kiriage_mangan = 0;
    double_wind_pair_fu = 4;
    nagashi_mangan = 1;
    discard_furiten = 1;
    temporary_furiten = 1;
    riichi_furiten = 1;
    track_tsumogiri = 1;
}
Rules::Rules(const rq_rules& value) : rq_rules(value) { validate(); }
void Rules::validate() const {
    if (red_fives != 0 && red_fives != 1) throw std::invalid_argument("red_fives must be boolean");
    if (kuitan != 0 && kuitan != 1) throw std::invalid_argument("kuitan must be boolean");
    if (atozuke != 0 && atozuke != 1) throw std::invalid_argument("atozuke must be boolean");
    if (ippatsu != 0 && ippatsu != 1) throw std::invalid_argument("ippatsu must be boolean");
    if (ura_dora != 0 && ura_dora != 1) throw std::invalid_argument("ura_dora must be boolean");
    if (kan_dora != 0 && kan_dora != 1) throw std::invalid_argument("kan_dora must be boolean");
    if (open_kan_dora_after_discard != 0 && open_kan_dora_after_discard != 1) throw std::invalid_argument("open_kan_dora_after_discard must be boolean");
    if (closed_kan_dora_immediate != 0 && closed_kan_dora_immediate != 1) throw std::invalid_argument("closed_kan_dora_immediate must be boolean");
    if (double_ron != 0 && double_ron != 1) throw std::invalid_argument("double_ron must be boolean");
    if (atama_hane != 0 && atama_hane != 1) throw std::invalid_argument("atama_hane must be boolean");
    if (triple_ron_abort != 0 && triple_ron_abort != 1) throw std::invalid_argument("triple_ron_abort must be boolean");
    if (kyuushu_kyuuhai != 0 && kyuushu_kyuuhai != 1) throw std::invalid_argument("kyuushu_kyuuhai must be boolean");
    if (suufon_renda != 0 && suufon_renda != 1) throw std::invalid_argument("suufon_renda must be boolean");
    if (suucha_riichi != 0 && suucha_riichi != 1) throw std::invalid_argument("suucha_riichi must be boolean");
    if (suukaikan != 0 && suukaikan != 1) throw std::invalid_argument("suukaikan must be boolean");
    if (tobi != 0 && tobi != 1) throw std::invalid_argument("tobi must be boolean");
    if (agari_yame != 0 && agari_yame != 1) throw std::invalid_argument("agari_yame must be boolean");
    if (kazoe_yakuman != 0 && kazoe_yakuman != 1) throw std::invalid_argument("kazoe_yakuman must be boolean");
    if (double_yakuman != 0 && double_yakuman != 1) throw std::invalid_argument("double_yakuman must be boolean");
    if (multiple_yakuman != 0 && multiple_yakuman != 1) throw std::invalid_argument("multiple_yakuman must be boolean");
    if (pao != 0 && pao != 1) throw std::invalid_argument("pao must be boolean");
    if (kiriage_mangan != 0 && kiriage_mangan != 1) throw std::invalid_argument("kiriage_mangan must be boolean");
    if (nagashi_mangan != 0 && nagashi_mangan != 1) throw std::invalid_argument("nagashi_mangan must be boolean");
    if (discard_furiten != 0 && discard_furiten != 1) throw std::invalid_argument("discard_furiten must be boolean");
    if (temporary_furiten != 0 && temporary_furiten != 1) throw std::invalid_argument("temporary_furiten must be boolean");
    if (riichi_furiten != 0 && riichi_furiten != 1) throw std::invalid_argument("riichi_furiten must be boolean");
    if (track_tsumogiri != 0 && track_tsumogiri != 1) throw std::invalid_argument("track_tsumogiri must be boolean");
    if (rounds < 0 || rounds > 1 || renhou < 0 || renhou > 2) throw std::invalid_argument("unsupported rounds or renhou option");
    if (start_points < 0 || start_points % 100) throw std::invalid_argument("start_points must be a nonnegative multiple of 100");
    if (return_points < 0 || return_points % 100) throw std::invalid_argument("return_points must be a nonnegative multiple of 100");
    if (tenpai_payment < 0 || tenpai_payment % 100) throw std::invalid_argument("tenpai_payment must be a nonnegative multiple of 100");
    if (riichi_stick < 0 || riichi_stick % 100) throw std::invalid_argument("riichi_stick must be a nonnegative multiple of 100");
    if (honba < 0 || honba % 100) throw std::invalid_argument("honba must be a nonnegative multiple of 100");
    if (return_points < start_points || start_points == 0) throw std::invalid_argument("return points must be at least positive starting points");
    if (honba % 300 || tenpai_payment % 300) throw std::invalid_argument("honba and tenpai payment must be divisible by 300");
    if (std::accumulate(std::begin(uma), std::end(uma), std::int64_t{0}) != 0 ||
        !std::is_sorted(std::begin(uma), std::end(uma), std::greater<int>()))
        throw std::invalid_argument("uma must be descending and sum to zero");
    if (double_wind_pair_fu != 2 && double_wind_pair_fu != 4) throw std::invalid_argument("double wind pair must score 2 or 4 fu");
}
}
