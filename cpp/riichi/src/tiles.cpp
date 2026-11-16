#include "riichi/tiles.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
namespace riichi {
void validate_counts(const Counts& c) {
    for (int n : c) if (n < 0 || n > 4) throw std::invalid_argument("expected 34 integer counts in 0..4");
}
void validate_ids(std::span<const int> ids) {
    std::array<bool, 136> seen{};
    for (int t : ids) {
        if (t < 0 || t >= 136) throw std::invalid_argument("physical tile IDs must be integers in 0..135");
        if (seen[t]) throw std::invalid_argument("duplicate physical tile ID");
        seen[t] = true;
    }
}
Counts to_counts(std::span<const int> ids) {
    validate_ids(ids); Counts c{};
    for (int t : ids) ++c[t/4];
    return c;
}
namespace {
std::vector<std::pair<int, bool>> codes(std::string_view code) {
    std::vector<std::pair<int, bool>> result;
    std::size_t pos = 0;
    while (pos < code.size()) {
        const auto start = pos;
        while (pos < code.size() && code[pos] >= '0' && code[pos] <= '9') ++pos;
        if (pos == start || pos == code.size()) throw std::invalid_argument("invalid tile notation");
        const auto suit = std::string_view("mpsz").find(code[pos]);
        if (suit == std::string_view::npos) throw std::invalid_argument("invalid tile notation");
        for (auto i = start; i < pos; ++i) {
            int n = code[i] - '0';
            if (suit == 3 && (n < 1 || n > 7)) throw std::invalid_argument("honors use 1z..7z");
            result.emplace_back(static_cast<int>(suit)*9 + (n ? n : 5)-1, n == 0);
        }
        ++pos;
    }
    return result;
}
}
Counts parse_counts(std::string_view code) {
    Counts c{}; for (auto [t, red] : codes(code)) { (void)red; ++c[t]; }
    validate_counts(c); return c;
}
std::vector<int> parse_tiles(std::string_view code, bool red_fives) {
    std::array<bool,136> used{}; std::vector<int> result;
    for (auto [t, red] : codes(code)) {
        if (red && !red_fives) throw std::invalid_argument("red fives are disabled");
        int id = t*4;
        const int end = id + (red ? 1 : 4);
        while (id < end && (used[id] || (!red && red_fives && red_id(id)))) ++id;
        if (id == end) throw std::invalid_argument("too many copies of a tile");
        result.push_back(id); used[id] = true;
    }
    return result;
}
std::string format_counts(const Counts& c) {
    validate_counts(c); std::string result;
    for (int s = 0; s < 4; ++s) {
        std::string part;
        for (int t = s*9; t < std::min(34,s*9+9); ++t) part.append(c[t], static_cast<char>('1'+t%9));
        if (!part.empty()) { result += part; result += "mpsz"[s]; }
    }
    return result;
}
std::string format_tiles(std::span<const int> ids, bool red_fives) {
    validate_ids(ids); std::vector<int> sorted(ids.begin(),ids.end()); std::sort(sorted.begin(),sorted.end());
    std::array<std::string,4> parts{};
    for (int t : sorted) parts[t/36] += red_fives && red_id(t) ? '0' : static_cast<char>('1'+t/4%9);
    std::string result;
    for (int s=0;s<4;++s) if (!parts[s].empty()) { result+=parts[s]; result+="mpsz"[s]; }
    return result;
}
int dora_type(int t) {
    if (t < 0 || t >= 34) throw std::invalid_argument("indicator type must be in 0..33");
    if (t < 27) return t/9*9+(t+1)%9;
    const int start = t<31 ? 27 : 31, length = t<31 ? 4 : 3;
    return start+(t-start+1)%length;
}
std::array<int,4> Wall::rinshan() const { return {dead[0],dead[1],dead[2],dead[3]}; }
std::array<int,5> Wall::dora_indicators() const { return {dead[4],dead[6],dead[8],dead[10],dead[12]}; }
std::array<int,5> Wall::ura_indicators() const { return {dead[5],dead[7],dead[9],dead[11],dead[13]}; }
Wall build_wall(Random& rng) {
    std::array<int,136> ids{}; std::iota(ids.begin(),ids.end(),0); rng.shuffle<int>(ids);
    Wall wall; std::copy_n(ids.begin(),122,wall.live.begin()); std::copy_n(ids.begin()+122,14,wall.dead.begin()); return wall;
}
Wall build_wall(std::uint64_t seed) { Random rng(seed); return build_wall(rng); }
}
