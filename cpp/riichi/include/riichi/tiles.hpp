#pragma once
#include "riichi/rng.hpp"
#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>
namespace riichi {
using Counts = std::array<int, 34>;
constexpr bool terminal(int t) { return t < 27 && (t % 9 == 0 || t % 9 == 8); }
constexpr bool orphan(int t) { return t >= 27 || terminal(t); }
constexpr bool red_id(int t) { return t == 16 || t == 52 || t == 88; }
void validate_counts(const Counts& counts);
void validate_ids(std::span<const int> ids);
Counts to_counts(std::span<const int> ids);
Counts parse_counts(std::string_view code);
std::vector<int> parse_tiles(std::string_view code, bool red_fives = true);
std::string format_counts(const Counts& counts);
std::string format_tiles(std::span<const int> ids, bool red_fives = true);
int dora_type(int indicator);
struct Wall {
    std::array<int, 122> live{};
    std::array<int, 14> dead{};
    std::array<int, 4> rinshan() const;
    std::array<int, 5> dora_indicators() const;
    std::array<int, 5> ura_indicators() const;
};
Wall build_wall(Random& rng);
Wall build_wall(std::uint64_t seed);
}
