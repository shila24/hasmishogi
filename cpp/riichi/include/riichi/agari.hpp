#pragma once
#include "riichi/tiles.hpp"
#include <optional>
#include <string>
#include <vector>
namespace riichi {
struct Meld {
    std::vector<int> tiles;
    std::string kind;
    bool open = true;
    Meld(std::vector<int> tiles, std::string kind, bool open = true);
    void validate() const;
};
struct Group {
    std::vector<int> tiles;
    bool open = false;
    bool declared = false;
    bool sequence() const { return tiles.at(0) != tiles.at(1); }
    bool kan() const { return tiles.size() == 4; }
};
struct Decomposition {
    std::string kind;
    std::vector<Group> groups{};
    std::optional<int> pair{}, winning_group{}, winning_tile{};
    std::vector<int> special_pairs{};
    bool menzen() const;
    std::string wait() const;
};
Counts hand_context(const Counts& counts, const std::vector<Meld>& melds = {});
std::vector<Decomposition> decompose(const Counts& counts, const std::vector<Meld>& melds = {}, std::optional<int> winning_tile = {});
bool is_agari(const Counts& counts, const std::vector<Meld>& melds = {});
}
