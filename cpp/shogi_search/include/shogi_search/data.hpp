// SPDX-License-Identifier: MIT
#ifndef SHOGI_SEARCH_DATA_HPP
#define SHOGI_SEARCH_DATA_HPP
#include "shogi/game.hpp"
#include <fstream>
#include <limits>
#include <set>
#include <span>

namespace shogi_search::nnue {
inline constexpr std::size_t record_size=137;
struct Record {
    shogi::Position position;
    std::int32_t score{};
    std::uint16_t depth{};
    std::uint64_t nodes{};
    std::uint16_t best_move{};
    std::int8_t result{};
    std::uint32_t ply{}, flags{};
    float teacher{std::numeric_limits<float>::quiet_NaN()};
    std::uint64_t game_id{};
};
std::vector<std::uint8_t> pack(const Record& record);
Record unpack(std::span<const std::uint8_t> bytes);
class DataWriter {
public:
    explicit DataWriter(const std::string& path);
    void append(const Record&);
    void close(); // Publishes the sidecar only after successful completion.
private:
    std::string path_;
    std::ofstream stream_;
    std::uint64_t count_{};
    std::set<std::uint64_t> games_;
    bool closed_{};
};
class DataReader {
public:
    explicit DataReader(const std::string& path);
    bool next(Record&);
private:
    std::ifstream stream_;
    std::uint64_t expected_{}, expected_games_{}, count_{};
    std::set<std::uint64_t> games_;
};
}
#endif
