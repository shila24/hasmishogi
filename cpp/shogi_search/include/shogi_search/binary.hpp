// SPDX-License-Identifier: MIT
#ifndef SHOGI_SEARCH_BINARY_HPP
#define SHOGI_SEARCH_BINARY_HPP
#include <array>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <span>
#include <string>
#include <vector>

namespace shogi_search::binary {
class Sha256 {
public:
    void update(std::span<const std::uint8_t> bytes);
    std::array<std::uint8_t, 32> finish();
private:
    std::array<std::uint32_t, 8> state_{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                                      0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    std::array<std::uint8_t, 64> block_{};
    std::uint64_t bytes_{};
    std::size_t used_{};
    void compress();
};
std::string hex(std::span<const std::uint8_t> bytes);
std::string file_hash(const std::string& path);
void put(std::vector<std::uint8_t>& out, std::uint64_t value, int bytes);
std::uint64_t get(std::span<const std::uint8_t> data, std::size_t& offset, int bytes);
std::vector<std::uint8_t> read(std::istream& stream, std::size_t size);
void text(std::vector<std::uint8_t>& out, const std::string& value, std::size_t size);
}
#endif
