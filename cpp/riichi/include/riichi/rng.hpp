#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>

namespace riichi {
// CPython integer seeding and rejection sampling, independent of std::shuffle.
class Random {
public:
    explicit Random(std::uint64_t seed = 0) { reseed(seed); }
    void reseed(std::uint64_t seed);
    std::uint32_t getrandbits(unsigned bits);
    std::uint32_t randbelow(std::uint32_t n);
    template<class T> void shuffle(std::span<T> values) {
        for (std::size_t i = values.size(); i > 1; --i) {
            if (i > UINT32_MAX) throw std::invalid_argument("shuffle too large");
            std::swap(values[i - 1], values[randbelow(static_cast<std::uint32_t>(i))]);
        }
    }
private:
    std::array<std::uint32_t, 624> state_{};
    std::size_t index_ = 624;
    std::uint32_t next();
};
}
