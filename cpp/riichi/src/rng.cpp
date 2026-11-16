#include "riichi/rng.hpp"
#include <algorithm>
namespace riichi {
void Random::reseed(std::uint64_t seed) {
    state_[0] = 19650218U;
    for (std::size_t i = 1; i < 624; ++i)
        state_[i] = 1812433253U * (state_[i-1] ^ (state_[i-1] >> 30)) + static_cast<std::uint32_t>(i);
    const std::array<std::uint32_t, 2> key{static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
    const std::size_t length = key[1] ? 2 : 1;
    std::size_t i = 1, j = 0;
    for (int k = 624; k > 0; --k) {
        state_[i] = (state_[i] ^ ((state_[i-1] ^ (state_[i-1] >> 30)) * 1664525U)) + key[j] + static_cast<std::uint32_t>(j);
        if (++i == 624) { state_[0] = state_[623]; i = 1; }
        if (++j == length) j = 0;
    }
    for (int k = 623; k > 0; --k) {
        state_[i] = (state_[i] ^ ((state_[i-1] ^ (state_[i-1] >> 30)) * 1566083941U)) - static_cast<std::uint32_t>(i);
        if (++i == 624) { state_[0] = state_[623]; i = 1; }
    }
    state_[0] = 0x80000000U;
    index_ = 624;
}
std::uint32_t Random::next() {
    if (index_ == 624) {
        for (std::size_t i = 0; i < 624; ++i) {
            const auto y = (state_[i] & 0x80000000U) | (state_[(i+1)%624] & 0x7fffffffU);
            state_[i] = state_[(i+397)%624] ^ (y >> 1) ^ ((y & 1U) ? 0x9908b0dfU : 0U);
        }
        index_ = 0;
    }
    auto y = state_[index_++];
    y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680U; y ^= (y << 15) & 0xefc60000U; y ^= y >> 18;
    return y;
}
std::uint32_t Random::getrandbits(unsigned bits) {
    if (bits > 32) throw std::invalid_argument("getrandbits requires 0..32 bits");
    return bits ? next() >> (32 - bits) : 0;
}
std::uint32_t Random::randbelow(std::uint32_t n) {
    if (!n) throw std::invalid_argument("randbelow requires positive n");
    const unsigned bits = std::bit_width(n);
    auto r = getrandbits(bits);
    while (r >= n) r = getrandbits(bits);
    return r;
}
}
