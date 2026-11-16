#pragma once
#include "hasami/game.hpp"
#include <atomic>
#include <cstdint>
#include <memory>

namespace hasami {
struct SearchLimits {
    int max_depth = 3;
    int time_ms = 1000;
    std::uint64_t node_limit = 200000;
    void validate() const;
};
struct SearchResult {
    int action = -1;
    int score = 0;
    int depth = 0;
    std::uint64_t nodes = 0;
    double elapsed_ms = 0;
    bool completed = false;
};
// Agents may inspect and copy a game, but must not mutate the authoritative game.
class IAgent {
public:
    virtual ~IAgent() = default;
    virtual SearchResult choose(const Game& game, const SearchLimits& limits,
                                const std::atomic_bool* cancel = nullptr) = 0;
};
class AlphaBetaAgent final : public IAgent {
public:
    SearchResult choose(const Game& game, const SearchLimits& limits,
                        const std::atomic_bool* cancel = nullptr) override;
};
class RandomAgent final : public IAgent {
public:
    explicit RandomAgent(std::uint64_t seed = 0) : seed_(seed) {}
    SearchResult choose(const Game& game, const SearchLimits& limits,
                        const std::atomic_bool* cancel = nullptr) override;
private:
    std::uint64_t seed_;
};
int evaluate(const Position& p);
}  // namespace hasami
