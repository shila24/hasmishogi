#include "riichi/game.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}
int main() {
    try {
        riichi::Game game(0);
        check(game.state().remaining_wall == 69, "initial live wall");
        check(game.actors() == std::vector<int>{0}, "initial actor");
        const auto before = game.observation_json(0);
        try { game.play_joint({-1, -1, -1, -1}); throw std::runtime_error("invalid joint accepted"); }
        catch (const std::invalid_argument&) {}
        check(game.observation_json(0) == before, "invalid action mutated state");
        auto clone = game;
        for (int seat = 0; seat < 4; ++seat) {
            riichi::Joint joint{-1, -1, -1, -1};
            joint[seat] = clone.legal_actions(seat).front(); clone.play_joint(joint);
            check(clone.actors() == std::vector<int>{(seat + 1) % 4, (seat + 2) % 4, (seat + 3) % 4}, "claim turn order");
            joint.fill(109); joint[seat] = -1; clone.play_joint(joint);
        }
        check(game.observation_json(0) == before, "clone independence");
        riichi::Random rng(24024);
        for (int seed = 0; seed < 10; ++seed) {
            riichi::Rules rules; rules.rounds = seed % 2; rules.nagashi_mangan = seed % 3 == 0;
            riichi::Game match(seed, rules), replay(seed, rules);
            while (!match.state().terminal) {
                check(match.moves().size() < 20000, "match did not terminate");
                riichi::Joint joint{-1, -1, -1, -1};
                for (int p : match.actors()) {
                    const auto legal = match.legal_actions(p);
                    check(!legal.empty(), "actor without legal actions");
                    joint[p] = legal[rng.randbelow(static_cast<unsigned>(legal.size()))];
                    std::array<float, RQ_PLANES_SIZE> planes{};
                    std::array<float, RQ_ACTION_SIZE> mask{};
                    match.encode(p, planes); match.legal_mask(p, mask);
                    check(mask[joint[p]] == 1.0F && planes.back() == 1.0F, "encoding and mask");
                }
                match.play_joint(joint); replay.play_joint(joint);
                check(match.last_hand_json() == replay.last_hand_json(), "replay summary");
                int sum = match.state().riichi_sticks * rules.riichi_stick;
                for (int score : match.state().scores) sum += score;
                check(sum == rules.start_points * 4, "point conservation");
            }
            check(match.actors().empty(), "terminal actors");
            for (int p = 0; p < 4; ++p) check(match.observation_json(p) == replay.observation_json(p), "replay observation");
        }
        std::cout << "game state, encoding, replay, conservation: passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
