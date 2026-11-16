// SPDX-License-Identifier: MIT
#ifndef SHOGI_SEARCH_NNUE_HPP
#define SHOGI_SEARCH_NNUE_HPP
#include "shogi_search/search.hpp"
#include "shogi_search/nnue_layout.hpp"
#include <string>

namespace shogi_search::nnue {
struct Features {
    std::array<int,max_active> indices{};
    int size{};
    int king{};
};
Features features(const shogi::Position&,int perspective);
void validate_position(const shogi::Position&);
struct Weights {
    int width{256}, hidden1{32}, hidden2{32};
    std::vector<float> embedding, bias, w1, b1, w2, b2, w3, b3;
    std::string architecture() const;
    void validate() const;
    void save(const std::string& path) const;
    static std::shared_ptr<const Weights> load(const std::string& path);
};
class NnueEvaluator final : public Evaluator {
public:
    explicit NnueEvaluator(std::shared_ptr<const Weights> weights);
    void reset(const Position&) override;
    void push(const Position&,const Undo&) override;
    void pop() noexcept override;
    int evaluate(const Position&) const override;
    float evaluate_float(const Position&) const;
    std::span<const float> accumulator(int side) const;
private:
    std::shared_ptr<const Weights> weights_;
    // Fixed capacity allocated once; make/unmake never allocates.
    mutable std::vector<float> stack_, compensation_;
    std::vector<float> w1_columns_, w2_columns_;
    std::array<std::array<Features,2>,max_ply+2> features_{};
    struct Delta {
        std::array<int,2> removed{}, added{};
        int removals{}, additions{};
        bool refresh{};
    };
    std::array<std::array<Delta,2>,max_ply+2> deltas_{};
    mutable std::array<std::array<bool,2>,max_ply+2> ready_{};
    int depth_{};
    std::span<float> current(int side,int depth) const;
    void add(int side,int depth,int feature,bool remove) const;
    void refresh(int side,int depth,const Features&) const;
    void ensure(int side,int depth) const;
};
}
#endif
