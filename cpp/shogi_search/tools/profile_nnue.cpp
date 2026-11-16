// SPDX-License-Identifier: MIT
#include "shogi_search/nnue.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

namespace {
using Clock=std::chrono::steady_clock;
float scalar_tail(const shogi_search::Position& position,const shogi_search::nnue::NnueEvaluator& evaluator,
                  const shogi_search::nnue::Weights& weights) {
    std::vector<float> x;
    for (int side=0;side<2;++side) for (float value:evaluator.accumulator(position.board().turn^side))
        x.push_back(std::clamp(value,0.0F,1.0F));
    auto dense=[&](const auto& w,const auto& b,bool clip) {
        std::vector<float> y=b;
        for (std::size_t row=0;row<b.size();++row) {
            for (std::size_t column=0;column<x.size();++column) y[row]+=w[row*x.size()+column]*x[column];
            if (clip) y[row]=std::clamp(y[row],0.0F,1.0F);
        }
        x=std::move(y);
    };
    dense(weights.w1,weights.b1,true); dense(weights.w2,weights.b2,true); dense(weights.w3,weights.b3,false);
    return x[0]*shogi_search::nnue::output_scale;
}
}
int main(int argc,char** argv) {
    try {
        if (argc!=2) throw shogi::Error("usage: shogi_search_profile_nnue WEIGHTS");
        auto weights=shogi_search::nnue::Weights::load(argv[1]);
        shogi_search::Position position(shogi::parse_sfen(shogi::start_sfen));
        shogi_search::nnue::NnueEvaluator evaluator(weights);
        evaluator.reset(position);
        const int move=position.legal().values[0];
        const float reference=scalar_tail(position,evaluator,*weights);
        if (reference!=evaluator.evaluate_float(position)) throw shogi::Error("Dense parity mismatch");
        constexpr int iterations=100000;
        volatile float sink=0;
        auto measure=[&](auto operation) {
            const auto start=Clock::now();
            for (int i=0;i<iterations;++i) sink=operation();
            return std::chrono::duration<double>(Clock::now()-start).count()/iterations*1e6;
        };
        const double scalar=measure([&] { return scalar_tail(position,evaluator,*weights); });
        const double vectorised=measure([&] { return evaluator.evaluate_float(position); });
        const double transition=measure([&] {
            const auto undo=position.make(move); evaluator.push(position,undo);
            evaluator.pop(); position.unmake(undo); return 0.0F;
        });
        const double evaluated=measure([&] {
            const auto undo=position.make(move); evaluator.push(position,undo);
            const float value=evaluator.evaluate_float(position);
            evaluator.pop(); position.unmake(undo); return value;
        });
        std::cout << "{\"iterations\":" << iterations << ",\"scalar_dense_us\":" << scalar
                  << ",\"vector_dense_us\":" << vectorised << ",\"lazy_make_unmake_us\":" << transition
                  << ",\"evaluated_make_unmake_us\":" << evaluated << "}\n";
        (void)sink;
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
