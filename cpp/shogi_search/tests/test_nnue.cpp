// SPDX-License-Identifier: MIT
#include "shogi_search/nnue.hpp"
#include "shogi_search/binary.hpp"
#include "shogi_search/data.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <random>

using namespace shogi_search;
namespace {
void check(bool condition,const char* message) { if (!condition) throw shogi::Error(message); }
std::shared_ptr<nnue::Weights> test_weights(int width) {
    auto w=std::make_shared<nnue::Weights>(); w->width=width;
    std::mt19937 random(70050);
    auto fill=[&](auto& tensor,std::size_t count,float scale) {
        tensor.resize(count);
        for (auto& v:tensor) v=(static_cast<int>(random()%2001)-1000)*scale;
    };
    fill(w->embedding,static_cast<std::size_t>(nnue::feature_count)*width,0.00001F);
    w->bias.assign(width,0.25F);
    fill(w->w1,2*width*32,0.0001F); w->b1.assign(32,0.1F);
    fill(w->w2,32*32,0.0001F); w->b2.assign(32,0.1F);
    fill(w->w3,32,0.0001F); w->b3.assign(1,0.0F); w->validate(); return w;
}
float scalar_tail(const Position& p,const nnue::NnueEvaluator& evaluator,const nnue::Weights& w) {
    std::vector<float> x;
    for (int order=0;order<2;++order) for (float value:evaluator.accumulator(p.board().turn^order))
        x.push_back(std::clamp(value,0.0F,1.0F));
    auto dense=[&](const auto& weights,const auto& bias,bool clip) {
        std::vector<float> y=bias;
        for (std::size_t row=0;row<y.size();++row) {
            for (std::size_t column=0;column<x.size();++column) y[row]+=weights[row*x.size()+column]*x[column];
            if (clip) y[row]=std::clamp(y[row],0.0F,1.0F);
        }
        x=std::move(y);
    };
    dense(w.w1,w.b1,true); dense(w.w2,w.b2,true); dense(w.w3,w.b3,false);
    return x[0]*nnue::output_scale;
}
void compare(const Position& p,const nnue::NnueEvaluator& inc,nnue::NnueEvaluator& full,float& worst) {
    full.reset(p);
    for (int side=0;side<2;++side) {
        const auto a=inc.accumulator(side), b=full.accumulator(side);
        for (std::size_t i=0;i<a.size();++i) {
            worst=std::max(worst,std::abs(a[i]-b[i]));
            check(std::abs(a[i]-b[i])<1e-4F,"Incremental accumulator differs from full refresh");
        }
    }
    const float a=inc.evaluate_float(p), b=full.evaluate_float(p);
    if (std::abs(a-b)>1e-4F+1e-4F*std::abs(b)) {
        std::cerr << "incremental=" << a << " full=" << b << " error=" << std::abs(a-b)
                  << " max_accumulator_error=" << worst << " sfen=" << p.board().sfen() << '\n';
        throw shogi::Error("Incremental evaluation differs from full refresh");
    }
}
}
int main(int argc,char** argv) {
    try {
        binary::Sha256 hash;
        const std::array<std::uint8_t,3> abc{'a','b','c'}; hash.update(abc);
        check(binary::hex(hash.finish())=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad","SHA256 known answer");
        std::mt19937 rng(50050);
        std::shared_ptr<const nnue::Weights> weights=argc>1 ? nnue::Weights::load(argv[1]) : test_weights(16);
        const auto directory=std::filesystem::temp_directory_path()/
            ("hl0050-native-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(directory);
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() { std::error_code error; std::filesystem::remove_all(path,error); }
        } cleanup{directory};
        const auto weight_path=(directory/"weights.hlnnw1").string();
        weights->save(weight_path);
        auto loaded=nnue::Weights::load(weight_path);
        check(loaded->embedding==weights->embedding && loaded->w3==weights->w3,"Weights round trip");
        {
            std::fstream corrupt(weight_path,std::ios::in|std::ios::out|std::ios::binary);
            corrupt.seekp(104); corrupt.put('x');
        }
        bool refused=false;
        try { nnue::Weights::load(weight_path); } catch (const shogi::Error&) { refused=true; }
        check(refused,"Corrupt weights accepted");
        const auto data_path=(directory/"positions.hlnn1").string();
        nnue::Record record; record.position=shogi::parse_sfen(shogi::start_sfen);
        record.score=-123; record.nodes=123456789; record.result=-1; record.best_move=1; record.game_id=42;
        nnue::DataWriter writer(data_path); writer.append(record); writer.close();
        nnue::DataReader reader(data_path); nnue::Record copy;
        check(reader.next(copy) && nnue::pack(copy)==nnue::pack(record) && !reader.next(copy),"Data round trip");
        {
            std::fstream corrupt(data_path,std::ios::in|std::ios::out|std::ios::binary);
            corrupt.seekp(20); corrupt.put('x');
        }
        refused=false;
        try { nnue::DataReader invalid(data_path); } catch (const shogi::Error&) { refused=true; }
        check(refused,"Corrupt shard accepted");
        nnue::NnueEvaluator inc(weights), full(weights);
        std::uint64_t positions=0, drops=0, promotions=0, captures=0, kings=0;
        float worst=0;
        for (int game_id=0;game_id<1000;++game_id) {
            shogi::Game game({96,1});
            Position p(game.state().position); inc.reset(p);
            std::vector<Undo> stack;
            while (!game.state().outcome.terminal()) {
                const auto moves=game.legal_actions();
                const int code=moves[rng()%moves.size()];
                if (code==shogi::declare) { game.play(code); break; }
                const auto u=p.make(code); inc.push(p,u); game.play(code); stack.push_back(u);
                check(p.board()==game.state().position,"NNUE traversal reference mismatch");
                compare(p,inc,full,worst); ++positions;
                check(inc.evaluate_float(p)==scalar_tail(p,inc,*weights),"SIMD dense tail changed output bits");
                drops+=code>=shogi::drop_base; promotions+=code<shogi::drop_base && (code&1);
                captures+=u.captured!=0; kings+=(u.moved&15)==8;
                // Search-only null transitions change perspective without changing features.
                const auto null=p.make(-1); inc.push(p,null); compare(p,inc,full,worst);
                inc.pop(); p.unmake(null); compare(p,inc,full,worst);
            }
            while (!stack.empty()) {
                const auto u=stack.back(); stack.pop_back(); inc.pop(); p.unmake(u);
                compare(p,inc,full,worst); ++positions;
                check(inc.evaluate_float(p)==scalar_tail(p,inc,*weights),"SIMD dense tail changed output bits");
            }
            check(p.board()==shogi::parse_sfen(shogi::start_sfen),"NNUE unmake failed");
        }
        check(drops && promotions && captures && kings,"Missing incremental transition coverage");
        // Exercise the default width, asymmetric hands, and both perspectives as well.
        auto large=test_weights(256); nnue::NnueEvaluator big(large), fresh(large);
        Position p(shogi::parse_sfen("4k4/9/9/4p4/9/4P4/9/9/4K4 b RBGSLNP2p 1"));
        big.reset(p);
        for (int i=0;i<100;++i) {
            auto legal=p.legal(); if (!legal.size) break;
            const auto u=p.make(legal.values[rng()%legal.size]); big.push(p,u); compare(p,big,fresh,worst);
            big.pop(); p.unmake(u); compare(p,big,fresh,worst);
        }
        // Several unevaluated plies must materialise correctly, including after pop/branch.
        Position lazy_position(shogi::parse_sfen(shogi::start_sfen));
        big.reset(lazy_position);
        std::vector<Undo> delayed;
        for (int i=0;i<60;++i) {
            const auto legal=lazy_position.legal(false);
            if (!legal.size) break;
            delayed.push_back(lazy_position.make(legal.values[rng()%legal.size]));
            big.push(lazy_position,delayed.back());
        }
        compare(lazy_position,big,fresh,worst);
        while (!delayed.empty()) {
            big.pop(); lazy_position.unmake(delayed.back()); delayed.pop_back();
            if (delayed.size()%7==0) compare(lazy_position,big,fresh,worst);
        }
        check(big.evaluate_float(lazy_position)==scalar_tail(lazy_position,big,*large),"Wide SIMD output differs");
        std::cout << "1000 games; make/unmake comparisons=" << positions << " drops=" << drops
                  << " promotions=" << promotions << " captures=" << captures << " king_moves=" << kings
                  << " max_accumulator_error=" << worst << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
