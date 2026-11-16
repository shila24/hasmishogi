// SPDX-License-Identifier: MIT
#include "shogi_search/c_api.h"
#include "shogi_search/search.hpp"
#include "shogi_search/nnue.hpp"
#include "shogi_search/data.hpp"
#include <algorithm>
#include <cstdio>
#include <memory>

struct ss_handle {
    shogi_search::Searcher search;
    std::unique_ptr<shogi_search::Root> root;
    shogi_search::Result result;
    std::shared_ptr<const shogi_search::nnue::Weights> weights;
    std::string evaluator{"bootstrap-v1"};
    explicit ss_handle(shogi_search::Options options) : search(options) {}
};
namespace {
thread_local char error[512]{};
void failure(const char* text) noexcept { std::snprintf(error,sizeof(error),"%s",text); }
template<class F> int32_t guard(F&& f) noexcept {
    error[0] = '\0';
    try { return f(); }
    catch (const std::exception& e) { failure(e.what()); }
    catch (...) { failure("Unknown native search error"); }
    return -1;
}
void require(bool condition, const char* message) { if (!condition) throw shogi::Error(message); }
const shogi_search::Root& root(const ss_handle* h) {
    require(h && h->root,"Search position is not set"); return *h->root;
}
template<class T> void buffer(T* out, int32_t capacity, std::size_t size) {
    require(capacity >= 0,"Invalid buffer capacity");
    if (!out && capacity == 0) return;
    require(out && static_cast<std::size_t>(capacity) >= size,"Output buffer is too small");
}
ss_result result_view(const shogi_search::Result& result) {
    return {result.nodes,result.elapsed_ms,result.move,result.score,result.depth,result.completed,
            result.stopped,static_cast<int32_t>(result.roots.size()),result.mate_plies};
}
}
extern "C" {
int32_t ss_abi_version(void) { return SS_ABI_VERSION; }
size_t ss_sizeof_options(void) { return sizeof(ss_options); }
size_t ss_sizeof_limits(void) { return sizeof(ss_limits); }
size_t ss_sizeof_result(void) { return sizeof(ss_result); }
size_t ss_sizeof_root_score(void) { return sizeof(ss_root_score); }
size_t ss_sizeof_state(void) { return sizeof(ss_state); }
const char* ss_last_error(void) { return error; }
ss_handle* ss_create(const ss_options* o) {
    ss_handle* result = nullptr;
    guard([&] { const shogi_search::Options options = o ? shogi_search::Options{o->features,o->hash_mb,o->qdepth} : shogi_search::Options{};
        result = std::make_unique<ss_handle>(options).release(); return 0; });
    return result;
}
void ss_destroy(ss_handle* h) { guard([&] { const std::unique_ptr<ss_handle> owned(h); return 0; }); }
int32_t ss_set_position(ss_handle* h,const char* sfen,int32_t cap,int32_t declaration,const int32_t* moves,int32_t count,int32_t resigned) {
    return guard([&] {
        require(h,"Invalid search handle");
        require(count >= 0 && count <= shogi::max_game_moves && (!count || moves),"Invalid history buffer");
        require(resigned == 0 || resigned == 1,"Invalid resignation flag");
        shogi::Game game({cap,declaration},sfen ? std::string_view(sfen) : shogi::start_sfen);
        for (int i = 0; i < count; ++i) game.play(moves[i]);
        if (resigned) game.resign();
        auto next = std::make_unique<shogi_search::Root>(game);
        h->root = std::move(next); h->result = {}; return 0;
    });
}
int32_t ss_get_state(const ss_handle* h,ss_state* out) {
    return guard([&] { const auto& r = root(h); require(out,"Null state output");
        *out = {r.position.key(),r.position.board().turn,static_cast<int32_t>(r.history.size()-1),r.outcome.winner,r.outcome.reason}; return 0; });
}
int32_t ss_legal_moves(ss_handle* h,int32_t* out,int32_t capacity) {
    return guard([&] { const auto& r = root(h); auto p = r.position;
        auto moves = p.legal(r.rules.declaration != 0); if (r.outcome.terminal()) moves.size = 0;
        buffer(out,capacity,moves.size); if (out) std::copy(moves.begin(),moves.end(),out); return moves.size; });
}
int64_t ss_perft(ss_handle* h,int32_t depth) {
    int64_t result = -1;
    guard([&] { const auto& r = root(h); require(depth >= 0 && depth <= 8,"Invalid perft depth");
        auto p = r.position; result = depth && r.outcome.terminal() ? 0 : shogi_search::perft(p,depth,r.rules.declaration != 0); return 0; });
    return result;
}
int32_t ss_evaluate(ss_handle* h,int32_t* out) {
    return guard([&] { const auto& r = root(h); require(out,"Null score output");
        if (h->weights) {
            shogi_search::nnue::NnueEvaluator eval(h->weights); eval.reset(r.position); *out = eval.evaluate(r.position);
        } else {
            shogi_search::BootstrapEvaluator eval; eval.reset(r.position); *out = eval.evaluate(r.position);
        }
        return 0; });
}
int32_t ss_load_nnue(ss_handle* h,const char* path) {
    return guard([&] {
        require(h,"Invalid search handle");
        auto weights = path && *path ? shogi_search::nnue::Weights::load(path) : nullptr;
        std::unique_ptr<shogi_search::Evaluator> evaluator;
        if (weights) evaluator = std::make_unique<shogi_search::nnue::NnueEvaluator>(weights);
        else evaluator = std::make_unique<shogi_search::BootstrapEvaluator>();
        std::string name = weights ? weights->architecture() : "bootstrap-v1";
        h->search.set_evaluator(std::move(evaluator));
        h->weights = std::move(weights); h->evaluator = std::move(name); h->result = {}; return 0;
    });
}
const char* ss_evaluator(const ss_handle* h) {
    const char* name=nullptr;
    guard([&] { require(h,"Invalid search handle"); name=h->evaluator.c_str(); return 0; }); return name;
}
int32_t ss_evaluate_float(ss_handle* h,float* out) {
    return guard([&] { const auto& r=root(h); require(out,"Null score output");
        if (h->weights) {
            shogi_search::nnue::NnueEvaluator eval(h->weights); eval.reset(r.position); *out=eval.evaluate_float(r.position);
        } else {
            shogi_search::BootstrapEvaluator eval; eval.reset(r.position); *out=static_cast<float>(eval.evaluate(r.position));
        }
        return 0; });
}
int32_t ss_nnue_features(ss_handle* h,int32_t side,int32_t* out,int32_t capacity) {
    return guard([&] { const auto& r=root(h); const auto f=shogi_search::nnue::features(r.position.board(),side);
        buffer(out,capacity,f.size); if (out) std::copy_n(f.indices.begin(),f.size,out); return f.size; });
}
int32_t ss_save_nnue(const ss_handle* h,const char* path) {
    return guard([&] { require(h && h->weights && path && *path,"Missing NNUE weights/path"); h->weights->save(path); return 0; });
}
int32_t ss_data_roundtrip(const char* input,const char* output) {
    return guard([&] { require(input && output,"Null hlnn1 path");
        shogi_search::nnue::DataReader reader(input); shogi_search::nnue::DataWriter writer(output);
        shogi_search::nnue::Record record;
        while (reader.next(record)) writer.append(record);
        writer.close(); return 0; });
}
int32_t ss_nnue_encode_records(const uint8_t* records,size_t bytes,int32_t* out,size_t capacity) {
    return guard([&] {
        using namespace shogi_search::nnue;
        require(bytes % record_size == 0,"Partial hlnn1 record");
        const auto count=bytes/record_size;
        require(count<=1000000 && capacity>=count*2*max_active,"Invalid feature batch capacity");
        require(!count || (records && out),"Null feature batch buffer");
        for (std::size_t row=0;row<count;++row) {
            const auto record=unpack(std::span(records+row*record_size,record_size));
            for (int order=0;order<2;++order) {
                const auto f=features(record.position,record.position.turn^order);
                auto* destination=out+(row*2+order)*max_active;
                std::fill_n(destination,max_active,-1);
                std::copy_n(f.indices.begin(),f.size,destination);
            }
        }
        return 0;
    });
}
int32_t ss_see(ss_handle* h,int32_t move,int32_t* out) {
    return guard([&] { const auto& r = root(h); require(out,"Null score output"); auto p = r.position;
        const auto legal = p.legal(r.rules.declaration != 0);
        require(!r.outcome.terminal() && std::find(legal.begin(),legal.end(),move) != legal.end(),"Illegal move");
        *out = p.see(move); return 0; });
}
int32_t ss_go(ss_handle* h,const ss_limits* l,const int32_t* moves,int32_t count,ss_result* out) {
    return ss_go_with_progress(h,l,moves,count,out,nullptr,nullptr);
}
int32_t ss_go_with_progress(ss_handle* h,const ss_limits* l,const int32_t* moves,int32_t count,
                            ss_result* out,ss_progress progress,void* user_data) {
    return guard([&] { const auto& r = root(h); require(l && out,"Null search limits or result");
        require(count >= 0 && count <= 2048 && (!count || moves),"Invalid root move buffer");
        shogi_search::Limits limits{l->depth,l->nodes,l->time_ms,l->soft_time_ms,l->multipv,{}};
        if (count) limits.root_moves.assign(moves,moves+count);
        shogi_search::Searcher::Progress observer;
        if (progress) observer = [&](const shogi_search::Result& iteration) {
            const auto view = result_view(iteration);
            const std::vector<int32_t> pv(iteration.pv.begin(),iteration.pv.end());
            if (progress(&view,pv.data(),static_cast<int32_t>(pv.size()),user_data)) h->search.stop();
        };
        auto result = h->search.go(r,limits,observer);
        *out = result_view(result);
        h->result = std::move(result); return 0; });
}
int32_t ss_root_scores(const ss_handle* h,ss_root_score* out,int32_t capacity) {
    return guard([&] { root(h); const auto& rows = h->result.roots; buffer(out,capacity,rows.size());
        if (out) for (std::size_t i = 0; i < rows.size(); ++i) out[i] = {rows[i].move,rows[i].score,rows[i].depth};
        return static_cast<int32_t>(rows.size()); });
}
int32_t ss_pv(const ss_handle* h,int32_t index,int32_t* out,int32_t capacity) {
    return guard([&] { root(h); require(index >= -1 && (index == -1 || static_cast<std::size_t>(index) < h->result.roots.size()),"Invalid PV index");
        const auto& pv = index == -1 ? h->result.pv : h->result.roots[index].pv;
        buffer(out,capacity,pv.size()); if (out) std::copy(pv.begin(),pv.end(),out); return static_cast<int32_t>(pv.size()); });
}
int32_t ss_stop(ss_handle* h) { return guard([&] { require(h,"Invalid search handle"); h->search.stop(); return 0; }); }
int32_t ss_clear_stop(ss_handle* h) { return guard([&] { require(h,"Invalid search handle"); h->search.clear_stop(); return 0; }); }
}
