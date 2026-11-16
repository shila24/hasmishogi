// SPDX-License-Identifier: MIT
#include "shogi_search/search.hpp"
#include "shogi_search/c_api.h"
#include <algorithm>
#include <barrier>
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace {
using namespace shogi_search;
void require(bool ok,const char* text) { if (!ok) throw shogi::Error(text); }
int minimax(shogi::Game& game,int depth,int ply) {
    const auto& state = game.state();
    if (state.outcome.terminal()) {
        const int winner = state.outcome.winner;
        return winner == -1 ? 0 : winner == state.position.turn ? mate_score-ply : -mate_score+ply;
    }
    if (!depth) { BootstrapEvaluator eval; Position p(state.position); eval.reset(p); return eval.evaluate(p); }
    int best = -infinity;
    const int actor = state.position.turn;
    for (int move : game.legal_actions()) {
        game.play(move);
        int score = minimax(game,depth-1,ply+1);
        if (game.state().position.turn != actor) score = -score;
        game.undo(); best = std::max(best,score);
    }
    return best;
}
void exact_search() {
    for (const char* sfen : {"4k4/9/5g3/9/9/9/3G5/9/4K4 b - 1", "k8/9/1GN6/1R7/9/9/9/9/4K4 b - 1"}) {
        shogi::Game game({0,1},sfen);
        for (int depth = 1; depth <= 3; ++depth) {
            const int expected = minimax(game,depth,0);
            for (unsigned features : {0U,unsigned(pvs),unsigned(aspiration),unsigned(transposition),unsigned(ordering),unsigned(pvs|aspiration|transposition|ordering)}) {
                Searcher search({features,1,0}); Limits limits; limits.depth = depth;
                auto result = search.go(Root(game),limits);
                require(result.completed && result.score == expected,"Independent minimax mismatch");
            }
        }
    }
    shogi::Game game({0,1},"4k4/9/5g3/9/9/9/3G5/9/4K4 b - 1");
    for (unsigned features : {0U,unsigned(all_features)}) {
        Searcher search({features,1,8}); Limits limits; limits.depth = 2; limits.multipv = 0;
        auto result = search.go(Root(game),limits);
        require(result.roots.size() == game.legal_actions().size(),"Missing root scores");
        for (const auto& row : result.roots) {
            limits.multipv = 1; limits.root_moves = {row.move};
            auto single = search.go(Root(game),limits);
            require(single.score == row.score && row.depth == 2,"MultiPV/single-root mismatch");
        }
    }
}
void budgets() {
    shogi::Game game;
    Searcher search({all_features,1,8}); Limits limits; limits.depth = 4;
    const auto a = search.go(Root(game),limits), b = search.go(Root(game),limits);
    require(a.completed && a.move == b.move && a.score == b.score && a.nodes == b.nodes && a.pv == b.pv,"Depth determinism");
    limits.depth = 64; limits.nodes = 777;
    const auto c = search.go(Root(game),limits), d = search.go(Root(game),limits);
    require(c.nodes == 777 && c.move == d.move && c.score == d.score && c.pv == d.pv && c.depth == d.depth,"Node determinism");
    search.stop(); auto stopped = search.go(Root(game),limits);
    require(stopped.stopped && stopped.nodes == 0 && stopped.depth == 0 && game.is_legal(stopped.move),"Pre-set stop");
    search.clear_stop(); limits.nodes = 0; limits.time_ms = 1;
    auto timed = search.go(Root(game),limits);
    require(!timed.completed && game.is_legal(timed.move),"Time budget");
}
void comparable_roots() {
    for (const auto& [sfen, depth] : std::array{
            std::pair{shogi::start_sfen,5},
            std::pair{std::string_view("lnsgkgsn1/2r4bl/p1pp2ppp/1p2pp3/9/4P4/PPPP1PPPP/1B1KR1S1L/LNSG1G1N1 b - 11"),4}}) {
        const shogi::Game game({},sfen);
        const Root root(game);
        Searcher search({all_features,1,12});
        Limits limits; limits.depth = depth; limits.multipv = 0;
        const auto all = search.go(root,limits);
        require(all.completed && all.roots.size() == game.legal_actions().size(),"Deep MultiPV incomplete");
        for (const auto& row : all.roots) {
            limits.multipv = 1; limits.root_moves = {row.move};
            const auto single = search.go(root,limits);
            require(single.completed && row.score == single.score && row.pv == single.pv,
                    "Selective MultiPV/single-root mismatch");
        }
        limits.multipv = 0; limits.root_moves = game.legal_actions();
        std::reverse(limits.root_moves.begin(),limits.root_moves.end());
        const auto reversed = search.go(root,limits);
        require(reversed.completed,"Reversed MultiPV incomplete");
        for (const auto& row : all.roots) {
            const auto found = std::find_if(reversed.roots.begin(),reversed.roots.end(),
                [&](const RootScore& other) { return other.move == row.move; });
            require(found != reversed.roots.end() && found->score == row.score && found->pv == row.pv,
                    "Root scores depend on candidate order");
        }
    }
}
void progress_results() {
    Searcher search({all_features,1,12}); Limits limits; limits.depth = 64;
    const Root root(shogi::Game{});
    std::vector<Result> iterations;
    const auto result = search.go(root,limits,[&](const Result& completed) {
        iterations.push_back(completed);
        if (completed.depth == 3) search.stop();
    });
    require(iterations.size() == 3 && result.stopped && !result.completed,"Progress stop");
    for (std::size_t i = 0; i < iterations.size(); ++i) {
        require(iterations[i].depth == static_cast<int>(i)+1 && iterations[i].nodes > 0,
                "Only completed iterations reported");
        if (i) require(iterations[i].nodes > iterations[i-1].nodes,"Progress cumulative nodes");
    }
    const auto& last = iterations.back();
    require(result.move == last.move && result.score == last.score && result.pv == last.pv
            && result.nodes == last.nodes && result.depth == last.depth,"Stopped result preserves progress");
}
void concurrency() {
    std::barrier start(4);
    std::array<std::future<Result>,4> jobs;
    for (auto& job : jobs) job = std::async(std::launch::async,[&] {
        Searcher search({all_features,1,8}); shogi::Game game;
        Limits limits; limits.depth = 64; limits.nodes = 2000;
        start.arrive_and_wait(); return search.go(Root(game),limits);
    });
    auto expected = jobs[0].get();
    for (int i = 1; i < 4; ++i) {
        auto actual = jobs[i].get();
        require(actual.move == expected.move && actual.score == expected.score && actual.nodes == expected.nodes && actual.pv == expected.pv,"Concurrent independent searches");
    }
}
struct Signal {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered{}, released{};
};
class SignallingEvaluator final : public Evaluator {
public:
    explicit SignallingEvaluator(Signal& signal) : signal_(signal) {}
    void reset(const Position& p) override { eval_.reset(p); }
    void push(const Position& p,const Undo& u) override { eval_.push(p,u); }
    void pop() noexcept override { eval_.pop(); }
    int evaluate(const Position& p) const override {
        std::unique_lock lock(signal_.mutex);
        if (!signal_.entered) {
            signal_.entered = true; signal_.changed.notify_all();
            signal_.changed.wait(lock,[&]{return signal_.released;});
        }
        return eval_.evaluate(p);
    }
private:
    Signal& signal_;
    BootstrapEvaluator eval_;
};
void external_stop() {
    Signal signal;
    Searcher search({all_features,1,8},std::make_unique<SignallingEvaluator>(signal));
    auto job = std::async(std::launch::async,[&] { Limits l; l.depth = 64; return search.go(Root(shogi::Game{}),l); });
    {
        std::unique_lock lock(signal.mutex);
        signal.changed.wait(lock,[&]{return signal.entered;});
        search.stop(); signal.released = true; signal.changed.notify_all();
    }
    const auto result = job.get();
    require(result.stopped && !result.completed,"External atomic stop");
}
int move(const char* text) {
    const auto sq = [](const char* s){return (s[1]-'a')*9+'9'-s[0];};
    return (sq(text)*81+sq(text+2))*2;
}
void history_rules() {
    for (const auto* sfen : {"4k4/9/9/9/9/9/9/9/4K4 b - 1", "4k4/9/5R3/9/9/9/9/9/4K4 b - 1"}) {
        const bool check = std::string(sfen).find('R') != std::string::npos;
        const std::array<const char*,4> cycle = check ? std::array{"4c5c","5a4a","5c4c","4a5a"} : std::array{"5i6i","5a6a","6i5i","6a5a"};
        shogi::Game game({12,1},sfen);
        for (int i = 0; i < 11; ++i) game.play(move(cycle[i%4]));
        Root root(game);
        auto p = root.position;
        p.make(move(cycle[3])); root.history.push_back({p.key(),p.board().turn,p.check()});
        auto outcome = adjudicate(p,p.legal(),root.rules,root.history);
        game.play(move(cycle[3]));
        require(outcome == game.state().outcome,"Repetition/perpetual check precedence");
        game.undo(); Limits l; l.depth = 1; l.root_moves = {move(cycle[3])};
        Searcher search({0,1,0}); auto result = search.go(Root(game),l);
        require(result.score == (check ? mate_score-1 : 0),"History passed into search");
    }
    shogi::Game cap({1,1}); Searcher search({0,1,0}); Limits l; l.depth = 2;
    require(search.go(Root(cap),l).score == 0,"max_moves draw");
    shogi::Game mate({1,1},"k8/9/1GN6/1R7/9/9/9/9/4K4 b - 1");
    require(search.go(Root(mate),l).score == mate_score-1,"Mate before move cap");
    std::ifstream input(std::string(SS_SOURCE_ROOT)+"/cpp/shogi_search/tests/data/parity-v1.tsv");
    std::string sfen; int declarations = 0;
    while (std::getline(input,sfen)) if (!sfen.empty() && sfen[0] != '#') {
        shogi::Game game({0,1},sfen);
        if (!game.can_declare()) continue;
        ++declarations;
        const auto result = search.go(Root(game),l);
        require(result.move == shogi::declare && result.score == mate_score-1,"Declaration win");
        Position p(game.state().position); const auto before = p.board(); const auto key = p.key();
        const auto u = p.make(shogi::declare); p.unmake(u);
        require(p.board() == before && p.key() == key,"Declaration undo");
    }
    require(declarations >= 2,"Both declaration sides covered");
}
void see_cases() {
    Position free(shogi::parse_sfen("k8/9/4p4/9/4R4/9/9/9/4K4 b - 1"));
    require(free.see(move("5e5c")) == 200,"Free pawn capture SEE");
    require(free.see(move("5e5c")+1) == 400,"Capture promotion SEE");
    Position defended(shogi::parse_sfen("k8/9/4pr3/9/4R4/9/9/9/4K4 b - 1"));
    require(defended.see(move("5e5c")) == -1800,"Losing rook exchange SEE");
    Position xray(shogi::parse_sfen("k8/9/4pr3/9/4R4/9/4L4/9/4K4 b - 1"));
    require(xray.see(move("5e5c")) == 200,"X-ray recapture SEE");
}
void abi_errors() {
    require(ss_abi_version() == 1 && ss_sizeof_limits() == sizeof(ss_limits),"ABI layout");
    require(ss_go(nullptr,nullptr,nullptr,0,nullptr) == -1,"Null ABI rejected");
    ss_options options{SS_ALL_FEATURES,1,8};
    std::unique_ptr<ss_handle,decltype(&ss_destroy)> h(ss_create(&options),ss_destroy);
    require(h != nullptr,"ABI create");
    require(ss_set_position(h.get(),nullptr,320,1,nullptr,0,0) == 0,"ABI set");
    ss_state before{}, after{}; ss_get_state(h.get(),&before);
    int32_t invalid = -1;
    require(ss_set_position(h.get(),nullptr,320,1,&invalid,1,0) == -1,"History validation");
    ss_get_state(h.get(),&after); require(before.key == after.key && before.ply == after.ply,"Failure atomicity");
    int32_t sentinel = 123;
    require(ss_legal_moves(h.get(),&sentinel,1) == -1 && sentinel == 123,"Buffer atomicity");
    require(ss_legal_moves(h.get(),nullptr,-1) == -1,"Negative capacity");
    require(ss_stop(nullptr) == -1,"Null stop rejected");
    struct ProgressState { int depth{}; bool valid{true}; } progress;
    const auto observer = [](const ss_result* result,const int32_t* pv,int32_t count,void* data) -> int32_t {
        auto& state = *static_cast<ProgressState*>(data);
        state.valid = state.valid && result->depth == state.depth+1 && result->nodes > 0
            && count > 0 && pv && pv[0] == result->move && !result->mate_plies;
        state.depth = result->depth;
        return result->depth == 3;
    };
    ss_limits limits{0,64,0,0,1}; ss_result result{};
    require(ss_go_with_progress(h.get(),&limits,nullptr,0,&result,observer,&progress) == 0,"ABI progress");
    require(progress.valid && progress.depth == 3 && result.depth == 3 && result.stopped,"ABI progress stop");
    require(ss_go_with_progress(nullptr,nullptr,nullptr,0,nullptr,observer,&progress) == -1,"Null progress ABI rejected");
}
}
int main(int argc,char**) {
    try {
        concurrency(); external_stop();
        if (argc == 1) { exact_search(); comparable_roots(); progress_results(); budgets(); history_rules(); see_cases(); abi_errors(); }
        std::cout << "search contracts passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
