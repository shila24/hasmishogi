// SPDX-License-Identifier: MIT
#include "shogi_search/search.hpp"
#include <algorithm>
#include <cmath>

namespace shogi_search {
void Options::validate() const {
    if (features & ~all_features || hash_mb < 1 || hash_mb > 512 || qdepth < 0 || qdepth > 32)
        throw shogi::Error("Invalid search options");
}
void Limits::validate() const {
    if (depth < 1 || depth > 64 || time_ms < 0 || time_ms > 3600000 || soft_time_ms < 0 || soft_time_ms > 3600000
        || (time_ms && soft_time_ms > time_ms) || multipv < 0 || multipv > 2048)
        throw shogi::Error("Invalid search limits");
}
void BootstrapEvaluator::reset(const Position& p) {
    count_ = 0; material_ = 0;
    const auto& b = p.board();
    for (int sq = 0; sq < 81; ++sq) material_ += square_value(b.board[sq], sq);
    for (int s = 0; s < 2; ++s) for (int h = 0; h < 7; ++h)
        material_ += (s ? -1 : 1) * b.hands[s][h] * piece_value(h+1);
}
void BootstrapEvaluator::push(const Position& p, const Undo& u) {
    stack_[count_++] = material_;
    if (u.target < 0) return;
    const int sign = p.board().turn ? 1 : -1;
    material_ += square_value(p.board().board[u.target], u.target);
    if (u.source >= 0) {
        material_ -= square_value(u.moved, u.source) + square_value(u.captured, u.target);
        if (u.hand >= 0) material_ += sign * piece_value(u.hand+1);
    } else material_ -= sign * piece_value(u.hand+1);
}
void BootstrapEvaluator::pop() noexcept { material_ = stack_[--count_]; }
int BootstrapEvaluator::evaluate(const Position& p) const {
    int score = material_;
    const auto& b = p.board();
    // 8 cp per adjacent friendly king defender, as in the Python bootstrap.
    for (int sq = 0; sq < 81; ++sq) if ((b.board[sq] & 15) == 8) {
        const int side = b.board[sq] >> 4;
        for (int dr = -1; dr <= 1; ++dr) for (int dc = -1; dc <= 1; ++dc) {
            const int r = sq/9+dr, c = sq%9+dc;
            if ((!dr && !dc) || r < 0 || r > 8 || c < 0 || c > 8) continue;
            const int cell = b.board[r*9+c];
            if (cell && cell >> 4 == side) score += side ? -8 : 8;
        }
    }
    return b.turn ? -score : score;
}
Searcher::Searcher(Options options, std::unique_ptr<Evaluator> evaluator) : options_(options), evaluator_(std::move(evaluator)) {
    options_.validate();
    if (!evaluator_) evaluator_ = std::make_unique<BootstrapEvaluator>();
    table_.resize(static_cast<std::size_t>(options_.hash_mb) * 1024 * 1024 / sizeof(Entry));
}
void Searcher::set_evaluator(std::unique_ptr<Evaluator> evaluator) {
    if (!evaluator) throw shogi::Error("Missing evaluator");
    evaluator_ = std::move(evaluator);
}
double Searcher::elapsed() const { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start_).count(); }
bool Searcher::budget() {
    if (aborted_) return false;
    if (stopped_.load(std::memory_order_relaxed) || (limits_.nodes && nodes_ >= limits_.nodes)
        || (limits_.time_ms && elapsed() >= limits_.time_ms)) aborted_ = true;
    return !aborted_;
}
int Searcher::terminal(const shogi::Outcome& o, int side, int ply) const {
    return o.winner == -1 ? 0 : o.winner == side ? mate_score-ply : -mate_score+ply;
}
void Searcher::update_pv(int ply, int move) {
    pv_[ply][0] = move;
    const int children = ply+1 < max_ply ? pv_size_[ply+1] : 0;
    pv_size_[ply] = std::min(max_ply-ply, children+1);
    for (int i = 1; i < pv_size_[ply]; ++i) pv_[ply][i] = pv_[ply+1][i-1];
}
void Searcher::order(Position& p, MoveList& moves, int hint, int ply) {
    if (!enabled(ordering)) return;
    struct Rank { int move; int score; };
    std::array<Rank, 2048> ranked;
    for (int i = 0; i < moves.size; ++i) {
        const int move = moves.values[i], gain = p.gain(move);
        int score = history_[p.board().turn][move];
        if (move == hint) score = 2000000000;
        else if (move == shogi::declare) score = 1900000000;
        else if (gain) {
            const int see = p.see(move);
            score = (see >= 0 ? 1000000 : -1000000) + gain * 16 + see;
        } else if (move == killers_[ply][0]) score = 900000;
        else if (move == killers_[ply][1]) score = 800000;
        ranked[i] = {move,score};
    }
    std::sort(ranked.begin(), ranked.begin()+moves.size, [](const Rank& a, const Rank& b) {
        return a.score != b.score ? a.score > b.score : a.move < b.move;
    });
    for (int i = 0; i < moves.size; ++i) moves.values[i] = ranked[i].move;
}
Searcher::Value Searcher::visit(Position& p, int depth, int alpha, int beta, int ply, int qleft,
                                int extensions, bool synthetic, bool allow_null) {
    pv_size_[ply] = 0;
    if (!budget()) return {0,true};
    ++nodes_;
    const auto repetitions_before = repetitions_;
    auto moves = p.legal(rules_.declaration != 0);
    const auto outcome = synthetic ? (moves.size ? shogi::Outcome{} : shogi::Outcome{1-p.board().turn,p.check()?1:2})
                                   : adjudicate(p,moves,rules_,events_);
    if (outcome.terminal()) {
        const bool repetition = outcome.reason == 4 || outcome.reason == 5;
        if (repetition) ++repetitions_;
        return {terminal(outcome,p.board().turn,ply),repetition};
    }
    if (ply >= max_ply-1) return {evaluator_->evaluate(p),true};
    const bool checked = p.check();
    if (depth > 0 && checked && enabled(check_extension) && extensions < 8) { ++depth; ++extensions; }
    const bool quiet = depth <= 0;
    const int stand = evaluator_->evaluate(p);
    if (quiet && !enabled(quiescence)) return {stand,false};
    if (quiet && qleft <= 0 && !checked) return {stand,false};
    int best = -infinity, best_move = -1;
    bool dependent = false;
    const int original_alpha = alpha;
    if (quiet && !checked) {
        best = stand;
        if (best >= beta) return {best,false};
        alpha = std::max(alpha,best);
    }
    // Remaining rule plies are part of the table key. Repetition is checked first.
    const auto remaining = rules_.max_moves ? static_cast<std::uint64_t>(rules_.max_moves) - (events_.size()-1) : 0;
    const auto key = p.key() ^ (remaining * 0x9e3779b97f4a7c15ULL);
    auto& entry = table_[key % table_.size()];
    int hint = -1;
    if (!quiet && !synthetic && enabled(transposition) && entry.depth >= 0 && entry.key == key) {
        hint = entry.move;
        int score = entry.score;
        if (score > mate_score-max_ply) score -= ply;
        if (score < -mate_score+max_ply) score += ply;
        if (beta == alpha+1 && entry.depth >= depth && (entry.bound == 0 || (entry.bound == 1 && score >= beta) || (entry.bound == 2 && score <= alpha))) {
            // Retain a legal prefix; no stale or history-incompatible move is published.
            if (std::find(moves.begin(),moves.end(),hint) != moves.end()) { pv_[ply][0] = hint; pv_size_[ply] = 1; }
            return {score,false};
        }
    }
    const bool narrow = beta == alpha+1;
    if (!quiet && !checked && narrow && std::abs(beta) < mate_score-max_ply) {
        if (enabled(razoring) && depth == 1 && stand + 400 < alpha) {
            auto v = visit(p,0,alpha,beta,ply,qleft,extensions,synthetic,false);
            if (v.score < alpha || aborted_) return v;
        }
        bool nonpawn = false;
        for (int c : p.board().board) if (c && c >> 4 == p.board().turn && (c & 15) > 1 && (c & 15) != 8) nonpawn = true;
        // Null branches never enter real history or reuse/store table bounds.
        if (enabled(null_move) && allow_null && !synthetic && depth >= 3 && stand >= beta && nonpawn) {
            const auto u = p.make(-1); evaluator_->push(p,u);
            auto v = visit(p,depth-3,-beta,-beta+1,ply+1,qleft,extensions,true,false);
            evaluator_->pop(); p.unmake(u);
            if (aborted_) return {0,true};
            if (-v.score >= beta && std::abs(v.score) < mate_score-max_ply) return {-v.score,true};
        }
    }
    order(p,moves,hint,ply);
    int searched = 0;
    for (int move : moves) {
        if (!budget()) return {0,true};
        const int gain = p.gain(move);
        const bool tactical = gain != 0 || move == shogi::declare;
        if (quiet && !checked && !tactical) continue;
        if (quiet && !checked && enabled(see_pruning) && gain && move != shogi::declare && p.see(move) < 0) continue;
        if (move == shogi::declare) {
            const int score = mate_score-ply-1;
            if (score > best) { best = score; best_move = move; pv_size_[ply] = 1; pv_[ply][0] = move; }
            alpha = std::max(alpha,best);
            if (alpha >= beta) break;
            ++searched; continue;
        }
        const int side = p.board().turn;
        const auto u = p.make(move);
        evaluator_->push(p,u);
        const bool gives_check = p.check();
        if (!synthetic) events_.push_back({p.key(),p.board().turn,gives_check});
        const bool prune = !quiet && !checked && !gives_check && !tactical && searched > 0 && narrow
            && depth <= 2 && enabled(futility) && stand + 180 * depth <= alpha;
        Value v{0,false};
        if (!prune) {
            const int child_depth = quiet ? 0 : depth-1;
            const bool reduce = enabled(lmr) && !quiet && depth >= 3 && searched >= 4 && !checked && !gives_check && !tactical;
            if (reduce) {
                v = visit(p,child_depth-1,-alpha-1,-alpha,ply+1,quiet ? qleft-1 : qleft,extensions,synthetic,true);
                v.score = -v.score;
            }
            if (!reduce || v.score > alpha) {
                if (enabled(pvs) && searched > 0) {
                    v = visit(p,child_depth,-alpha-1,-alpha,ply+1,quiet ? qleft-1 : qleft,extensions,synthetic,true);
                    v.score = -v.score;
                    if (v.score > alpha && v.score < beta && !aborted_) {
                        auto full = visit(p,child_depth,-beta,-alpha,ply+1,quiet ? qleft-1 : qleft,extensions,synthetic,true);
                        v.path_dependent = v.path_dependent || full.path_dependent; v.score = -full.score;
                    }
                } else {
                    v = visit(p,child_depth,-beta,-alpha,ply+1,quiet ? qleft-1 : qleft,extensions,synthetic,true);
                    v.score = -v.score;
                }
            }
        }
        if (!synthetic) events_.pop_back();
        evaluator_->pop(); p.unmake(u);
        if (aborted_) return {0,true};
        if (prune) continue;
        ++searched; dependent = dependent || v.path_dependent;
        if (v.score > best) { best = v.score; best_move = move; update_pv(ply,move); }
        alpha = std::max(alpha,best);
        if (alpha >= beta) {
            if (!tactical && enabled(ordering)) {
                killers_[ply][1] = killers_[ply][0]; killers_[ply][0] = move;
                auto& h = history_[side][move]; h = std::min(700000,h+depth*depth);
            }
            break;
        }
    }
    dependent = dependent || repetitions_ != repetitions_before;
    if (!quiet && !synthetic && !dependent && enabled(transposition)) {
        int score = best;
        if (score > mate_score-max_ply) score += ply;
        if (score < -mate_score+max_ply) score -= ply;
        if (entry.age != age_ || entry.depth <= depth || entry.key == key)
            entry = {key,score,best_move,static_cast<std::int16_t>(depth),static_cast<std::uint8_t>(best <= original_alpha ? 2 : best >= beta ? 1 : 0),age_};
    }
    return {best,dependent};
}
bool Searcher::prove_mate(Position& p, int remaining, int attacker) {
    if (!budget()) return false;
    ++nodes_;
    const auto moves = p.legal(rules_.declaration != 0);
    const auto outcome = adjudicate(p,moves,rules_,events_);
    if (outcome.terminal()) return outcome.reason == 1 && outcome.winner == attacker;
    if (!remaining) return false;
    const bool attack = p.board().turn == attacker;
    for (int move : moves) {
        if (move == shogi::declare) { if (!attack) return false; continue; }
        const auto undo = p.make(move);
        events_.push_back({p.key(),p.board().turn,p.check()});
        const bool proved = prove_mate(p,remaining-1,attacker);
        events_.pop_back(); p.unmake(undo);
        if (aborted_) return false;
        if (attack && proved) return true;
        if (!attack && !proved) return false;
    }
    return !attack;
}
void Searcher::clear_ordering_state() {
    std::fill(table_.begin(),table_.end(),Entry{}); ++age_;
    for (auto& row : killers_) row.fill(-1);
    for (auto& row : history_) row.fill(0);
}
Result Searcher::go(const Root& root, const Limits& limits, const Progress& progress) {
    limits.validate(); limits_ = limits; rules_ = root.rules;
    start_ = std::chrono::steady_clock::now(); nodes_ = 0; repetitions_ = 0; aborted_ = false;
    // A fresh table/heuristics per request makes fixed-node runs independent of previous calls.
    clear_ordering_state();
    events_ = root.history; events_.reserve(events_.size()+max_ply+2);
    Position position(root.position); evaluator_->reset(position);
    auto legal = position.legal(rules_.declaration != 0);
    if (root.outcome.terminal()) throw shogi::Error("No legal action is available");
    if (!limits.root_moves.empty()) {
        MoveList selected;
        for (int move : limits.root_moves) {
            if (std::find(legal.begin(),legal.end(),move) == legal.end()) throw shogi::Error("Illegal root move");
            if (std::find(selected.begin(),selected.end(),move) != selected.end()) throw shogi::Error("Duplicate root move");
            selected.add(move);
        }
        legal = selected;
    }
    if (!legal.size) throw shogi::Error("No legal action is available");
    if (root.history.size()-1 >= shogi::max_game_moves) throw shogi::Error("Move limit exceeded");
    Result result; result.move = legal.values[0]; result.pv = {result.move};
    // Comparable root scores must not depend on other candidates, their order,
    // or previous iterations: TT hints/bounds, killers and history affect LMR.
    // Restricted-root requests use this same full-window scoring procedure.
    const bool all = limits.multipv != 1 || !limits.root_moves.empty();
    for (int depth = 1; depth <= limits.depth && budget(); ++depth) {
        int width = enabled(aspiration) && depth >= 3 && !all ? 40 : infinity;
        std::vector<RootScore> rows;
        while (true) {
            rows.clear();
            const int low = width == infinity ? -infinity : std::max(-infinity,result.score-width);
            const int high = width == infinity ? infinity : std::min(infinity,result.score+width);
            int alpha = low;
            for (int move : legal) {
                if (!budget()) break;
                RootScore row{move,0,depth,{move}};
                if (move == shogi::declare) { ++nodes_; row.score = mate_score-1; }
                else {
                    if (all) clear_ordering_state();
                    const auto u = position.make(move); evaluator_->push(position,u);
                    events_.push_back({position.key(),position.board().turn,position.check()});
                    auto v = visit(position,depth-1,all ? -infinity : -high,all ? infinity : -alpha,1,options_.qdepth,0,false,true);
                    row.score = -v.score;
                    row.pv.insert(row.pv.end(),pv_[1].begin(),pv_[1].begin()+pv_size_[1]);
                    events_.pop_back(); evaluator_->pop(); position.unmake(u);
                }
                if (aborted_) break;
                rows.push_back(std::move(row));
                alpha = std::max(alpha,rows.back().score);
                if (!all && alpha >= high) break;
            }
            if (aborted_) break;
            std::stable_sort(rows.begin(),rows.end(),[](const RootScore& a,const RootScore& b){return a.score > b.score;});
            if (!all && width != infinity && (rows.front().score <= low || rows.front().score >= high)) {
                width = width >= 640 ? infinity : width*4;
                continue;
            }
            break;
        }
        if (aborted_) break;
        result.move = rows.front().move; result.score = rows.front().score; result.pv = rows.front().pv;
        result.depth = depth;
        if (limits.multipv > 0 && rows.size() > static_cast<std::size_t>(limits.multipv)) rows.resize(limits.multipv);
        result.roots = std::move(rows);
        result.nodes = nodes_; result.elapsed_ms = elapsed(); result.completed = depth == limits.depth;
        if (progress) progress(result);
        if (!all && enabled(ordering)) {
            auto it = std::find(legal.begin(),legal.end(),result.move); std::rotate(legal.begin(),it,it+1);
        }
        if (limits.soft_time_ms && elapsed() >= limits.soft_time_ms) break;
    }
    // Win-distance scores also encode declaration, perpetual-check and no-reply
    // wins. Publish a mate claim only after a separate exhaustive checkmate proof.
    // This uses the same node/time/stop budgets; lack of budget means "unproven".
    const int distance = mate_score-std::abs(result.score);
    if (distance > 0 && distance < max_ply-1 && result.move != shogi::declare && budget()) {
        const int attacker = result.score > 0 ? position.board().turn : 1-position.board().turn;
        bool proved = false;
        if (result.score > 0) {
            const auto undo = position.make(result.move);
            events_.push_back({position.key(),position.board().turn,position.check()});
            proved = prove_mate(position,distance-1,attacker);
            events_.pop_back(); position.unmake(undo);
        } else proved = prove_mate(position,distance,attacker);
        if (proved) result.mate_plies = result.score > 0 ? distance : -distance;
    }
    result.nodes = nodes_; result.elapsed_ms = elapsed(); result.completed = result.depth == limits.depth;
    result.stopped = stopped_.load(std::memory_order_relaxed);
    return result;
}
}
