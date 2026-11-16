// SPDX-License-Identifier: MIT
// Complete-game self-play. Commit markers are the recovery authority.
#include "shogi_search/data.hpp"
#include "shogi_search/nnue.hpp"
#include "shogi_search/binary.hpp"
#include <atomic>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace ss = shogi_search;
namespace {
static_assert(std::atomic<bool>::is_always_lock_free);
std::atomic<bool> stopping{false};
void stop_signal(int) { stopping.store(true, std::memory_order_relaxed); }
class RunLock {
public:
    explicit RunLock(const fs::path& directory) {
#ifndef _WIN32
        descriptor_=::open((directory/"run.lock").c_str(),O_CREAT|O_RDWR,0600);
        if (descriptor_<0) throw shogi::Error("Cannot open generator lock");
        if (::flock(descriptor_,LOCK_EX|LOCK_NB)) {
            ::close(descriptor_); descriptor_=-1;
            throw shogi::Error("Generator output is locked");
        }
#else
        (void)directory;
#endif
    }
    ~RunLock() {
#ifndef _WIN32
        if (descriptor_>=0) ::close(descriptor_);
#endif
    }
    RunLock(const RunLock&)=delete;
    RunLock& operator=(const RunLock&)=delete;
    RunLock(RunLock&&)=delete;
    RunLock& operator=(RunLock&&)=delete;
private:
#ifndef _WIN32
    int descriptor_{-1};
#endif
};
void sync(const fs::path& path, bool directory = false) {
#ifndef _WIN32
    const int fd = ::open(path.c_str(), O_RDONLY | (directory ? O_DIRECTORY : 0));
    if (fd < 0) throw shogi::Error("Cannot open for fsync: " + path.string());
    const int result = ::fsync(fd);
    ::close(fd);
    if (result) throw shogi::Error("fsync failed: " + path.string());
#else
    (void)path; (void)directory;
#endif
}
void atomic_text(const fs::path& path, const std::string& text) {
    const auto temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    out << text;
    out.flush();
    if (!out) throw shogi::Error("Cannot write " + temporary);
    out.close(); sync(temporary);
    fs::rename(temporary, path); sync(path.parent_path(), true);
}
std::string read_text(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw shogi::Error("Cannot read " + path.string());
    return {std::istreambuf_iterator<char>(stream), {}};
}
std::string quoted(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) throw shogi::Error("Control character in path");
        else out << c;
    }
    return out.str() + '"';
}
std::uint64_t integer(const std::string& text) {
    std::uint64_t value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data()+text.size(), value);
    if (ec != std::errc{} || end != text.data()+text.size()) throw shogi::Error("Invalid integer: " + text);
    return value;
}
struct Config {
    fs::path out, stop_file;
    std::string weights;
    std::uint64_t games{1000}, seed{}, nodes{2000}, bytes{30ULL<<30}, first_id{};
    int threads{8}, max_threads{24}, hash_mb{16}, max_hash_mb{1024}, depth{64};
    int opening{8}, random_until{32}, max_moves{320};
    bool quiet_check{true}, quiet_capture{true};
    double random_probability{0.05}, temperature{}, near_best{100};
};
Config parse(int argc, char** argv) {
    Config c;
    std::map<std::string,std::string> args;
    for (int i=1; i<argc; i+=2) {
        if (i+1==argc || !args.emplace(argv[i],argv[i+1]).second) throw shogi::Error("Expected unique --option value pairs");
    }
    auto take = [&](const std::string& key, auto& field) {
        const auto it=args.find("--"+key);
        if (it==args.end()) return;
        using T = std::decay_t<decltype(field)>;
        if constexpr (std::is_same_v<T,fs::path> || std::is_same_v<T,std::string>) field=it->second;
        else if constexpr (std::is_same_v<T,double>) {
            std::size_t end{}; field=std::stod(it->second,&end);
            if (end!=it->second.size() || !std::isfinite(field)) throw shogi::Error("Invalid real option");
        } else {
            const auto value=integer(it->second);
            if (value>static_cast<std::uint64_t>(std::numeric_limits<T>::max())) throw shogi::Error("Option overflow");
            field=static_cast<T>(value);
        }
        args.erase(it);
    };
    take("out",c.out); take("stop-file",c.stop_file); take("weights",c.weights);
    take("games",c.games); take("seed",c.seed); take("nodes",c.nodes); take("byte-budget",c.bytes);
    take("first-id",c.first_id); take("threads",c.threads); take("max-threads",c.max_threads);
    take("hash-mb",c.hash_mb); take("max-hash-mb",c.max_hash_mb); take("depth",c.depth);
    take("opening-plies",c.opening); take("random-until",c.random_until); take("max-moves",c.max_moves);
    take("quiet-check",c.quiet_check); take("quiet-capture",c.quiet_capture);
    take("random-probability",c.random_probability); take("temperature",c.temperature); take("near-best",c.near_best);
    if (!args.empty()) throw shogi::Error("Unknown option " + args.begin()->first);
    if (c.out.empty() || !c.games || c.games>1000000 || c.first_id>UINT64_MAX-c.games || c.threads<1
        || c.threads>c.max_threads || c.max_threads>256 || c.hash_mb<1 || c.hash_mb>512
        || c.max_hash_mb<1 || c.hash_mb>c.max_hash_mb/c.threads || c.depth<1 || c.depth>64
        || c.max_moves<2 || c.max_moves>10000 || c.opening>=c.max_moves || c.random_until>c.max_moves
        || c.random_probability<0 || c.random_probability>1 || c.temperature<0 || c.near_best<0
        || c.bytes<16+static_cast<std::uint64_t>(c.max_moves)*ss::nnue::record_size+4096)
        throw shogi::Error("Invalid generation limits (budget must hold one maximum-sized game shard)");
    return c;
}
std::uint64_t game_seed(std::uint64_t seed, std::uint64_t id) {
    auto x=seed+0x9e3779b97f4a7c15ULL*(id+1);
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL; x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
struct GameData {
    std::uint64_t id{}, seed{}, nodes{}, bytes{};
    int winner{-2}, reason{}, plies{};
    std::vector<int> moves;
    std::vector<ss::nnue::Record> records;
    std::uint64_t count{};
    std::string hash, side_hash;
    bool evicted{};
};
std::string stem(std::uint64_t id) {
    std::ostringstream out; out << 'g' << std::setfill('0') << std::setw(12) << id; return out.str();
}
GameData play(const Config& c, std::uint64_t id, ss::Searcher& search) {
    GameData data; data.id=id; data.seed=game_seed(c.seed,id);
    std::mt19937_64 rng(data.seed);
    auto uniform=[&]() { return static_cast<double>(rng()>>11)*0x1.0p-53; };
    shogi::Game initial({c.max_moves,1}); ss::Root root(initial);
    while (!root.outcome.terminal()) {
        const auto legal=root.position.legal(true);
        ss::Limits limits; limits.depth=c.depth; limits.nodes=c.nodes;
        if (c.temperature>0) limits.multipv=0;
        const auto result=search.go(root,limits);
        if (std::find(legal.begin(),legal.end(),result.move)==legal.end()) throw shogi::Error("Illegal search move");
        const auto& p=root.position.board();
        const bool check=root.position.check();
        const bool capture=result.move<shogi::drop_base && p.board[(result.move/2)%81]!=0;
        data.nodes+=result.nodes;
        if ((!c.quiet_check || !check) && (!c.quiet_capture || !capture)) {
            ss::nnue::Record row;
            row.position=p; row.score=result.score; row.depth=static_cast<std::uint16_t>(result.depth);
            row.nodes=result.nodes; row.best_move=static_cast<std::uint16_t>(result.move);
            row.ply=static_cast<std::uint32_t>(data.moves.size()); row.game_id=id;
            row.flags=(check ? 1U:0U) | (capture ? 2U:0U) | (1U<<8);
            data.records.push_back(row);
        }
        int move=result.move;
        if (static_cast<int>(data.moves.size())<c.opening
            || (static_cast<int>(data.moves.size())<c.random_until && uniform()<c.random_probability)) {
            move=legal.values[rng()%static_cast<std::uint64_t>(legal.size)];
        } else if (c.temperature>0 && !result.roots.empty()) {
            double total=0;
            for (const auto& r:result.roots) if (r.score>=result.score-c.near_best) total+=std::exp((r.score-result.score)/c.temperature);
            double sample=uniform()*total;
            for (const auto& r:result.roots) if (r.score>=result.score-c.near_best) {
                sample-=std::exp((r.score-result.score)/c.temperature);
                if (sample<=0) { move=r.move; break; }
            }
        }
        data.moves.push_back(move);
        if (move==shogi::declare) root.outcome={p.turn,3};
        else {
            root.position.make(move);
            root.history.push_back({root.position.key(),root.position.board().turn,root.position.check()});
            const auto replies=root.position.legal(true);
            root.outcome=ss::adjudicate(root.position,replies,root.rules,root.history);
        }
    }
    data.winner=root.outcome.winner; data.reason=root.outcome.reason;
    data.plies=static_cast<int>(data.moves.size()); data.count=data.records.size();
    for (auto& r:data.records) r.result=static_cast<std::int8_t>(data.winner<0 ? 0 : data.winner==r.position.turn ? 1:-1);
    return data;
}
std::string marker(const GameData& d) {
    std::ostringstream out;
    out << d.id << ' ' << d.seed << ' ' << d.count << ' ' << d.winner << ' ' << d.reason << ' '
        << d.plies << ' ' << d.nodes << ' ' << d.bytes << ' ' << d.hash << ' ' << d.side_hash << '\n';
    for (int move:d.moves) out << move << ' ';
    out << '\n'; return out.str();
}
GameData recover(const fs::path& path) {
    std::istringstream in(read_text(path)); GameData d;
    in >> d.id >> d.seed >> d.count >> d.winner >> d.reason >> d.plies >> d.nodes >> d.bytes >> d.hash >> d.side_hash;
    if (!in || d.plies<1 || d.plies>10000 || d.hash.size()!=64 || d.side_hash.size()!=64) throw shogi::Error("Invalid commit marker");
    for (int i=0;i<d.plies;++i) { int move=-1; in>>move; if (!in || move<0 || move>shogi::declare) throw shogi::Error("Invalid committed move"); d.moves.push_back(move); }
    in>>std::ws; if (!in.eof()) throw shogi::Error("Trailing commit fields");
    return d;
}
std::string settings(const Config& c, const std::string& evaluator, const std::string& hash) {
    std::ostringstream out; out << std::setprecision(17);
    out << "{\"games\":" << c.games << ",\"seed\":" << c.seed << ",\"first_id\":" << c.first_id
        << ",\"nodes\":" << c.nodes << ",\"depth\":" << c.depth << ",\"hash_mb\":" << c.hash_mb
        << ",\"max_moves\":" << c.max_moves << ",\"opening_plies\":" << c.opening
        << ",\"random_until\":" << c.random_until << ",\"random_probability\":" << c.random_probability
        << ",\"temperature\":" << c.temperature << ",\"near_best\":" << c.near_best
        << ",\"quiet_check\":" << (c.quiet_check?"true":"false") << ",\"quiet_capture\":" << (c.quiet_capture?"true":"false")
        << ",\"byte_budget\":" << c.bytes << ",\"evaluator\":" << quoted(evaluator) << ",\"evaluator_sha256\":" << quoted(hash) << '}';
    return out.str();
}
class Run {
public:
    Config config;
    std::vector<GameData> games;
    std::string identity;
    std::uint64_t resident{}, positions{}, nodes{};
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    std::size_t initial_count{};
    explicit Run(Config c, std::string id) : config(std::move(c)), identity(std::move(id)) {
        const auto path=config.out/"settings.json";
        if (fs::exists(path)) {
            if (read_text(path)!=identity+"\n") throw shogi::Error("Resume settings/evaluator mismatch");
        } else {
            // No settings means no ownership, even if a shard lacks our commit marker.
            for (const auto& entry:fs::directory_iterator(config.out)) {
                const auto name=entry.path().filename().string();
                if (name.find(".hlnn1")!=std::string::npos || name.find(".commit")!=std::string::npos
                    || name.find(".evicted")!=std::string::npos || name.find(".pending")!=std::string::npos
                    || name=="manifest.json" || name=="manifest.json.tmp")
                    throw shogi::Error("Refusing existing dataset in a new output directory");
            }
            atomic_text(path,identity+"\n");
        }
        for (std::uint64_t i=0;i<config.games;++i) {
            const auto name=stem(config.first_id+i); const auto commit=config.out/(name+".commit");
            if (!fs::exists(commit)) break;
            auto d=recover(commit);
            if (d.id!=config.first_id+i || d.seed!=game_seed(config.seed,d.id)) throw shogi::Error("Committed game identity mismatch");
            d.evicted=fs::exists(config.out/(name+".evicted"));
            const auto shard=config.out/(name+".hlnn1");
            if (d.evicted) { fs::remove(shard); fs::remove(shard.string()+".json"); }
            else {
                if (ss::binary::file_hash(shard.string())!=d.hash || ss::binary::file_hash(shard.string()+".json")!=d.side_hash) throw shogi::Error("Committed shard corruption");
                resident+=d.bytes;
            }
            positions+=d.count; nodes+=d.nodes; games.push_back(std::move(d));
            const auto pending=config.out/(name+".pending");
            if (fs::exists(pending)) {
                if (read_text(pending)!=identity+"\n") throw shogi::Error("Pending shard ownership mismatch");
                fs::remove(pending);
            }
        }
        std::size_t commit_count=0;
        for (const auto& entry:fs::directory_iterator(config.out)) if (entry.path().extension()==".commit") ++commit_count;
        if (commit_count!=games.size()) throw shogi::Error("Noncontiguous committed games");
        if (games.size()<config.games) {
            const auto name=stem(config.first_id+games.size());
            const auto pending=config.out/(name+".pending");
            if (fs::exists(pending)) {
                if (read_text(pending)!=identity+"\n") throw shogi::Error("Pending shard ownership mismatch");
                // A durable intent precedes every shard write in this established run.
                for (const auto* suffix:{".hlnn1", ".hlnn1.json", ".hlnn1.json.tmp", ".commit.tmp"})
                    fs::remove(config.out/(name+suffix));
                sync(config.out,true);
                fs::remove(pending);
                sync(config.out,true);
            }
        }
        initial_count=games.size(); evict(); manifest();
    }
    void evict(std::uint64_t reserve = 0) {
        for (auto& d:games) {
            if (resident<=config.bytes-reserve) break;
            if (d.evicted) continue;
            const auto name=stem(d.id);
            atomic_text(config.out/(name+".evicted"),"oldest-shard byte budget\n");
            d.evicted=true;
            fs::remove(config.out/(name+".hlnn1")); fs::remove(config.out/(name+".hlnn1.json"));
            resident-=d.bytes;
        }
        sync(config.out,true);
    }
    void commit(GameData d) {
        const auto name=stem(d.id); const auto shard=config.out/(name+".hlnn1");
        for (const auto* suffix:{".hlnn1", ".hlnn1.json", ".hlnn1.json.tmp", ".commit", ".commit.tmp", ".pending"})
            if (fs::exists(config.out/(name+suffix))) throw shogi::Error("Refusing existing dataset artifact: " + name+suffix);
        evict(16+d.count*ss::nnue::record_size+512);
        atomic_text(config.out/(name+".pending"),identity+"\n");
        ss::nnue::DataWriter writer(shard.string());
        for (const auto& row:d.records) writer.append(row);
        writer.close(); sync(shard); sync(shard.string()+".json");
        d.hash=ss::binary::file_hash(shard.string()); d.side_hash=ss::binary::file_hash(shard.string()+".json");
        d.bytes=fs::file_size(shard)+fs::file_size(shard.string()+".json");
        atomic_text(config.out/(name+".commit"),marker(d));
        fs::remove(config.out/(name+".pending")); sync(config.out,true);
        d.records.clear(); d.records.shrink_to_fit();
        resident+=d.bytes; positions+=d.count; nodes+=d.nodes; games.push_back(std::move(d)); evict();
        if (games.size()%32==0 || games.size()==config.games) manifest();
        std::cout << "completed " << games.size() << " games " << positions << " positions\n" << std::flush;
    }
    void manifest() const {
        const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::ostringstream out;
        out << std::setprecision(17) << "{\"format\":\"hlnn-run-v1\",\"status\":\"" << (games.size()==config.games?"complete":"partial")
            << "\",\"settings\":" << identity << ",\"threads\":" << config.threads << ",\"completed_games\":" << games.size()
            << ",\"positions\":" << positions << ",\"nodes\":" << nodes << ",\"resident_bytes\":" << resident
            << ",\"session_seconds\":" << elapsed << ",\"session_games\":" << games.size()-initial_count << ",\"games\":[";
        std::map<int,int> reasons;
        std::uint64_t session_positions=0;
        for (std::size_t i=0;i<games.size();++i) {
            const auto& d=games[i]; ++reasons[d.reason]; if (i>=initial_count) session_positions+=d.count;
            if (i) out << ',';
            out << "{\"id\":" << d.id << ",\"seed\":" << d.seed << ",\"records\":" << d.count << ",\"winner\":" << d.winner
                << ",\"reason\":" << d.reason << ",\"plies\":" << d.plies << ",\"nodes\":" << d.nodes << ",\"bytes\":" << d.bytes
                << ",\"sha256\":" << quoted(d.hash) << ",\"sidecar_sha256\":" << quoted(d.side_hash)
                << ",\"shard\":" << quoted(stem(d.id)+".hlnn1") << ",\"evicted\":" << (d.evicted?"true":"false") << ",\"moves\":[";
            for (std::size_t j=0;j<d.moves.size();++j) { if (j) out << ','; out << d.moves[j]; }
            out << "]}";
        }
        out << "],\"outcomes_by_reason\":{";
        bool first=true; for (const auto& [reason,count]:reasons) { if (!first) out << ','; first=false; out << quoted(std::to_string(reason)) << ':' << count; }
        out << "},\"session_positions\":" << session_positions << ",\"positions_per_hour\":" << (elapsed>0?session_positions*3600/elapsed:0)
            << ",\"games_per_hour\":" << (elapsed>0?(games.size()-initial_count)*3600/elapsed:0) << "}\n";
        atomic_text(config.out/"manifest.json",out.str());
    }
};
}
int main(int argc, char** argv) {
    try {
        const auto config=parse(argc,argv); fs::create_directories(config.out);
        const RunLock lock(config.out);
        std::shared_ptr<const ss::nnue::Weights> weights;
        std::string evaluator="bootstrap-v1", hash;
        if (!config.weights.empty()) { weights=ss::nnue::Weights::load(config.weights); evaluator=weights->architecture(); hash=ss::binary::file_hash(config.weights); }
        else { ss::binary::Sha256 sha; sha.update(std::span(reinterpret_cast<const std::uint8_t*>(evaluator.data()),evaluator.size())); hash=ss::binary::hex(sha.finish()); }
        Run run(config,settings(config,evaluator,hash));
        std::signal(SIGTERM,stop_signal); std::signal(SIGINT,stop_signal);
        std::atomic<std::uint64_t> next{run.games.size()};
        std::mutex mutex; std::condition_variable changed; std::exception_ptr failure;
        std::atomic<bool> failed{false};
        std::vector<std::jthread> workers;
        for (int t=0;t<config.threads;++t) workers.emplace_back([&] {
            try {
                std::unique_ptr<ss::Evaluator> eval;
                if (weights) eval=std::make_unique<ss::nnue::NnueEvaluator>(weights);
                ss::Searcher search({ss::all_features,config.hash_mb,12},std::move(eval));
                for (;;) {
                    if (stopping || failed || (!config.stop_file.empty() && fs::exists(config.stop_file))) break;
                    const auto index=next.fetch_add(1);
                    if (index>=config.games) break;
                    auto data=play(config,config.first_id+index,search);
                    std::unique_lock guard(mutex);
                    changed.wait(guard,[&] { return failed || index==run.games.size(); });
                    if (failed) break;
                    run.commit(std::move(data)); changed.notify_all();
                }
            } catch (...) {
                std::lock_guard guard(mutex); if (!failure) failure=std::current_exception(); failed=true; changed.notify_all();
            }
        });
        workers.clear(); run.manifest();
        if (failure) std::rethrow_exception(failure);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
