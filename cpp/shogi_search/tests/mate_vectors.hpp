// SPDX-License-Identifier: MIT
#ifndef SHOGI_SEARCH_MATE_VECTORS_HPP
#define SHOGI_SEARCH_MATE_VECTORS_HPP
#include "shogi_search/search.hpp"
#include <fstream>
#include <sstream>
namespace mate_vectors {
struct Row { std::string id; int plies; std::string sfen; };
inline std::vector<Row> read(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw shogi::Error("Cannot open mate suite");
    std::string line; std::getline(file,line);
    if (line != "# shogi-search-mates-v1") throw shogi::Error("Unsupported mate suite version");
    std::vector<Row> rows;
    while (std::getline(file,line)) if (!line.empty() && line[0] != '#') {
        std::istringstream stream(line); Row row; std::string depth;
        if (!std::getline(stream,row.id,'\t') || !std::getline(stream,depth,'\t') || !std::getline(stream,row.sfen,'\t'))
            throw shogi::Error("Malformed mate vector");
        row.plies = std::stoi(depth);
        if (row.plies < 1 || row.plies > 5 || row.plies%2 != 1) throw shogi::Error("Invalid mate depth");
        rows.push_back(std::move(row));
    }
    if (rows.empty()) throw shogi::Error("Empty mate suite");
    return rows;
}
inline bool forced(shogi::Position position,int depth,int attacker) {
    const auto legal = shogi::legal_moves(position);
    if (legal.empty()) return position.turn != attacker && shogi::in_check(position,position.turn);
    if (!depth) return false;
    for (int move : legal) {
        auto next = position; shogi::apply_move(next,move); next.turn ^= 1;
        const bool win = forced(next,depth-1,attacker);
        if (position.turn == attacker && win) return true;
        if (position.turn != attacker && !win) return false;
    }
    return position.turn != attacker;
}
}
#endif
