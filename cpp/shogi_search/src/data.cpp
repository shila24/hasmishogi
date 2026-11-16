// SPDX-License-Identifier: MIT
#include "shogi_search/data.hpp"
#include "shogi_search/nnue.hpp"
#include "shogi_search/binary.hpp"
#include <bit>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <map>
#include <sstream>

namespace shogi_search::nnue {
namespace {
constexpr std::uint32_t flags_mask=0x00ffff07;
std::vector<std::uint8_t> data_header() {
    std::vector<std::uint8_t> b;
    binary::text(b,"hlnn1",8); binary::put(b,1,4); binary::put(b,record_size,4); return b;
}
// The sidecar grammar is deliberately a flat object of exact known fields, not arbitrary JSON.
std::map<std::string,std::string> sidecar(const std::string& path) {
    std::ifstream stream(path);
    if (!stream || std::filesystem::file_size(path)>4096) throw shogi::Error("Missing/oversized hlnn1 sidecar");
    std::map<std::string,std::string> fields;
    auto character=[&](char expected) { stream>>std::ws; if (stream.get()!=expected) throw shogi::Error("Invalid hlnn1 sidecar"); };
    auto string=[&]() {
        character('"'); std::string out;
        while (stream && stream.peek()!='"') {
            const int c=stream.get();
            if (c<32 || c=='\\') throw shogi::Error("Invalid hlnn1 sidecar string");
            out.push_back(static_cast<char>(c));
        }
        character('"'); return out;
    };
    character('{');
    for (;;) {
        const auto key=string(); character(':'); stream>>std::ws;
        std::string value;
        if (key=="format" || key=="sha256") value=string();
        else while (stream.peek()>='0' && stream.peek()<='9') value.push_back(static_cast<char>(stream.get()));
        if (value.empty() || !fields.emplace(key,value).second) throw shogi::Error("Invalid hlnn1 sidecar field");
        stream>>std::ws;
        const int next=stream.get();
        if (next=='}') break;
        if (next!=',') throw shogi::Error("Invalid hlnn1 sidecar separator");
    }
    stream>>std::ws;
    if (stream.peek()!=std::char_traits<char>::eof() || fields.size()!=7) throw shogi::Error("Invalid hlnn1 sidecar fields");
    return fields;
}
std::uint64_t number(const std::map<std::string,std::string>& fields,const std::string& key) {
    const auto& text=fields.at(key);
    std::uint64_t value{};
    const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    if (error!=std::errc{} || end!=text.data()+text.size() || (text.size()>1 && text[0]=='0')) throw shogi::Error("Invalid sidecar number");
    return value;
}
}
std::vector<std::uint8_t> pack(const Record& r) {
    validate_position(r.position);
    if (r.best_move>shogi::declare || r.result < -1 || r.result>1 || r.ply>1000000 || (r.flags&~flags_mask)
        || (!std::isnan(r.teacher) && (!std::isfinite(r.teacher) || std::abs(r.teacher)>1))) throw shogi::Error("Invalid hlnn1 record");
    std::vector<std::uint8_t> out; out.reserve(record_size);
    for (int c:r.position.board) binary::put(out,c,1);
    for (const auto& hand:r.position.hands) for (int c:hand) binary::put(out,c,1);
    binary::put(out,r.position.turn,1); binary::put(out,r.position.move_number,4);
    binary::put(out,std::bit_cast<std::uint32_t>(r.score),4); binary::put(out,r.depth,2);
    binary::put(out,r.nodes,8); binary::put(out,r.best_move,2);
    binary::put(out,std::bit_cast<std::uint8_t>(r.result),1); binary::put(out,r.ply,4); binary::put(out,r.flags,4);
    binary::put(out,std::isnan(r.teacher) ? 0x7fc00000U : std::bit_cast<std::uint32_t>(r.teacher),4);
    binary::put(out,r.game_id,8); return out;
}
Record unpack(std::span<const std::uint8_t> bytes) {
    if (bytes.size()!=record_size) throw shogi::Error("Invalid hlnn1 record size");
    Record r; std::size_t offset=0;
    auto get=[&](int size) { return binary::get(bytes,offset,size); };
    for (auto& cell:r.position.board) cell=static_cast<int>(get(1));
    for (auto& hand:r.position.hands) for (auto& count:hand) count=static_cast<int>(get(1));
    r.position.turn=static_cast<int>(get(1));
    const auto move_number=get(4);
    if (move_number>2000000) throw shogi::Error("Invalid hlnn1 move number");
    r.position.move_number=static_cast<int>(move_number);
    r.score=std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(get(4)));
    r.depth=static_cast<std::uint16_t>(get(2)); r.nodes=get(8);
    r.best_move=static_cast<std::uint16_t>(get(2)); r.result=std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(get(1)));
    r.ply=static_cast<std::uint32_t>(get(4)); r.flags=static_cast<std::uint32_t>(get(4));
    r.teacher=std::bit_cast<float>(static_cast<std::uint32_t>(get(4))); r.game_id=get(8);
    const auto canonical=pack(r);
    if (!std::equal(canonical.begin(),canonical.end(),bytes.begin())) throw shogi::Error("Noncanonical hlnn1 record");
    return r;
}
DataWriter::DataWriter(const std::string& path) : path_(path) {
    if (std::filesystem::exists(path) || std::filesystem::exists(path+".json")) throw shogi::Error("hlnn1 shard already exists");
    stream_.open(path,std::ios::binary);
    if (!stream_) throw shogi::Error("Cannot create hlnn1 shard");
    const auto h=data_header(); stream_.write(reinterpret_cast<const char*>(h.data()),h.size());
}
void DataWriter::append(const Record& record) {
    if (closed_) throw shogi::Error("Closed hlnn1 writer");
    const auto data=pack(record);
    stream_.write(reinterpret_cast<const char*>(data.data()),data.size());
    if (!stream_) throw shogi::Error("Cannot write hlnn1 record");
    ++count_; games_.insert(record.game_id);
}
void DataWriter::close() {
    if (closed_) return;
    stream_.flush();
    if (!stream_) throw shogi::Error("Cannot finish hlnn1 shard");
    stream_.close();
    const auto sha=binary::file_hash(path_);
    const auto temporary=path_+".json.tmp";
    std::ofstream side(temporary);
    side << "{\"format\":\"hlnn1\",\"version\":1,\"record_size\":" << record_size
         << ",\"records\":" << count_ << ",\"games\":" << games_.size() << ",\"bytes\":"
         << (16+count_*record_size) << ",\"sha256\":\"" << sha << "\"}\n";
    side.flush(); if (!side) throw shogi::Error("Cannot write hlnn1 sidecar"); side.close();
    std::filesystem::rename(temporary,path_+".json"); closed_=true;
}
DataReader::DataReader(const std::string& path) : stream_(path,std::ios::binary) {
    const auto fields=sidecar(path+".json");
    expected_=number(fields,"records"); expected_games_=number(fields,"games");
    const auto bytes=number(fields,"bytes");
    if (fields.at("format")!="hlnn1" || number(fields,"version")!=1 || number(fields,"record_size")!=record_size
        || expected_>(std::numeric_limits<std::uint64_t>::max()-16)/record_size || expected_games_>expected_
        || bytes!=16+expected_*record_size || bytes!=std::filesystem::file_size(path)
        || fields.at("sha256")!=binary::file_hash(path)) throw shogi::Error("hlnn1 sidecar/count/sha256 mismatch");
    if (binary::read(stream_,16)!=data_header()) throw shogi::Error("hlnn1 header mismatch");
}
bool DataReader::next(Record& out) {
    if (count_==expected_) {
        if (games_.size()!=expected_games_ || stream_.peek()!=std::char_traits<char>::eof()) throw shogi::Error("hlnn1 count mismatch");
        return false;
    }
    out=unpack(binary::read(stream_,record_size)); ++count_; games_.insert(out.game_id); return true;
}
}
