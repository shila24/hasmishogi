// SPDX-License-Identifier: MIT
// Original implementation of the published SHA-256 compression algorithm.
#include "shogi_search/binary.hpp"
#include "shogi/game.hpp"
#include <algorithm>
#include <bit>
#include <fstream>

namespace shogi_search::binary {
namespace {
constexpr std::array<std::uint32_t,64> constants{
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};
}
void Sha256::compress() {
    std::array<std::uint32_t,64> w{};
    for (int i=0;i<16;++i) for (int j=0;j<4;++j) w[i]=(w[i]<<8)|block_[i*4+j];
    for (int i=16;i<64;++i) {
        const auto x=w[i-15], y=w[i-2];
        const auto s0=std::rotr(x,7)^std::rotr(x,18)^(x>>3);
        const auto s1=std::rotr(y,17)^std::rotr(y,19)^(y>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    auto v=state_;
    for (int i=0;i<64;++i) {
        const auto s1=std::rotr(v[4],6)^std::rotr(v[4],11)^std::rotr(v[4],25);
        const auto choose=(v[4]&v[5])^(~v[4]&v[6]);
        const auto t1=v[7]+s1+choose+constants[i]+w[i];
        const auto s0=std::rotr(v[0],2)^std::rotr(v[0],13)^std::rotr(v[0],22);
        const auto majority=(v[0]&v[1])^(v[0]&v[2])^(v[1]&v[2]);
        for (int j=7;j>0;--j) v[j]=v[j-1];
        v[4]+=t1; v[0]=t1+s0+majority;
    }
    for (int i=0;i<8;++i) state_[i]+=v[i];
}
void Sha256::update(std::span<const std::uint8_t> bytes) {
    bytes_+=bytes.size();
    while (!bytes.empty()) {
        const auto count=std::min(bytes.size(),block_.size()-used_);
        std::copy_n(bytes.begin(),count,block_.begin()+static_cast<std::ptrdiff_t>(used_));
        used_+=count; bytes=bytes.subspan(count);
        if (used_==64) { compress(); used_=0; }
    }
}
std::array<std::uint8_t,32> Sha256::finish() {
    const auto bits=bytes_*8;
    const std::array<std::uint8_t,1> high{0x80}, zero{0};
    update(high);
    while (used_!=56) update(zero);
    std::array<std::uint8_t,8> length{};
    for (int i=0;i<8;++i) length[7-i]=static_cast<std::uint8_t>(bits>>(i*8));
    update(length);
    std::array<std::uint8_t,32> out{};
    for (int i=0;i<8;++i) for (int j=0;j<4;++j) out[4*i+j]=static_cast<std::uint8_t>(state_[i]>>(24-j*8));
    return out;
}
std::string hex(std::span<const std::uint8_t> bytes) {
    std::string out;
    for (auto b:bytes) { out.push_back("0123456789abcdef"[b>>4]); out.push_back("0123456789abcdef"[b&15]); }
    return out;
}
std::string file_hash(const std::string& path) {
    std::ifstream stream(path,std::ios::binary);
    if (!stream) throw shogi::Error("Cannot open hash input");
    Sha256 hash;
    std::array<std::uint8_t,65536> buffer{};
    while (stream) {
        stream.read(reinterpret_cast<char*>(buffer.data()),buffer.size());
        hash.update(std::span(buffer).first(static_cast<std::size_t>(stream.gcount())));
    }
    if (!stream.eof()) throw shogi::Error("Cannot read hash input");
    return hex(hash.finish());
}
void put(std::vector<std::uint8_t>& out,std::uint64_t value,int bytes) {
    if (bytes<1 || bytes>8) throw shogi::Error("Invalid integer width");
    for (int i=0;i<bytes;++i) out.push_back(static_cast<std::uint8_t>(value>>(i*8)));
}
std::uint64_t get(std::span<const std::uint8_t> data,std::size_t& offset,int bytes) {
    if (bytes<1 || bytes>8 || offset>data.size() || data.size()-offset<static_cast<std::size_t>(bytes))
        throw shogi::Error("Truncated binary field");
    std::uint64_t out=0;
    for (int i=0;i<bytes;++i) out|=std::uint64_t{data[offset++]}<<(i*8);
    return out;
}
std::vector<std::uint8_t> read(std::istream& stream,std::size_t size) {
    std::vector<std::uint8_t> out(size);
    stream.read(reinterpret_cast<char*>(out.data()),static_cast<std::streamsize>(size));
    if (static_cast<std::size_t>(stream.gcount())!=size) throw shogi::Error("Truncated binary file");
    return out;
}
void text(std::vector<std::uint8_t>& out,const std::string& value,std::size_t size) {
    if (value.size()>size) throw shogi::Error("Oversized text field");
    out.insert(out.end(),value.begin(),value.end()); out.insert(out.end(),size-value.size(),0);
}
}
