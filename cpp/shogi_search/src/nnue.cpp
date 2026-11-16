// SPDX-License-Identifier: MIT
#include "shogi_search/nnue.hpp"
#include "shogi_search/binary.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>

namespace shogi_search::nnue {
namespace {
constexpr int board_features=26*81, hand_features=76;
constexpr std::size_t header_size=104;
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559);
int relative(int square,int side) { return side ? 80-square : square; }
std::array<std::size_t,8> sizes(int w,int h1,int h2) {
    return {static_cast<std::size_t>(feature_count)*w,static_cast<std::size_t>(w),
            static_cast<std::size_t>(2*w*h1),static_cast<std::size_t>(h1),
            static_cast<std::size_t>(h1*h2),static_cast<std::size_t>(h2),static_cast<std::size_t>(h2),1};
}
void dimensions(int w,int h1,int h2) {
    if (w<1 || w>512 || h1<1 || h1>128 || h2<1 || h2>128) throw shogi::Error("Invalid NNUE architecture dimensions");
}
std::vector<std::uint8_t> header(const Weights& w) {
    std::vector<std::uint8_t> out;
    binary::text(out,"hlnnw1",8); binary::put(out,1,4);
    binary::text(out,feature_set,16); binary::text(out,w.architecture(),48);
    binary::put(out,feature_count,4); binary::put(out,w.width,4);
    binary::put(out,w.hidden1,4); binary::put(out,w.hidden2,4);
    binary::put(out,std::bit_cast<std::uint32_t>(output_scale),4);
    std::size_t count=0; for (auto size:sizes(w.width,w.hidden1,w.hidden2)) count+=size;
    binary::put(out,count,8); return out;
}
}
void validate_position(const shogi::Position& p) {
    if (p.turn<0 || p.turn>1 || p.move_number<1 || p.move_number>2000000) throw shogi::Error("Invalid NNUE position");
    std::array<int,7> totals{};
    std::array<int,2> kings{};
    for (int cell:p.board) {
        if (!cell) continue;
        const int side=cell>>4, kind=cell&15;
        if (cell<0 || side>1 || kind<1 || kind>14) throw shogi::Error("Invalid NNUE board piece");
        if (kind==8) ++kings[side];
        else ++totals[shogi::unpromoted(kind)-1];
    }
    for (const auto& hand:p.hands) for (int h=0;h<7;++h) {
        if (hand[h]<0 || hand[h]>hand_limits[h]) throw shogi::Error("Invalid NNUE hand");
        totals[h]+=hand[h];
    }
    if (kings[0]!=1 || kings[1]!=1) throw shogi::Error("NNUE requires two kings");
    for (int h=0;h<7;++h) if (totals[h]>hand_limits[h]) throw shogi::Error("NNUE material overflow");
}
Features features(const shogi::Position& p,int side) {
    if (side<0 || side>1) throw shogi::Error("Invalid NNUE perspective");
    Features out;
    out.king=relative(shogi::king_square(p,side),side);
    const int offset=out.king*king_stride;
    auto add=[&](int index) {
        if (out.size>=max_active || index<0 || index>=feature_count) throw shogi::Error("Invalid NNUE feature");
        out.indices[out.size++]=index;
    };
    for (int square=0;square<81;++square) {
        const int cell=p.board[square], kind=cell&15;
        if (!cell || kind==8) continue;
        const auto it=std::find(board_types.begin(),board_types.end(),kind);
        if (it==board_types.end() || (cell>>4)>1) throw shogi::Error("Invalid NNUE piece type");
        const int type=static_cast<int>(it-board_types.begin());
        add(offset+(((cell>>4)^side)*13+type)*81+relative(square,side));
    }
    for (int owner=0;owner<2;++owner) {
        int base=offset+board_features+owner*38;
        for (int h=0;h<7;++h) {
            if (p.hands[owner^side][h]<0 || p.hands[owner^side][h]>hand_limits[h]) throw shogi::Error("Invalid NNUE hand");
            for (int i=0;i<p.hands[owner^side][h];++i) add(base+i);
            base+=hand_limits[h];
        }
    }
    add(offset+board_features+hand_features+relative(shogi::king_square(p,1-side),side));
    std::sort(out.indices.begin(),out.indices.begin()+out.size);
    return out;
}
std::string Weights::architecture() const {
    return std::string(feature_set)+":w"+std::to_string(width)+":"+std::to_string(hidden1)+":"+
        std::to_string(hidden2)+":f32:cp600";
}
void Weights::validate() const {
    dimensions(width,hidden1,hidden2);
    const auto expected=sizes(width,hidden1,hidden2);
    const std::array<const std::vector<float>*,8> tensors{&embedding,&bias,&w1,&b1,&w2,&b2,&w3,&b3};
    for (std::size_t i=0;i<tensors.size();++i) {
        if (tensors[i]->size()!=expected[i]) throw shogi::Error("NNUE tensor size mismatch");
        for (float value:*tensors[i]) if (!std::isfinite(value) || std::abs(value)>1e6F) throw shogi::Error("Invalid NNUE tensor");
    }
}
void Weights::save(const std::string& path) const {
    validate();
    std::ofstream stream(path,std::ios::binary);
    if (!stream) throw shogi::Error("Cannot write NNUE weights");
    binary::Sha256 hash;
    auto write=[&](const auto& data) {
        hash.update(data); stream.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));
    };
    write(header(*this));
    for (const auto* tensor:{&embedding,&bias,&w1,&b1,&w2,&b2,&w3,&b3}) {
        std::vector<std::uint8_t> bytes; bytes.reserve(tensor->size()*4);
        for (float value:*tensor) binary::put(bytes,std::bit_cast<std::uint32_t>(value),4);
        write(bytes);
    }
    const auto digest=hash.finish();
    stream.write(reinterpret_cast<const char*>(digest.data()),digest.size()); stream.flush();
    if (!stream) throw shogi::Error("Cannot finish NNUE weights");
}
std::shared_ptr<const Weights> Weights::load(const std::string& path) {
    std::ifstream stream(path,std::ios::binary);
    if (!stream) throw shogi::Error("Cannot open NNUE weights");
    const auto bytes=binary::read(stream,header_size);
    auto w=std::make_shared<Weights>();
    std::size_t offset=76;
    if (binary::get(bytes,offset,4)!=feature_count) throw shogi::Error("NNUE feature count mismatch");
    const auto width=binary::get(bytes,offset,4), h1=binary::get(bytes,offset,4), h2=binary::get(bytes,offset,4);
    if (width>512 || h1>128 || h2>128) throw shogi::Error("NNUE architecture dimensions exceed bounds");
    w->width=static_cast<int>(width); w->hidden1=static_cast<int>(h1); w->hidden2=static_cast<int>(h2);
    dimensions(w->width,w->hidden1,w->hidden2);
    if (bytes!=header(*w)) throw shogi::Error("hlnnw1 feature/architecture/header mismatch");
    const auto expected=sizes(w->width,w->hidden1,w->hidden2);
    std::size_t count=0; for (auto n:expected) count+=n;
    stream.seekg(0,std::ios::end);
    if (stream.tellg()!=static_cast<std::streamoff>(header_size+4*count+32)) throw shogi::Error("hlnnw1 file size mismatch");
    stream.seekg(header_size);
    binary::Sha256 hash; hash.update(bytes);
    const std::array<std::vector<float>*,8> tensors{&w->embedding,&w->bias,&w->w1,&w->b1,&w->w2,&w->b2,&w->w3,&w->b3};
    for (std::size_t i=0;i<tensors.size();++i) {
        const auto data=binary::read(stream,expected[i]*4); hash.update(data);
        auto& tensor=*tensors[i]; tensor.resize(expected[i]); offset=0;
        for (auto& value:tensor) value=std::bit_cast<float>(static_cast<std::uint32_t>(binary::get(data,offset,4)));
    }
    const auto digest=binary::read(stream,32); const auto expected_hash=hash.finish();
    if (!std::equal(digest.begin(),digest.end(),expected_hash.begin())) throw shogi::Error("hlnnw1 sha256 mismatch");
    w->validate(); return w;
}
NnueEvaluator::NnueEvaluator(std::shared_ptr<const Weights> weights) : weights_(std::move(weights)) {
    if (!weights_) throw shogi::Error("Missing NNUE weights");
    stack_.resize(static_cast<std::size_t>(max_ply+2)*2*weights_->width);
    compensation_.resize(stack_.size());
    auto transpose=[](const auto& weights,int inputs,int outputs) {
        std::vector<float> columns(weights.size());
        for (int col=0;col<inputs;++col) for (int row=0;row<outputs;++row)
            columns[col*outputs+row]=weights[row*inputs+col];
        return columns;
    };
    w1_columns_=transpose(weights_->w1,2*weights_->width,weights_->hidden1);
    w2_columns_=transpose(weights_->w2,weights_->hidden1,weights_->hidden2);
}
std::span<float> NnueEvaluator::current(int side,int depth) const {
    return std::span(stack_).subspan(static_cast<std::size_t>(depth*2+side)*weights_->width,weights_->width);
}
std::span<const float> NnueEvaluator::accumulator(int side) const {
    if (side<0 || side>1) throw shogi::Error("Invalid accumulator perspective");
    ensure(side,depth_);
    return current(side,depth_);
}
void NnueEvaluator::add(int side,int depth,int feature,bool remove) const {
    auto values=current(side,depth);
    auto errors=std::span(compensation_).subspan(static_cast<std::size_t>(depth*2+side)*weights_->width,weights_->width);
    const auto base=static_cast<std::size_t>(feature)*weights_->width;
    for (int j=0;j<weights_->width;++j) {
        const float term=remove ? -weights_->embedding[base+j] : weights_->embedding[base+j];
        const float adjusted=term-errors[j];
        const float next=values[j]+adjusted;
        errors[j]=(next-values[j])-adjusted;
        values[j]=next;
    }
}
void NnueEvaluator::refresh(int side,int depth,const Features& f) const {
    auto values=current(side,depth); std::copy(weights_->bias.begin(),weights_->bias.end(),values.begin());
    auto errors=std::span(compensation_).subspan(static_cast<std::size_t>(depth*2+side)*weights_->width,weights_->width);
    std::fill(errors.begin(),errors.end(),0.0F);
    for (int i=0;i<f.size;++i) add(side,depth,f.indices[i],false);
}
void NnueEvaluator::reset(const Position& p) {
    depth_=0; validate_position(p.board());
    for (int side=0;side<2;++side) {
        features_[0][side]=features(p.board(),side);
        refresh(side,0,features_[0][side]); ready_[0][side]=true;
    }
}
void NnueEvaluator::push(const Position& after,const Undo& undo) {
    if (depth_>=max_ply) throw shogi::Error("NNUE accumulator stack overflow");
    ++depth_;
    const auto& p=after.board();
    const int mover=1-p.turn;
    for (int side=0;side<2;++side) {
        ready_[depth_][side]=false;
        auto& delta=deltas_[depth_][side]; delta={};
        auto& next=features_[depth_][side];
        const auto& previous=features_[depth_-1][side];
        if ((undo.moved&15)==8 && mover==side) {
            next=features(p,side); delta.refresh=true; continue;
        }
        next=previous;
        if (undo.code<0 || undo.target<0) continue;
        const int offset=next.king*king_stride;
        auto board_feature=[&](int cell,int square) {
            const int kind=cell&15;
            if (kind==8) return offset+board_features+hand_features+relative(square,side);
            const int type=kind<=7 ? kind-1 : kind-2;
            return offset+(((cell>>4)^side)*13+type)*81+relative(square,side);
        };
        if (undo.source>=0) delta.removed[delta.removals++]=board_feature(undo.moved,undo.source);
        if (undo.captured) delta.removed[delta.removals++]=board_feature(undo.captured,undo.target);
        delta.added[delta.additions++]=board_feature(p.board[undo.target],undo.target);
        if (undo.hand>=0) {
            int base=offset+board_features+(mover^side)*38;
            for (int h=0;h<undo.hand;++h) base+=hand_limits[h];
            const int count=p.hands[mover][undo.hand];
            if (undo.source<0) delta.removed[delta.removals++]=base+count;
            else delta.added[delta.additions++]=base+count-1;
        }
        std::sort(delta.removed.begin(),delta.removed.begin()+delta.removals);
        std::sort(delta.added.begin(),delta.added.begin()+delta.additions);
        // Maintain canonical feature order without rescanning the board or hands.
        for (int i=0;i<delta.removals;++i) {
            auto it=std::lower_bound(next.indices.begin(),next.indices.begin()+next.size,delta.removed[i]);
            if (it==next.indices.begin()+next.size || *it!=delta.removed[i]) throw shogi::Error("Missing NNUE delta feature");
            std::move(it+1,next.indices.begin()+next.size,it); --next.size;
        }
        for (int i=0;i<delta.additions;++i) {
            auto it=std::lower_bound(next.indices.begin(),next.indices.begin()+next.size,delta.added[i]);
            if (next.size>=max_active) throw shogi::Error("NNUE delta feature overflow");
            std::move_backward(it,next.indices.begin()+next.size,next.indices.begin()+next.size+1);
            *it=delta.added[i]; ++next.size;
        }
    }
}
void NnueEvaluator::ensure(int side,int depth) const {
    if (ready_[depth][side]) return;
    if (depth==0) throw shogi::Error("NNUE evaluator must be reset before evaluation");
    const auto& delta=deltas_[depth][side];
    if (delta.refresh) refresh(side,depth,features_[depth][side]);
    else {
        ensure(side,depth-1);
        const auto previous=current(side,depth-1);
        std::copy(previous.begin(),previous.end(),current(side,depth).begin());
        const auto from=compensation_.begin()+static_cast<std::ptrdiff_t>((depth-1)*2+side)*weights_->width;
        const auto to=compensation_.begin()+static_cast<std::ptrdiff_t>(depth*2+side)*weights_->width;
        std::copy_n(from,weights_->width,to);
        // Preserve the previous incremental arithmetic order exactly, including Kahan errors.
        for (int i=0;i<delta.removals;++i) add(side,depth,delta.removed[i],true);
        for (int i=0;i<delta.additions;++i) add(side,depth,delta.added[i],false);
    }
    ready_[depth][side]=true;
}
void NnueEvaluator::pop() noexcept { if (depth_>0) --depth_; }
float NnueEvaluator::evaluate_float(const Position& p) const {
    std::array<float,1024> input{};
    const int width=weights_->width;
    for (int order=0;order<2;++order) {
        const auto a=accumulator(p.board().turn^order);
        for (int j=0;j<width;++j) input[order*width+j]=std::clamp(a[j],0.0F,1.0F);
    }
    auto dense=[](std::span<const float> x,const std::vector<float>& columns,const std::vector<float>& bias,bool clip) {
        alignas(64) std::array<float,128> y{};
        std::copy(bias.begin(),bias.end(),y.begin());
        // Independent output neurons are SIMD lanes. Each lane retains ascending
        // input additions and separate multiply/add, so no reduction order changes.
        for (std::size_t column=0;column<x.size();++column) {
            const auto* w=columns.data()+column*bias.size();
            for (std::size_t row=0;row<bias.size();++row) y[row]+=w[row]*x[column];
        }
        if (clip) for (std::size_t row=0;row<bias.size();++row) y[row]=std::clamp(y[row],0.0F,1.0F);
        return y;
    };
    const auto first=dense(std::span(input).first(2*width),w1_columns_,weights_->b1,true);
    const auto second=dense(std::span(first).first(weights_->hidden1),w2_columns_,weights_->b2,true);
    return dense(std::span(second).first(weights_->hidden2),weights_->w3,weights_->b3,false)[0]*output_scale;
}
int NnueEvaluator::evaluate(const Position& p) const {
    // Learned values must not enter the reserved mate-distance interval.
    return static_cast<int>(std::round(std::clamp(evaluate_float(p),-900000.0F,900000.0F)));
}
}
