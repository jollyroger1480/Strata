// metal/src/engine/stream.cpp - see stream.h.
#include "stream.h"

#include "ref_quant.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

namespace strata::metal_engine {

namespace {
constexpr int NL = 48, NE = 512;
constexpr uint64_t PAGE = 16384;
double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int env_int(const char* n, int d) {
    const char* e = std::getenv(n);
    return e && *e ? std::atoi(e) : d;
}
}  // namespace

bool ExpertStream::wanted(const gguf::File& f, const mtl::Context& cx) {
    const char* e = std::getenv("STRATA_METAL_STREAM");
    if (e && std::string(e) == "1") return true;
    if (e && std::string(e) == "0") return false;
    uint64_t exp = 0;
    for (const auto& kv : f.tensors)
        if (kv.first.find("_exps") != std::string::npos) exp += kv.second.bytes;
    return (double) exp > 0.6 * (double) cx.recommended_working_set();
}

ExpertStream::~ExpertStream() {
    {
        std::lock_guard<std::mutex> g(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : io_) t.join();
    if (fd_ >= 0) ::close(fd_);
}

bool ExpertStream::open(const gguf::File& f, std::string& err) {
    fd_ = ::open(f.path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "metal stream: cannot open " + f.path; return false; }
    fcntl(fd_, F_NOCACHE, 1);
    fcntl(fd_, F_RDAHEAD, 0);
    lay_.resize(NL);
    for (int l = 0; l < NL; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        const gguf::Tensor* g = f.find(b + "ffn_gate_exps.weight");
        const gguf::Tensor* u = f.find(b + "ffn_up_exps.weight");
        const gguf::Tensor* d = f.find(b + "ffn_down_exps.weight");
        if (!g || !u || !d) { err = "metal stream: layer " + std::to_string(l) + " has no routed experts"; return false; }
        lay_[(size_t) l] = {g->offset, g->bytes / NE, u->offset, u->bytes / NE, d->offset, d->bytes / NE, -1};
    }
    // size classes: slots of the same page count share one LRU pool
    std::map<uint64_t, int> by_pages;
    for (int l = 0; l < NL; ++l) {
        const Lay& x = lay_[(size_t) l];
        const uint64_t pages = (x.gs + x.us + x.ds + PAGE - 1) / PAGE;
        auto it = by_pages.find(pages);
        if (it == by_pages.end()) {
            it = by_pages.emplace(pages, (int) cls_.size()).first;
            Cls c;
            c.pages = pages;
            cls_.push_back(c);
        }
        lay_[(size_t) l].cls = it->second;
    }
    nio_ = std::max(1, env_int("STRATA_METAL_IO_THREADS", 6));
    split_ = std::max(1, std::min(8, env_int("STRATA_METAL_READ_SPLIT", 2)));
    pre_k_ = env_int("STRATA_METAL_PREFETCH", 10);
    pre_k2_ = env_int("STRATA_METAL_PREFETCH2", 0);
    prefetch_ = pre_k_ > 0;
    // timing only (WRONG answers): every missing expert is pointed at slot 0, nothing is read
    fake_ = env_int("STRATA_METAL_STREAM_FAKE", 0) != 0;
    return true;
}

bool ExpertStream::read(void* dst, uint64_t len, uint64_t off) const {
    uint8_t* p = (uint8_t*) dst;
    while (len > 0) {
        const ssize_t r = ::pread(fd_, p, (size_t) std::min<uint64_t>(len, 64ull << 20), (off_t) off);
        if (r <= 0) return false;
        p += r;
        off += (uint64_t) r;
        len -= (uint64_t) r;
    }
    return true;
}

bool ExpertStream::load_dense(gguf::File& f, mtl::Context& cx, const std::vector<std::string>& skip, mtl::Buffer& out,
                              std::string& err) {
    const double t0 = now_ms();
    std::vector<gguf::Tensor*> ts;
    uint64_t tot = 0;
    for (auto& kv : f.tensors) {
        gguf::Tensor& t = kv.second;
        if (t.name.find("_exps") != std::string::npos) continue;
        if (std::find(skip.begin(), skip.end(), t.name) != skip.end()) { t.offset = 0; continue; }
        ts.push_back(&t);
        tot += (t.bytes + 255) / 256 * 256;
    }
    out = cx.alloc((tot + PAGE - 1) / PAGE * PAGE);
    std::vector<uint64_t> src(ts.size()), dst(ts.size());
    uint64_t o = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        src[i] = ts[i]->offset;
        dst[i] = o;
        o += (ts[i]->bytes + 255) / 256 * 256;
    }
    std::atomic<size_t> next{0};
    std::atomic<bool> ok{true};
    std::vector<std::thread> th;
    for (int k = 0; k < 4; ++k)
        th.emplace_back([&] {
            for (;;) {
                const size_t i = next++;
                if (i >= ts.size()) break;
                if (!read(out.as<uint8_t>() + dst[i], ts[i]->bytes, src[i])) ok = false;
            }
        });
    for (auto& x : th) x.join();
    if (!ok) { err = "metal stream: reading the dense tensors failed"; return false; }
    for (size_t i = 0; i < ts.size(); ++i) ts[i]->offset = dst[i];
    std::fprintf(stderr, "strata metal stream: %zu dense tensors read into memory (%.2f GB, %.1f s; %zu left out)\n", ts.size(),
                 tot / 1e9, (now_ms() - t0) / 1e3, skip.size());
    return true;
}

int ExpertStream::cls_of(int s) const {   // the class ranges are contiguous
    int ci = 0;
    while (!(s >= cls_[(size_t) ci].base && s < cls_[(size_t) ci].base + cls_[(size_t) ci].n)) ++ci;
    return ci;
}

void ExpertStream::lru_unlink(int s) {
    Slot& x = slots_[(size_t) s];
    Cls& k = cls_[(size_t) cls_of(s)];
    if (x.prev >= 0) slots_[(size_t) x.prev].next = x.next; else k.head = x.next;
    if (x.next >= 0) slots_[(size_t) x.next].prev = x.prev; else k.tail = x.prev;
    x.prev = x.next = -1;
}

void ExpertStream::lru_back(int s) {
    lru_unlink(s);
    Cls& k = cls_[(size_t) cls_of(s)];
    Slot& x = slots_[(size_t) s];
    x.prev = k.tail;
    x.next = -1;
    if (k.tail >= 0) slots_[(size_t) k.tail].next = s;
    k.tail = s;
    if (k.head < 0) k.head = s;
}

void ExpertStream::lru_front(int s) {
    lru_unlink(s);
    Cls& k = cls_[(size_t) cls_of(s)];
    Slot& x = slots_[(size_t) s];
    x.next = k.head;
    x.prev = -1;
    if (k.head >= 0) slots_[(size_t) k.head].prev = s;
    k.head = s;
    if (k.tail < 0) k.tail = s;
}

int ExpertStream::victim(int ci) {
    Cls& k = cls_[(size_t) ci];
    for (int s = k.tail; s >= 0; s = slots_[(size_t) s].prev)
        if (slots_[(size_t) s].state.load() == 0) {
            Slot& x = slots_[(size_t) s];
            if (x.layer >= 0) {
                const int key = x.layer * NE + x.e;
                where_[(size_t) key] = -1;
                smap_.as<uint32_t>()[key] = 0xFFFFFFFFu;
            }
            x.layer = x.e = -1;
            return s;
        }
    return -1;
}

// One piece of an expert: slice part / split_ (gate, up, down), piece part % split_ of it.  The pieces of one expert
// go to several I/O threads, so a single miss is read with several requests in flight (its latency, not only the
// SSD's throughput, is what a layer waits for).
void ExpertStream::read_part(int s, int l, int e, int part) {
    const Lay& x = lay_[(size_t) l];
    const int k = part / split_, p = part % split_;
    const uint64_t len = k == 0 ? x.gs : (k == 1 ? x.us : x.ds);
    const uint64_t src = (k == 0 ? x.g : (k == 1 ? x.u : x.d)) + (uint64_t) e * len;
    const uint64_t dst = k == 0 ? 0 : (k == 1 ? x.gs : x.gs + x.us);
    const uint64_t a = len * (uint64_t) p / (uint64_t) split_ / 4096 * 4096;
    const uint64_t b = p + 1 == split_ ? len : len * (uint64_t) (p + 1) / (uint64_t) split_ / 4096 * 4096;
    if (b > a && !read(host_[(size_t) s] + dst + a, b - a, src + a)) {
        std::fprintf(stderr, "strata metal stream: reading expert %d of layer %d failed\n", e, l);
        std::exit(6);
    }
    bytes_read_ += b - a;
}

void ExpertStream::io_main() {
    for (;;) {
        Job j;
        {
            std::unique_lock<std::mutex> g(mu_);
            cv_.wait(g, [&] { return stop_ || !urgent_q_.empty() || !pre_q_.empty(); });
            if (stop_) return;
            if (!urgent_q_.empty()) { j = urgent_q_.front(); urgent_q_.pop_front(); }
            else { j = pre_q_.front(); pre_q_.pop_front(); }
            slots_[(size_t) j.slot].state = 2;
        }
        read_part(j.slot, j.layer, j.e, j.part);
        bool done = false;
        {
            std::lock_guard<std::mutex> g(mu_);
            Slot& x = slots_[(size_t) j.slot];
            if (--x.left == 0) {
                smap_.as<uint32_t>()[j.layer * NE + j.e] = code_[(size_t) j.slot];
                x.state = 0;
                --outstanding_;
                ++reads_;
                done = true;
            }
        }
        if (done) done_cv_.notify_all();
    }
}

void ExpertStream::submit(int s, int l, int e, bool urgent) {
    Slot& x = slots_[(size_t) s];
    x.layer = (int16_t) l;
    x.e = (int16_t) e;
    x.state = 1;
    x.pre = !urgent;
    x.left = 3 * split_;
    ++outstanding_;
    where_[(size_t) (l * NE + e)] = s;
    for (int p = 0; p < 3 * split_; ++p) (urgent ? urgent_q_ : pre_q_).push_back({s, l, e, p});
}

// a queued read-ahead nobody wants: its pieces dropped, the slot freed (only before any piece has started)
void ExpertStream::cancel_slot(int s) {
    Slot& x = slots_[(size_t) s];
    for (auto it = pre_q_.begin(); it != pre_q_.end();) it = it->slot == s ? pre_q_.erase(it) : it + 1;
    where_[(size_t) (x.layer * NE + x.e)] = -1;
    x.layer = x.e = -1;
    x.state = 0;
    x.pre = false;
    x.left = 0;
    --outstanding_;
    ++pre_cancel_;
    lru_back(s);
}

// a read-ahead that is wanted now: its queued pieces go first
void ExpertStream::promote_slot(int s) {
    std::vector<Job> mine;
    for (auto it = pre_q_.begin(); it != pre_q_.end();)
        if (it->slot == s) { mine.push_back(*it); it = pre_q_.erase(it); }
        else ++it;
    for (auto it = mine.rbegin(); it != mine.rend(); ++it) urgent_q_.push_front(*it);
}

bool ExpertStream::make_arena(mtl::Context& cx, uint64_t budget, const std::string& profile, std::string& err) {
    const double t0 = now_ms();
    // the ranking: the profile's pairs, then every pair it misses, layer-interleaved
    std::vector<std::pair<int, int>> rank;
    std::vector<uint8_t> seen((size_t) NL * NE, 0);
    if (FILE* f = std::fopen(profile.c_str(), "rb")) {
        char magic[4];
        uint32_t h[5];
        if (std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "STRP", 4) == 0 && std::fread(h, 4, 5, f) == 5 &&
            h[1] == (uint32_t) NL && h[2] == (uint32_t) NE) {
            for (uint32_t i = 0; i < h[4]; ++i) {
                uint16_t p[2];
                if (std::fread(p, 2, 2, f) != 2) break;
                if (p[0] < NL && p[1] < NE && !seen[(size_t) p[0] * NE + p[1]]) {
                    seen[(size_t) p[0] * NE + p[1]] = 1;
                    rank.push_back({p[0], p[1]});
                }
            }
        }
        std::fclose(f);
    } else {
        std::fprintf(stderr, "strata metal stream: no expert profile at %s: the arena starts layer-interleaved\n", profile.c_str());
    }
    const size_t from_profile = rank.size();
    for (int e = 0; e < NE; ++e)
        for (int l = 0; l < NL; ++l)
            if (!seen[(size_t) l * NE + e]) rank.push_back({l, e});
    // the pool of each size class: the top-ranked pairs that fit in the budget
    uint64_t used = 0;
    std::vector<std::pair<int, int>> fill;
    for (const auto& p : rank) {
        const Cls& k = cls_[(size_t) lay_[(size_t) p.first].cls];
        if (used + k.pages * PAGE > budget) break;
        used += k.pages * PAGE;
        ++cls_[(size_t) lay_[(size_t) p.first].cls].n;
        fill.push_back(p);
    }
    int base = 0;
    for (auto& k : cls_) {
        if (k.n < 64 && !fill.empty()) {   // a class needs room for at least a few windows' experts
            const int add = 64 - k.n;
            k.n = 64;
            used += (uint64_t) add * k.pages * PAGE;
        }
        k.base = base;
        base += k.n;
    }
    // the slots placed in chunks of at most maxBufferLength (and 8 GB), up to 4 (the kernels' buffers 0, 7, 8, 9)
    const uint64_t cmax = std::min<uint64_t>(cx.max_buffer_length(), 8ull << 30) / PAGE * PAGE;
    code_.assign((size_t) base, 0);
    host_.assign((size_t) base, nullptr);
    std::vector<uint64_t> csize;
    std::vector<std::pair<int, uint64_t>> at((size_t) base);
    uint64_t cur = 0;
    for (const auto& k : cls_)
        for (int s = k.base; s < k.base + k.n; ++s) {
            if (csize.empty() || cur + k.pages * PAGE > cmax) {
                if (!csize.empty()) csize.back() = cur;
                csize.push_back(0);
                cur = 0;
            }
            at[(size_t) s] = {(int) csize.size() - 1, cur / PAGE};
            cur += k.pages * PAGE;
        }
    if (!csize.empty()) csize.back() = cur;
    if (csize.size() > 4) { err = "metal stream: the arena needs more than 4 buffers (lower STRATA_METAL_ARENA_GB)"; return false; }
    for (uint64_t b : csize) { chunks_.push_back(cx.alloc(b)); arena_bytes_ += b; }
    for (int s = 0; s < base; ++s) {
        code_[(size_t) s] = ((uint32_t) at[(size_t) s].first << 28) | (uint32_t) at[(size_t) s].second;
        host_[(size_t) s] = chunks_[(size_t) at[(size_t) s].first].as<uint8_t>() + at[(size_t) s].second * PAGE;
    }
    smap_ = cx.alloc((size_t) NL * NE * 4);
    std::memset(smap_.ptr, 0xFF, (size_t) NL * NE * 4);
    slots_ = std::vector<Slot>((size_t) base);
    where_.assign((size_t) NL * NE, -1);
    for (auto& k : cls_) {
        for (int s = k.base; s < k.base + k.n; ++s) {
            slots_[(size_t) s].prev = s == k.base ? -1 : s - 1;
            slots_[(size_t) s].next = s + 1 == k.base + k.n ? -1 : s + 1;
        }
        k.head = k.n ? k.base : -1;
        k.tail = k.n ? k.base + k.n - 1 : -1;
    }
    for (int i = 0; i < nio_; ++i) io_.emplace_back([this] { io_main(); });
    // the fill: the ranked pairs, in file order, on the I/O threads; the best-ranked ends at the LRU front
    std::vector<std::pair<int, int>> order = fill;
    std::sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
        return lay_[(size_t) a.first].g + (uint64_t) a.second * lay_[(size_t) a.first].gs <
               lay_[(size_t) b.first].g + (uint64_t) b.second * lay_[(size_t) b.first].gs;
    });
    std::vector<int> next_free(cls_.size());
    for (size_t c = 0; c < cls_.size(); ++c) next_free[c] = cls_[c].base;
    {
        std::lock_guard<std::mutex> g(mu_);
        for (const auto& p : order) {
            const int c = lay_[(size_t) p.first].cls;
            submit(next_free[(size_t) c]++, p.first, p.second, true);
        }
    }
    cv_.notify_all();
    {
        std::unique_lock<std::mutex> g(mu_);
        done_cv_.wait(g, [&] { return outstanding_ == 0; });
        // the LRU order = the rank order (best at the front)
        for (auto it = fill.rbegin(); it != fill.rend(); ++it) lru_front(where_[(size_t) (it->first * NE + it->second)]);
    }
    fill_s_ = (now_ms() - t0) / 1e3;
    fill_bytes_ = bytes_read_.load();
    uint64_t pairs = 0;
    for (const auto& k : cls_) pairs += (uint64_t) k.n;
    std::fprintf(stderr, "strata metal stream: expert arena %.2f GB, %llu slots in %zu size classes, %zu experts read from the "
                         "profile's top (%zu ranked) in %.1f s (%.2f GB/s); %d I/O threads, read-ahead top %d; %zu buffers\n",
                 arena_bytes_ / 1e9, (unsigned long long) pairs, cls_.size(), fill.size(), from_profile, fill_s_,
                 fill_bytes_ / 1e9 / std::max(1e-9, fill_s_), nio_, prefetch_ ? pre_k_ : 0, chunks_.size());
    (void) err;
    return true;
}

// the experts a predicted routing names, most likely first (by the best logit over the window's tokens)
void ExpertStream::ahead(int ln, int T, const float* pred, int K) {
    K = std::min(K, 32);
    std::pair<float, int> pw[8 * 32];
    int np = 0;
    for (int t = 0; t < T; ++t) {
        const float* lg = pred + (size_t) t * NE;
        int top[32];
        int nt = 0;
        for (int e = 0; e < NE; ++e) {
            if (nt < K) { top[nt++] = e; continue; }
            int m = 0;
            for (int j = 1; j < K; ++j) if (lg[top[j]] < lg[top[m]]) m = j;
            if (lg[e] > lg[top[m]]) top[m] = e;
        }
        for (int j = 0; j < nt; ++j) {
            int q = 0;
            while (q < np && pw[q].second != top[j]) ++q;
            if (q == np) pw[np++] = {lg[top[j]], top[j]};
            else pw[q].first = std::max(pw[q].first, lg[top[j]]);
        }
    }
    std::sort(pw, pw + np, [](const auto& a, const auto& b) { return a.first > b.first; });
    bool any = false;
    {
        std::lock_guard<std::mutex> g(mu_);
        for (int j = 0; j < np; ++j) {
            const int key = ln * NE + pw[j].second;
            const int s0 = where_[(size_t) key];
            if (s0 >= 0) { lru_front(s0); continue; }
            const int s = victim(lay_[(size_t) ln].cls);
            if (s < 0) break;
            submit(s, ln, pw[j].second, false);
            lru_front(s);
            ++pre_issued_;
            any = true;
        }
    }
    if (any) cv_.notify_all();
}

void ExpertStream::at_layer(int l, int T, int topk, const int32_t* ids, const float* pred, const float* pred2) {
    const double t0 = now_ms();
    if (l == 0) ++windows_;
    int want[8 * 16];
    int nw = 0;
    for (int i = 0; i < T * topk; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= NE) continue;
        bool dup = false;
        for (int j = 0; j < nw; ++j) dup |= want[j] == e;
        if (!dup) want[nw++] = e;
    }
    if (fake_) {
        for (int j = 0; j < nw; ++j)
            if (where_[(size_t) (l * NE + want[j])] < 0) smap_.as<uint32_t>()[l * NE + want[j]] = code_[0];
        host_ms_ += now_ms() - t0;
        return;
    }
    std::vector<int> wait;
    {
        std::lock_guard<std::mutex> g(mu_);
        // read-aheads of this layer (or older) that were not started and are not wanted: dropped (the slot goes free)
        {
            std::vector<int> drop;
            for (const Job& j : pre_q_) {
                if (j.layer > l || slots_[(size_t) j.slot].state.load() != 1) continue;
                bool w = false;
                for (int q = 0; q < nw; ++q) w |= j.layer == l && want[q] == j.e;
                if (!w && std::find(drop.begin(), drop.end(), j.slot) == drop.end()) drop.push_back(j.slot);
            }
            for (int s : drop) cancel_slot(s);
        }
        for (int j = 0; j < nw; ++j) {
            const int e = want[j], key = l * NE + e;
            ++lookups_;
            int s = where_[(size_t) key];
            if (s >= 0) {
                Slot& x = slots_[(size_t) s];
                const int st = x.state.load();
                if (st == 0) { if (x.pre) ++pre_hits_; else ++hits_; }
                else {
                    ++pre_hits_;
                    ++late_;
                    promote_slot(s);   // a queued read-ahead becomes urgent
                    wait.push_back(s);
                }
                x.pre = false;
                lru_front(s);
                continue;
            }
            ++misses_;
            s = victim(lay_[(size_t) l].cls);
            if (s < 0) { std::fprintf(stderr, "strata metal stream: no free slot (layer %d)\n", l); std::exit(6); }
            submit(s, l, e, true);
            lru_front(s);
            wait.push_back(s);
        }
    }
    if (!wait.empty()) {
        cv_.notify_all();
        const double tw = now_ms();
        std::unique_lock<std::mutex> g(mu_);
        done_cv_.wait(g, [&] {
            for (int s : wait) if (slots_[(size_t) s].state.load() != 0) return false;
            return true;
        });
        wait_ms_ += now_ms() - tw;
    }
    // read-ahead: layer l + 1 (each token's top pre_k_ of the predicted router logits), then layer l + 2 (top pre_k2_)
    if (prefetch_ && pred && l + 1 < NL) ahead(l + 1, T, pred, pre_k_);
    if (pre_k2_ > 0 && pred2 && l + 2 < NL) ahead(l + 2, T, pred2, pre_k2_);
    host_ms_ += now_ms() - t0;
}

void ExpertStream::reset_stats() {
    lookups_ = hits_ = pre_hits_ = late_ = misses_ = pre_issued_ = pre_cancel_ = windows_ = 0;
    bytes_read_ = 0;
    reads_ = 0;
    wait_ms_ = host_ms_ = 0;
}

void ExpertStream::report(FILE* f) const {
    if (!ready()) return;
    const double w = std::max<uint64_t>(1, windows_);
    std::fprintf(f,
                 "expert stream            %llu windows: %llu distinct lookups (%.1f per window), hits %.4f, read-ahead hits "
                 "%.4f (%llu of them late), misses %.4f (%.1f per window); read-aheads %llu (%.1f per window, %llu dropped); %.1f MB "
                 "read per window (%llu reads); host %.2f ms per window, of it waiting %.2f ms\n",
                 (unsigned long long) windows_, (unsigned long long) lookups_, lookups_ / w,
                 hits_ / std::max(1.0, (double) lookups_), pre_hits_ / std::max(1.0, (double) lookups_),
                 (unsigned long long) late_, misses_ / std::max(1.0, (double) lookups_), misses_ / w,
                 (unsigned long long) pre_issued_, pre_issued_ / w, (unsigned long long) pre_cancel_,
                 (bytes_read_.load() - 0) / 1e6 / w, (unsigned long long) reads_.load(), host_ms_ / w, wait_ms_ / w);
}

}  // namespace strata::metal_engine
