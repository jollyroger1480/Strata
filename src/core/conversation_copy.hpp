#pragma once

#include "strata/core/conversation_cache.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <thread>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    cudaStream_t s = streams[(size_t) dev];
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

/// STRATA_TEST_SLOW_COPY_US=N (tests only): a parked conversation's running checkpoints copy as slowly as when the OS
/// has paged them out - N microseconds per MiB copied.  Unset: a plain copy.
inline long test_slow_copy_us() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_SLOW_COPY_US"); return e ? std::atol(e) : 0L; }();
    return v;
}

inline std::vector<ConversationCheckpoint> copy_checkpoints(const std::vector<ConversationCheckpoint>& cs) {
    std::vector<ConversationCheckpoint> out = cs;
    if (const long us = test_slow_copy_us(); us > 0) {
        size_t bytes = 0;
        for (const auto& c : out) bytes += c.bytes();
        std::this_thread::sleep_for(std::chrono::microseconds(us * (long) (bytes >> 20)));
    }
    return out;
}

} // namespace strata::core::conversation_detail
