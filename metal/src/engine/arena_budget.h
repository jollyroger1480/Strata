// metal/src/engine/arena_budget.h - how big the SSD-streaming expert arena may be (stream.h), as plain arithmetic so it
// can be tested without a Mac (metal/tests/arena_budget_test.cpp).
//
// The arena is anonymous memory the GPU keeps resident: macOS cannot page it out or drop it like file cache.  It must
// leave room for macOS and the desktop, and for the other big memory the engine itself uses while it runs:
//   keep      macOS and the desktop: max(7 GB, 15% of the RAM) - STRATA_METAL_KEEP_FREE_GB sets it.  The M5 Max
//             (128 GB) froze with an 89 GB arena and 7 GB kept beside a 50 GB n-gram table; with 15% it keeps 20.6 GB.
//             The 32 GB M4 (IQ3_XXS after a 6K prompt, 5 interleaved runs, memory_pressure every 3 s, vm_stat
//             swapouts): 7 GB kept + 2 GB reserve 5.93 tok/s (arena ~8.5 GB, least free 50%), 7 + 0.5 6.59 (~10 GB,
//             49%), 5 + 0.5 7.47 (~12 GB, 43%); no swap-out in any run.  But a 34-minute server soak with 5 GB kept
//             swapped out 93 MB while its 11.7 GB arena filled (none after): 7 GB again.
//   ple       the n-gram table: mapped whole in RAM (--ple-io ram) it is all of it; read row by row (the default,
//             F_NOCACHE: no file cache) only its bounded row cache, 1M rows of at most 192 bytes: 0.5 GB reserved.
//   GPU       the Metal working set less what is already allocated, less 1 GB.
// STRATA_METAL_ARENA_GB sets the arena outright.  The arena is placed just before it is filled, and the memory macOS
// can give is asked again then: when it has dropped (another program), the arena shrinks (arena_shrink).
// "Available" (arena_avail) counts free and purgeable pages and the file cache macOS drops without writing anything:
// inactive pages, but no more of them than there are file-backed pages.  Inactive anonymous memory (other programs')
// is left out: macOS frees it only by compressing or swapping it (the soak's 93 MB swapped out while an arena planned
// with every inactive page filled).  Free + purgeable alone is too little: at the engine's start the model file's own
// pages fill the cache (6.4 GB free + purgeable on the 32 GB M4 against 15 GB a minute later).  The arena is filled in steps (STRATA_METAL_ARENA_STEP_GB, default 1.5): before each step the free
// memory is asked again, and the fill stops - the rest of the arena's slots unused - when a step would take macOS
// below what it keeps (arena_grow).
#pragma once

#include <algorithm>

namespace strata::metal_engine {

struct ArenaMem {
    double phys = 0;          // bytes of RAM
    double working_set = 0;   // the Metal device's recommended working set
    double allocated = 0;     // already allocated by the engine (Metal buffers)
    double available = 0;     // what macOS can give now without paging anything out (arena_avail)
    double ple_mapped = 0;    // the n-gram table's bytes when it is mapped whole in RAM, else 0
    double keep_gb = -1;      // STRATA_METAL_KEEP_FREE_GB (< 0: unset)
    double arena_gb = -1;     // STRATA_METAL_ARENA_GB (< 0: unset)
};

constexpr double kGB = 1e9;

// The GPU keeps the whole arena (and the engine's other buffers) resident through a residency set, i.e. wired. Measured on the
// M5 Max (137.4 GB), UD-Q6_K_XL: 84 GB of arena + 7.5 GB allocated (91.5 GB, 67%) ran with no swap-out; 90 GB + 7.5 GB (97.5 GB)
// made macOS swap 29,000 pages in 8 s with a "warn" pressure level and the Mac went on to freeze; UD-Q5 with 90 GB + 5.7 GB ran,
// near the edge. So arena + allocated stay under this share of the RAM, whatever the free-memory count says.
constexpr double kWiredShare = 0.67;

/// Bytes macOS can give without compressing or swapping: free + purgeable + the clean file cache (inactive and
/// speculative pages, at most the file-backed ones).
inline double arena_avail(double free_b, double purgeable_b, double inactive_b, double speculative_b, double file_b) {
    return free_b + purgeable_b + std::min(inactive_b + speculative_b, file_b);
}

inline double arena_keep(const ArenaMem& m) {
    return m.keep_gb >= 0 ? m.keep_gb * kGB : std::max(7.0 * kGB, 0.15 * m.phys);
}

inline double arena_ple_reserve(const ArenaMem& m) { return m.ple_mapped > 0 ? m.ple_mapped : 0.5 * kGB; }

/// The arena's bytes (at least 1 GB: a smaller arena still runs, at a low hit rate).
inline double arena_budget(const ArenaMem& m) {
    if (m.arena_gb >= 0) return m.arena_gb * kGB;
    const double gpu = m.working_set - m.allocated - 1.0 * kGB;
    const double ram = m.available - arena_keep(m) - arena_ple_reserve(m);
    const double wired = kWiredShare * m.phys - m.allocated;
    return std::max(1.0 * kGB, std::min(std::min(gpu, ram), wired));
}

/// Just before the arena is filled: the bytes it may still have when macOS can now give `available_now` (it shrinks,
/// never grows; the explicit STRATA_METAL_ARENA_GB is kept as it is).
inline double arena_shrink(const ArenaMem& m, double planned, double available_now) {
    if (m.arena_gb >= 0) return planned;
    ArenaMem now = m;
    now.available = available_now;
    return std::min(planned, arena_budget(now));
}

/// The stepped fill: may the next `step` bytes of the arena be filled while macOS has `available_now` free + purgeable?
/// (the explicit STRATA_METAL_ARENA_GB always fills)
inline bool arena_grow(const ArenaMem& m, double available_now, double step) {
    if (m.arena_gb >= 0) return true;
    return available_now - step >= arena_keep(m) + arena_ple_reserve(m);
}

/// The fill as a plan, for the test: the pairs' bytes in rank order, steps of `step` bytes, `avail(filled)` the free
/// memory once `filled` bytes are in.  Returns how many pairs are filled.
template <class Avail>
inline size_t arena_fill_plan(const ArenaMem& m, const double* cost, size_t n, double step, Avail avail) {
    size_t i = 0;
    double filled = 0;
    while (i < n) {
        double s = 0;
        size_t j = i;
        while (j < n && (s < step || j == i)) s += cost[j++];
        if (!arena_grow(m, avail(filled), s)) break;
        filled += s;
        i = j;
    }
    return i;
}

}  // namespace strata::metal_engine
