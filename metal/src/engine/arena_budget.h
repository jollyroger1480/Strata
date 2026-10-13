// metal/src/engine/arena_budget.h - how big the SSD-streaming expert arena may be (stream.h), as plain arithmetic so it
// can be tested without a Mac (metal/tests/arena_budget_test.cpp).
//
// The arena is anonymous memory the GPU keeps resident: macOS cannot page it out or drop it like file cache.  It must
// leave room for macOS and the desktop, and for the other big memory the engine itself uses while it runs:
//   keep      macOS and the desktop: max(7 GB, 15% of the RAM) - STRATA_METAL_KEEP_FREE_GB sets it.  The M5 Max
//             (128 GB) froze with an 89 GB arena and 7 GB kept beside a 50 GB n-gram table; with 15% it keeps 20.6 GB.
//             On the 32 GB M4, 7 GB kept left macOS 55% of its memory free all run (memory_pressure) and decodes
//             7.8 tok/s; 12 GB kept cut the arena from 9.4 to 4.4 GB and decode to 4.8 tok/s (Q2_0, 200 tokens).
//   ple       the n-gram table: mapped whole in RAM (--ple-io ram) it is all of it; read row by row (the default)
//             its row cache and the OS's read-ahead, 2 GB.
//   GPU       the Metal working set less what is already allocated, less 1 GB.
// STRATA_METAL_ARENA_GB sets the arena outright.  The arena is placed just before it is filled, and the memory macOS
// can give is asked again then: when it has dropped (another program), the arena shrinks (arena_shrink).
#pragma once

#include <algorithm>

namespace strata::metal_engine {

struct ArenaMem {
    double phys = 0;          // bytes of RAM
    double working_set = 0;   // the Metal device's recommended working set
    double allocated = 0;     // already allocated by the engine (Metal buffers)
    double available = 0;     // what macOS can give now: free + inactive + speculative + purgeable pages
    double ple_mapped = 0;    // the n-gram table's bytes when it is mapped whole in RAM, else 0
    double keep_gb = -1;      // STRATA_METAL_KEEP_FREE_GB (< 0: unset)
    double arena_gb = -1;     // STRATA_METAL_ARENA_GB (< 0: unset)
};

constexpr double kGB = 1e9;

inline double arena_keep(const ArenaMem& m) {
    return m.keep_gb >= 0 ? m.keep_gb * kGB : std::max(7.0 * kGB, 0.15 * m.phys);
}

inline double arena_ple_reserve(const ArenaMem& m) { return m.ple_mapped > 0 ? m.ple_mapped : 2.0 * kGB; }

/// The arena's bytes (at least 1 GB: a smaller arena still runs, at a low hit rate).
inline double arena_budget(const ArenaMem& m) {
    if (m.arena_gb >= 0) return m.arena_gb * kGB;
    const double gpu = m.working_set - m.allocated - 1.0 * kGB;
    const double ram = m.available - arena_keep(m) - arena_ple_reserve(m);
    return std::max(1.0 * kGB, std::min(gpu, ram));
}

/// Just before the arena is filled: the bytes it may still have when macOS can now give `available_now` (it shrinks,
/// never grows; the explicit STRATA_METAL_ARENA_GB is kept as it is).
inline double arena_shrink(const ArenaMem& m, double planned, double available_now) {
    if (m.arena_gb >= 0) return planned;
    ArenaMem now = m;
    now.available = available_now;
    return std::min(planned, arena_budget(now));
}

}  // namespace strata::metal_engine
