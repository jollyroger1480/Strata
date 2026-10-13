// metal/tests/arena_budget_test.cpp - the SSD-streaming arena's size rule (src/engine/arena_budget.h), on the sizes of
// the Macs it was measured on.  No GPU: plain arithmetic.  Exit 0 = every check passed.
#include "../src/engine/arena_budget.h"

#include <cmath>
#include <cstdio>

using namespace strata::metal_engine;

static int failures = 0;
static void check(bool ok, const char* what, double got) {
    std::printf("%s %-70s %.2f GB\n", ok ? "ok  " : "FAIL", what, got / kGB);
    failures += !ok;
}

int main() {
    // the M5 Max as it froze: 137.4 GB RAM, 115.4 GB working set, 7.0 GB allocated, 97.7 GB available
    ArenaMem m5{137.4 * kGB, 115.4 * kGB, 7.0 * kGB, 97.7 * kGB};
    check(std::fabs(arena_keep(m5) - 0.15 * 137.4 * kGB) < 1e6, "M5 Max: macOS keeps 15% of the RAM (20.6 GB)", arena_keep(m5));
    const double b5 = arena_budget(m5);
    check(b5 <= 97.7 * kGB - 20.6 * kGB - 0.5 * kGB + 1e6, "M5 Max: the arena leaves >= 20 GB + the n-gram reserve", b5);
    check(b5 > 70 * kGB, "M5 Max: still a large arena (UD-Q5's 98 GB of experts mostly cached)", b5);
    // the n-gram table mapped whole (--ple-io ram, 50.7 GB): it comes out of the arena
    ArenaMem m5r = m5;
    m5r.ple_mapped = 50.7 * kGB;
    check(arena_budget(m5r) <= b5 - 48.0 * kGB, "M5 Max, table in RAM: the arena gives up the table's 50.7 GB", arena_budget(m5r));
    // the M4 Mac mini, 32 GB: the 7 GB floor (15% would be 5.16 GB)
    ArenaMem m4{34.4 * kGB, 26.8 * kGB, 3.3 * kGB, 17.8 * kGB};
    check(std::fabs(arena_keep(m4) - 7.0 * kGB) < 1e6, "M4 32 GB: macOS keeps the 7 GB floor", arena_keep(m4));
    check(std::fabs(arena_budget(m4) - (17.8 - 7.0 - 0.5) * kGB) < 1e6, "M4 32 GB: available - 7 - 0.5", arena_budget(m4));
    ArenaMem m16{17.2 * kGB, 11 * kGB, 3 * kGB, 9 * kGB};
    check(std::fabs(arena_keep(m16) - 7.0 * kGB) < 1e6, "a 16 GB Mac: the 7 GB floor", arena_keep(m16));
    // the overrides
    ArenaMem o = m4;
    o.keep_gb = 12.0;
    check(std::fabs(arena_budget(o) - (17.8 - 12.0 - 0.5) * kGB) < 1e6, "STRATA_METAL_KEEP_FREE_GB=12", arena_budget(o));
    o.arena_gb = 10.0;
    check(std::fabs(arena_budget(o) - 10.0 * kGB) < 1e6, "STRATA_METAL_ARENA_GB=10 wins", arena_budget(o));
    check(std::fabs(arena_shrink(o, 10.0 * kGB, 1.0 * kGB) - 10.0 * kGB) < 1e6, "an explicit arena is not shrunk", 10.0 * kGB);
    // never below 1 GB; the GPU working set bounds it too
    ArenaMem tiny{16 * kGB, 10 * kGB, 9.5 * kGB, 8 * kGB};
    check(std::fabs(arena_budget(tiny) - 1.0 * kGB) < 1e6, "a 16 GB Mac with nothing left: 1 GB (low hit rate)", arena_budget(tiny));
    ArenaMem gpu = m5;
    gpu.available = 200 * kGB;
    check(std::fabs(arena_budget(gpu) - (115.4 - 7.0 - 1.0) * kGB) < 1e6, "the working set bounds it", arena_budget(gpu));
    // memory taken between planning and committing: it shrinks, never grows
    const double s1 = arena_shrink(m5, b5, 60.0 * kGB);
    check(s1 < b5 && std::fabs(s1 - (60.0 - 20.61 - 0.5) * kGB) < 0.1 * kGB, "less free when committing: it shrinks", s1);
    check(std::fabs(arena_shrink(m5, b5, 130.0 * kGB) - b5) < 1e6, "more free when committing: it stays as planned", b5);
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
