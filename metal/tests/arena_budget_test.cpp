// metal/tests/arena_budget_test.cpp - the SSD-streaming arena's size rule (src/engine/arena_budget.h), on the sizes of
// the Macs it was measured on.  No GPU: plain arithmetic.  Exit 0 = every check passed.
#include "../src/engine/arena_budget.h"

#include <cmath>
#include <cstdio>
#include <vector>

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
    // the stepped fill (arena_grow / arena_fill_plan): 1.5 GB steps of 1.75 MB experts on the M4 (7 GB kept + 0.5)
    ArenaMem s4 = m4;
    s4.available = 17.8 * kGB;
    check(arena_grow(s4, 17.8 * kGB, 1.5 * kGB), "a step with room: it fills", 17.8 * kGB);
    check(!arena_grow(s4, 8.9 * kGB, 1.5 * kGB), "a step that would leave < 7.5 GB: it stops", 8.9 * kGB);
    check(arena_grow(s4, 9.0 * kGB, 1.5 * kGB), "a step that leaves exactly 7.5 GB: it fills", 9.0 * kGB);
    ArenaMem pin = s4;
    pin.arena_gb = 12.0;
    check(arena_grow(pin, 1.0 * kGB, 1.5 * kGB), "STRATA_METAL_ARENA_GB: always fills", 1.0 * kGB);
    std::vector<double> cost(6000, 1.75e6);   // 10.5 GB of experts planned
    // macOS gives exactly what is filled: the plan fills all of it while free stays >= 7.5 GB
    const size_t all = arena_fill_plan(s4, cost.data(), cost.size(), 1.5 * kGB, [](double f) { return 18.5 * kGB - f; });
    check(all == cost.size(), "free memory follows the fill: every planned expert is filled", all * 1.75e6);
    // another program takes 4 GB while the arena fills (after 3 GB): the fill stops at the floor, a step short
    const size_t part = arena_fill_plan(s4, cost.data(), cost.size(), 1.5 * kGB,
                                        [](double f) { return 18.5 * kGB - f - (f >= 3 * kGB ? 4 * kGB : 0); });
    const double left = 18.5 * kGB - part * 1.75e6 - 4 * kGB;
    check(part < cost.size() && left >= 7.5 * kGB && left - 1.5 * kGB < 7.5 * kGB + 1.75e6,
          "4 GB taken meanwhile: it stops with >= 7.5 GB free, less than a step above", part * 1.75e6);
    // no room even for the first step: nothing is filled (the min slots still exist, unfilled)
    check(arena_fill_plan(s4, cost.data(), cost.size(), 1.5 * kGB, [](double) { return 8.0 * kGB; }) == 0,
          "no room for one step: nothing filled", 0);
    // what counts as available: the M4 at the engine's start (16 KiB pages: 6.3 GB free + purgeable, 10.4 GB
    // inactive, 14 GB file-backed) - the clean file cache counts; another program's inactive memory does not
    check(std::fabs(arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 14.1 * kGB) - 16.87 * kGB) < 1e6,
          "inactive file cache counts", arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 14.1 * kGB));
    check(std::fabs(arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 2.0 * kGB) - 8.37 * kGB) < 1e6,
          "inactive anonymous memory does not (only 2 GB file-backed)", arena_avail(6.3 * kGB, 0.07 * kGB, 10.4 * kGB, 0.1 * kGB, 2.0 * kGB));
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
