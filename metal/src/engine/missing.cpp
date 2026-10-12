// metal/src/engine/missing.cpp - what a kernel entry point that has no Metal implementation yet does when called.
//
// Default: say which one and stop (exit code 86), so a run never continues on outputs nobody wrote.
// STRATA_METAL_STUBS=log: say it once per entry point and return (outputs untouched) - for listing which entry points a
// code path reaches, never for answers.
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace strata::metal_engine {

void missing(const char* name) {
    static const bool log_only = [] {
        const char* v = std::getenv("STRATA_METAL_STUBS");
        return v != nullptr && std::string(v) == "log";
    }();
    if (!log_only) {
        std::fprintf(stderr, "strata metal: %s has no Metal implementation yet\n", name);
        std::fflush(stderr);
        std::_Exit(86);
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(name).second) std::fprintf(stderr, "strata metal: STUB %s\n", name);
}

}  // namespace strata::metal_engine
