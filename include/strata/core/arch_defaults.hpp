// include/strata/core/arch_defaults.hpp - the per-architecture defaults of switches that are exact everywhere.
//
// Every gfx1151 (Strix Halo, RDNA3.5, unified memory) switch in the table was measured on Aurora (experiments/s23 .. s26,
// the c-best3-v4.2 config) with BYTE-IDENTICAL output against the same engine with the switch off: the same token ids at
// 4K / 32K / 64K and on four models, and a bitwise harness per kernel.  They are therefore on by default on that
// architecture and nowhere else: a CUDA build and every other HIP architecture run the code they always ran.
// Switches that change bits (STRATA_PF_GEMM, STRATA_PF_FUSED, STRATA_HC_UPMIX, STRATA_PA_FAST, STRATA_HIP_WMMA,
// STRATA_SELECT_WMMA, STRATA_HC_Q8: rounding-level, KL-gated) stay opt-in in THIS table; docs/STRIX_HALO.md lists them.  Five of them
// (HIP_WMMA, PF_GEMM, PA_FAST, SELECT_WMMA, HC_UPMIX) are on by default on an integrated Radeon through igpu_default_env() below.
//
// A switch the user set (to anything) is never overridden; STRATA_GFX1151_DEFAULTS=0 turns the whole table off.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// The environment settings that are the defaults of `gcn_arch` (hipDeviceProp_t::gcnArchName; "gfx1151" or
/// "gfx1151:sramecc-:xnack-"): empty for every other architecture.
std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch);

/// The integrated Radeons (Phoenix / Hawk Point 780M gfx1103, Strix Point 890M / 880M gfx1150, Strix Halo gfx1151) whose
/// matrix-core prompt set is on by default (153-igpu).  Pure architecture-name test; the caller adds the UMA test.
bool igpu_prompt_defaults_arch(const char* gcn_arch);

/// The matrix-core prompt set an integrated GPU takes by default: STRATA_HIP_WMMA, STRATA_PF_GEMM, STRATA_PA_FAST,
/// STRATA_SELECT_WMMA, STRATA_HC_UPMIX, all "1".  Empty unless `integrated` (hipDeviceProp_t::integrated: the device
/// shares the host's memory) AND the architecture is one of `igpu_prompt_defaults_arch`: a discrete card of the same
/// architecture family (RX 7900 XTX gfx1100, R9700 gfx1201, ...) and every CUDA / SYCL build get nothing.  These change
/// bits (KL-gated, notes-153-igpu.md); every switch keeps =0 as its off switch and STRATA_IGPU_DEFAULTS=0 turns the set off.
std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated);

/// Sets `arch_default_env(gcn_arch)` and, when `integrated`, `igpu_default_env(gcn_arch, true)` in this process's
/// environment where the user has not set the variable, prints one line to stderr per table when it set anything, and
/// returns the names it set.  `integrated` is true when every visible GPU is an integrated one.  Call once, before any
/// engine code reads its switches (they are read lazily, at first use).
std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated = false);

}  // namespace strata::core
