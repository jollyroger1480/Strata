# Strata build result — gfx1030 + gfx1034, 2026-10-02

The HIP engine is rebuilt for both cards (`gfx1030,gfx1034` is in the `strata` binary).
qwen35 kernels added beside the qwen4exp sigmoid path:

- `gdn_out_norm_silu`: RMS, then `silu(z) = z * sigmoid(z)` (llama.cpp `build_norm_gated`)
- `gdn_out_norm_silu_split` and `gdn_step_split`: value-heads split across HIP devices

`build-hip/qwen35_gdn_multi` exited 0 on the RX 6500 XT (HIP 0, 4 GiB) and the RX 6950 XT (HIP 1, 16 GiB). Device 0 SiLU matched the CPU reference (rel 1.991e-08). The two-device step matched the one-device step exactly. `strata generate` still requires a qwen4exp expert pack, so this binary does not load the Qwythos GGUF.

```
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES="gfx1030;gfx1034" \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip --target qwen35_gdn_multi strata -j8
```

# Earlier note — gfx1030 only, 2026-10-02

## Step 1 — Repo

Cloned https://github.com/Niko1221/Strata (depth 1) to
`/data/projects/Strata` (case-sensitive; `/data/projects/strata` does not exist).

## Step 2 — HIP build for gfx1030: SUCCESS

Exact cmake line:

```
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-hip --target strata -j8
```

Notes:
- Without `-DCMAKE_PREFIX_PATH=/opt/rocm`, find_package fails for `hip`,
  then `hipblas`, then `hipblas-common` (ROCm 7.1.25424 here). With the
  prefix set, configure completes cleanly.
- gfx1030 is a supported community-validated target per docs/AMD_HIP.md
  §RDNA2 (v_dot4_i32_i8, wave32).
- Build result: `[100%] Built target strata`, exit 0. Binary:
  `/data/projects/Strata/build-hip/strata` (8.6 MB). Full log:
  `/tmp/strata-build.log`. The binary runs and prints its usage.

Per instructions the 4 GB RX 6500 XT (gfx1034) was NOT added to the build.

## Step 3/5 — Verdict: Qwythos-9B-v2 CANNOT be loaded by Strata.
### This is not fixable with a small loader change in days. STOPPED per step 5.

Strata is a single-model engine hard-specialized to Qwen3.8-Flash-Next
(qwen4exp): a MoE model whose experts live in an SSD "PLE" lookup shard.
It has NO generic GGUF compute graph.

The local model /data/hermes/models/qwythos-9b-v2/Qwythos-9B-v2-MTP-Q6_K.gguf
is a different architecture entirely: `general.architecture = qwen35`,
a DENSE hybrid Gated-DeltaNet (ssm.conv_kernel=4, ssm.state_size=128,
ssm.group_count=16, ssm.time_step_rank=32, full_attention_interval=4,
33 blocks, 4096 embedding, YaRN 1M ctx, 248320 tokens). Verified by
parsing the GGUF metadata directly: it has ZERO expert keys
(no expert_count, no expert_used_count, no router weights, 442 dense
tensors).

### Files that assume Flash-Next expert packs (evidence):

1. `include/strata/artifact/gguf_reader.hpp:581-604` — the architecture
   guard: `if (arch->s != "qwen4exp") return "architecture is '...', this
   engine requires 'qwen4exp'"`, plus Qwen4ExpGuard requiring
   `qwen4exp.expert_count` / `expert_used_count` keys the qwen35 GGUF
   does not carry.
2. `src/program/generate.cpp:1820-1825` — reads MoE shape from
   `qwen4exp.expert_count` / `expert_used_count`.
3. `src/program/generate.cpp:1615,2644-2658` — `--ple-gguf` is a REQUIRED
   flag ("WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED"); FileExpertSource
   maps `experts.bin` / the GGUF expert shards.
4. `src/core/layout.cpp:60` — every layer's tensor layout hardcodes the
   MoE router `ffn_gate_inp.weight`.
5. `src/core/expert_source.cpp` (109 KB) + `src/core/expert_cache.cpp` —
   the entire expert-arena / PLE machinery, no dense-model path.
6. `src/core/layer.cpp` (~102 KB) + `src/kernels/cuda/*.cu` +
   `src/kernels/gdn_parity.cpp` — the compute graph and its kernels
   implement qwen4exp attention + QSA/PLE, not qwen35's Gated-DeltaNet
   recurrent layers.

### Why not "a real change in days":

Relaxing the arch guard would be the trivial part. Underneath, Strata
would need a NEW model class for qwen35: Gated-DeltaNet state-space
layers (chunked delta rule, conv, SSM state) with new HIP kernels, a new
weight layout, and removal of the PLE/expert pipeline for this model.
That is a multi-month engine port, not a loader tweak.

### NOT done (per instructions):

- No Strata config was pointed at llama-server and called "support".
  The existing llama.cpp build at
  /data/hermes/llama.cpp-upstream/build/bin/llama-server remains the only
  engine that runs Qwythos-9B-v2, unchanged.
- Nothing committed or pushed (working tree has only build-hip/ outputs
  and this file).
- The v2 GGUF was not touched.

## Done definition

BUILD-RESULT.md exists; the HIP binary built AND runs (usage output
above). Loading Qwythos on it is impossible by design, with file paths
listed above.
