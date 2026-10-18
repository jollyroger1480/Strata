# Fused int8 DPAS expert products on Battlemage (branch f-151-b70)

Behaviour note, written before the kernel (2026-10-18), kept in step with `sycl/src/prefill/esimd_moe.dp.cpp`.

## What it replaces

The prompt path computes a routed expert as dequantize (gate/up and down to FP16, 9.8 MB written) then three oneMKL / SwiGLU
launches. On the Arc Pro B70 (IQ3_S, 4K prompt) that chain is ~38 us of GPU time per expert (dequant 25.6 us, GEMMs and
SwiGLU ~12 us) and 23% of the prompt. The fused path decodes the quantized weights in registers straight into the DPAS B
operand, so the FP16 copy never exists and the weights are read once, compressed (2.1 MB per expert instead of 9.8 MB written and
read back).

## Tiles and thresholds

* One hardware thread (ESIMD, sub-group 16, 256 GRF where the device builds it) owns 16 output rows of one expert, all of K, and
  `MT x 8` activation rows. `MT` = 2, 4, 8 or 12, chosen per launch from the mean number of 8-row tiles per expert in the group
  (<=2 -> 2, <=4 -> 4, <=8 -> 8, else 12; `STRATA_I8_MT` overrides). 12 keeps one decode for up to 96 rows.
* An expert with more rows than `8 MT` is split over row-block threads; each decodes the same weights again, so the redundancy is
  `ceil(rows / (8 MT))`.
* A 128-output work-group tile (the 16/32/64 rows x 128 outputs of the plan) is eight such threads side by side. There is no
  shared memory and no barrier, because the decode is per thread and the activations are read straight from L1/L2 as ready-made
  A tiles; this was chosen over SLM staging because the K step is one DPAS (32), the B fragment never leaves registers, and the
  tiled activations make each A operand one 256-byte block load.
* K step 32 (one s8 DPAS 8x16x32) with the 128-wide unrolling inside the superblock loop for IQ3_S (8 steps per 256).
* Group of up to 16 experts per launch (the same group the FP16 grouped XMX path holds in the prompt walk).

## Scales

* Activations: llama.cpp's q8_1 grid. Per row and 32 of K, `d = amax / 127`, `q = round(x / d)` half away from zero, stored as int8
  in DPAS A-operand order (`[8-row tile][K/32][8 rows x 32 bytes]`), with the 8 row scales of a tile contiguous.
* Weights: IQ3_S `d (1 + 2 s)` per 32 (exact in float), IQ4_NL `d` per 32; the int8 values are the codebook entries with their
  signs (IQ3_S: odd 1..15, IQ4_NL: -127..113), never rounded.
* Each 32-block's int32 dot product (<= 32 * 127 * 127 < 2^24: exact in int32 and exactly convertible to float) is converted and
  accumulated as `acc += float(dot) * (sw[n] * dx[m])` in float, K ascending. The only roundings are the float multiply/add and the
  activation quantization.
* After gate/up, `silu(gate) * up` (the FP16 path's formula, in float, not rounded to FP16) is quantized per 32 of the 640 FF
  values into the down input.

## Edge cases and fallback rules

* Row counts are padded to tiles of 8 with zero rows (stores are masked to the real rows). An expert with no rows is not in the
  group.
* IQ3_S rows are 110-byte blocks (2-byte aligned only): every field is read as dwords, two gathers and a funnel shift when it is
  not dword aligned, and the second gather is clamped to the matrix end (an unclamped read past the last block faulted the GPU once).
* Supported now: gate/up IQ3_S (ggml type 21) with K a multiple of 256, down IQ4_NL (20) with K a multiple of 32, output rows a
  multiple of 16. Any other type or shape makes `gate_up()` / `down()` return false and the caller keeps the FP16 path.
* `available()` is false without the ESIMD aspect and an s8 8x16x32 matrix combination (Alchemist has 8x8x32; non-Intel devices),
  `STRATA_I8_FORCE=0` turns it off.
* Selection (`prefill.cpp`): `STRATA_PREFILL_MMQ=1` (alias `STRATA_PF_I8=1`) runs the layers whose experts are the supported
  types through it, `=0` keeps the FP16 path, `=auto` uses the measured crossover (below). Default off until the owner decides after
  the quality check.

## Different bits

The activation quantization makes the result differ from the dequantize + oneMKL path by about 0.3-0.5% rms per product (the
q8_1 error), which is also what llama.cpp's MMQ path carries. `esimd_moe_parity` reports it against the FP16 path and against an
FP64 reference; KL and the task suites are in `phaseA-tests/notes-151-b70.md`.
