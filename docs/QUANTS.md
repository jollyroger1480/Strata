# Unsloth's K-quants and Q8_0: formats, kernels and quality

This page covers Unsloth's files of the model ([unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
revision `38bb39e`; the repository's head `766911a` has the same files, byte for byte): what each file stores in
each layer, which of Strata's kernels run those formats on each kind of hardware, and how close each file comes to
the Q8_0 reference. Setup and the per-file details: [UD-Q4_K_XL](UNSLOTH_Q4.md), [UD-Q6_K_XL](UNSLOTH_Q6.md),
[MODELS.md](MODELS.md).

## What each file stores

Read from the GGUF headers alone with `tools/quant_formats.py --hf unsloth/Qwen3.8-Flash-Next-GGUF 38bb39ee... <dir>`
(HTTP range requests, nothing else is downloaded; on local shards: `tools/quant_formats.py <shard glob>`).

The routed experts (gate, up, down; 512 per layer, 10 used per token) are where the files differ:

| File | Download | Routed gate / up | Routed down | Routed experts | Per token (10 experts x 48 layers) |
|---|---:|---|---|---:|---:|
| UD-IQ4_XS | 93.7 GB | IQ3_S (IQ4_XS in layer 2) | IQ4_NL (Q8_0 in layers 2, 4, 30, 46, 47) | 55.4 GiB | 1.16 GB |
| UD-Q4_K_XL | 111.3 GB | Q4_K (Q5_K in layer 2) | Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) | 71.7 GiB | 1.50 GB |
| UD-Q5_K_XL | 158.3 GB | Q5_K (Q6_K in layer 2) | Q8_0 | 91.6 GiB | 1.92 GB |
| UD-Q6_K_XL | 169.2 GB | Q6_K (Q8_0 in layer 2) | Q8_0 | 101.7 GiB | 2.13 GB |
| Q8_0 | 188.2 GB | Q8_0 | Q8_0 | 119.5 GiB | 2.51 GB |

Everything else is the same in all five: the attention, SSM, hyper-connection and shared-expert projections, the
PLE key/value and the token embedding are Q8_0 (the small vectors, norms and the router F32; the QSA indexer BF16),
the output head is Q8_0 (Q6_K in UD-IQ4_XS). The PLE table (`per_layer_token_embd`) is IQ4_NL (26.8 GiB) in UD-IQ4_XS
and UD-Q4_K_XL and Q8_0 (50.7 GiB) in UD-Q5_K_XL, UD-Q6_K_XL and Q8_0. The dense side the GPU reads once per verify
window is about 4.8 GB in every file.

UD-Q5_K_XL, UD-Q6_K_XL and Q8_0 share their shards 2 (the output head) and 3 (the PLE table) byte for byte (the same
SHA-256), so a PC that has one of them can link those two shards instead of downloading them again.

## Kernels per format and backend

| Format (role) | CUDA decode (grouped experts) | CUDA prompt | CPU pool | AMD (HIP) decode | AMD prompt | Intel (SYCL) decode | Intel prompt |
|---|---|---|---|---|---|---|---|
| Q4_K (gate/up) | yes | FP16 path; MMQ with `STRATA_MMQ_KQ=1` | ggml AVX2 dot; multi-token AVX2 (`STRATA_KQ256=1`) | yes (CUDA layout; `STRATA_EXPERT_V2K=1` bitwise) | FP16 path; MMQ with `STRATA_MMQ_KQ=1`; gfx11 fused (`STRATA_PF_FUSED_KQ=1`) | kernels built, not run with these files | FP16 path |
| Q5_K (gate/up) | yes | as Q4_K | ggml AVX2 dot | as Q4_K | as Q4_K | kernels built, not run with these files | FP16 path |
| Q6_K (gate/up) | yes (since this change; was `-DSTRATA_Q6K_EXPERTS=ON`) | FP16 path; MMQ with `STRATA_MMQ_KQ=1` | ggml AVX2 dot | yes (CUDA layout) | FP16 path; MMQ with `STRATA_MMQ_KQ=1` | kernels built, not run with these files | FP16 path |
| Q5_1 (down) | yes | as Q4_K | ggml AVX2 dot; multi-token AVX2 (`STRATA_KQ256=1`) | yes (CUDA layout; V2K) | as Q4_K | kernels built, not run with these files | FP16 path |
| Q8_0 (gate/up, down) | yes | MMQ | ggml AVX2 dot; multi-token AVX2 down (`STRATA_KQ256=1`) | yes (CUDA layout; V2K down) | MMQ | kernels built, not run with these files | FP16 path |

- "FP16 path": the prompt's experts dequantized to FP16 and multiplied by the BLAS library. It works for every format
  above but reads each expert as ~10 MB of FP16.
- **Formats cost nothing unless used.** The engine loads every kernel of its own binary when it starts (so that no
  kernel needs VRAM later, mid-prompt), which is why kernels for formats a model does not have used to take expert
  slots from every model. The Q6_K expert kernels are now part of every build: one more instance per grouped kernel,
  +19 KB of sm_60 code, and the expert-cache slot counts measured unchanged (IQ3_XXS: 6,028 slots on a P100, 15,878
  and 1,980 on an R9700). The K-quant MMQ prompt kernels (Q4_K, Q5_K, Q5_1, Q6_K: ~3.8 MB of sm_60 code, 3 IQ3_XXS
  slots on the P100 when they were in the engine) are on CUDA a library of their own next to the engine
  (`libstrata_kq.so` / `strata_kq.dll`), opened only for a model with these experts and `STRATA_MMQ_KQ=1`; without the
  file those experts take the FP16 prompt path. On AMD a code object loads on first use, so they stay in the engine.
- `STRATA_MMQ_KQ=1` is opt-in because MMQ rounds the prompt's products differently from the FP16 path (the output
  differs at the rounding level); its speed and KL are measured before it becomes a default.

## Quality: the yardstick

`tools/quant_quality.py` scores a quant against Q8_0 on one fixed text set (`data/quant_quality/v1`: code, docs,
English prose, German/French/Japanese/Chinese, an agent transcript; 5 x 1,000 tokens in `set.ids.json`). Every token
is read through the decode windows (teacher-forced) with the experts on the CPU (one VRAM slot), no expert
adaptation, no PCIe share, no route tail skip, FP16 K/V; per position the engine writes the log-probability of the
text's next token and its 64 most likely tokens (`STRATA_LOGPOS`, `STRATA_LOGPOS_TOPK=64`). Top-1 agreement: the
same most likely token as Q8_0. KL: KL(Q8_0 || quant) over Q8_0's 64 most likely tokens.

```sh
python tools/quant_quality.py run --config strata-<quant>.json --out <quant>.tsv
python tools/quant_quality.py compare q8_0.tsv <quant>.tsv --segments
```

Measured on the AMD box (EPYC 7452, R9700, HIP, engine at 0601072d; 4,995 positions; Q8_0 against itself in a second
run: byte-identical rows):

| Quant | Same top token as Q8_0 | Mean KL | KL p99 | Perplexity (Q8_0: 4.052) |
|---|---:|---:|---:|---:|
| UD-Q5_K_XL | 94.55% | 0.038 | 0.636 | 4.024 |
| UD-Q4_K_XL | 93.09% | 0.056 | 0.829 | 4.068 |
| IQ3_S | 87.69% | 0.144 | 1.975 | 4.097 |
| IQ3_XXS | 84.46% | 0.202 | 2.433 | 4.119 |
| Q2_0 | 80.60% | 0.315 | 3.643 | 4.475 |

UD-Q6_K_XL and UD-IQ4_XS follow.
