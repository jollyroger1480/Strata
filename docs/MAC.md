# Strata on a Mac (Apple Silicon, Metal)

Strata runs on Apple Silicon Macs through its own engine on Metal. This is the same engine as on a PC. The NVIDIA
kernels are replaced by Metal ones (`metal/`), and the host code and the server are the PC's. `./setup.sh` sets it up the way
`START-HERE.bat` does on Windows. There is no ready-made Mac engine: setup builds it from source on your Mac
(about 1-15 minutes). The NVIDIA, AMD and Intel engines are unchanged.

**Status:** new. It is measured on two Macs: an M5 Max with 128 GB, where the model stays in memory, and an M4 Mac mini
with 32 GB, where the experts stream from the SSD. Other chips and memory sizes are untested, and reports help
([COMMUNITY_BENCHMARKS.md](COMMUNITY_BENCHMARKS.md)).

## What you need

- **An Apple Silicon Mac (M1 or newer).** Intel Macs cannot run it.
- **macOS 15 or newer.** The engine keeps its buffers resident through a residency set, which macOS 15 introduced.
- **The Command Line Tools:** `xcode-select --install`. That is the compiler. Xcode itself is not needed.
- **Enough unified memory for the pack.** The GPU maps the model's main file and keeps it in memory while the model runs. Metal lets one
  program keep about 3/4 of the memory resident (107.5 GB of 128 GB, measured on the M5 Max).

| Pack | The file the GPU maps | Needs about (with the prompt buffers, KV cache, draft layer) | Recommended from |
|---|---|---|---|
| Q2_0 | 37.6 GB | 46 GB | 64 GB |
| IQ3_XXS | 47.0 GB | 55 GB | 96 GB |
| IQ3_S (the best quality) | 54.8 GB | 63 GB | 110 GB (the 128 GB Macs) |

- **The n-gram table:** each pack also has a 28.8 GB n-gram table (the model's second file). It is read from the SSD a few rows at a time and does not count against the memory above.
- **Disk:** the download is 66-84 GB, plus about 7 GB for the pack and the draft layer.
- **A Mac whose memory cannot hold the pack (32-48 GB):** the engine streams the experts from the SSD instead. It decides this by itself when the experts are more than 60% of Metal's share.
  - It keeps the most-used experts in an arena: Metal's share, or the memory macOS reports as available less 7 GB for macOS, whichever is smaller.
  - It reads the rest as each token needs them.
  - Setup configures such a Mac without the MTP draft layer: one token per step is faster there.
  - Measured at 32 GB (below). Smaller Macs are untested.
- **Setup recommends, and never forces.** It recommends the pack by memory; a bigger pack chosen with `--model` is kept, with a warning about what it risks.

## Install

```
git clone https://github.com/Niko1221/Strata && cd Strata
./setup.sh                 # asks the size; --yes takes the recommended answers
```

What the first run does (each step is skipped when it is already done):

1. **Python.** It needs Python 3.10 or newer, but macOS ships 3.9. If that is all your Mac has, `setup.sh` downloads a standalone Python 3.12 (25 MB, pinned by SHA-256) into `.python`. No sudo, no Homebrew.
2. **Checks your Mac:** the chip, the memory, the macOS version, the Command Line Tools and the free disk.
3. **Python packages** into `.venv`. CMake and Ninja come from pip.
4. **Builds the engine** (`build-metal/strata`). It uses Apple's metal-cpp headers (pinned by SHA-256) and llama.cpp's ggml at the commit the PC build pins.
5. **Downloads the model.** The download resumes if interrupted. `--gguf-dir` uses files you already have.
6. **Prepares the model:** the pack (seconds), and the MTP draft layer (about 5 GB of the original checkpoint).
7. **Writes `strata-<model>.json` and `run-<model>.sh`, then starts the server.**

The config uses the Mac defaults: `--spec 2` and the Q8_0 hyper-connection read (below). After that, `./setup.sh` or `./run-q2_0.sh` starts the model directly. The server is the same as on a PC:
- the OpenAI-compatible API on `http://127.0.0.1:8080/v1`;
- the Anthropic API on `/v1/messages`;
- the chat page on `http://127.0.0.1:8080/`.

Options: `--model Q2_0|IQ3_XXS|IQ3_S`, `--context N`, `--port`, `--host` / `--api-key`, `--draft-vocab`, `--no-browser`,
`--data-dir`, `--models-dir`, `--gguf-dir`, `--setup` (another model or new settings), `--no-start`, `--check` (only
check the Mac), `--build` (build the engine again).

## Measured (M5 Max, 128 GB, macOS 27.2)

**Decode**, in tokens per second over 200 generated tokens, with greedy answers. The table gives two prompts: a short question
("prompt1") and a story.

| | Q2_0 | IQ3_XXS | IQ3_S |
|---|---|---|---|
| Setup's defaults (`--spec 2` with MTP, the suffix drafter on) | **119.5 / 108.2** | **108.9 / 97.0** | - |
| `--spec 2` with MTP, suffix drafter off | 114.0 / 107.9 | 103.5 / 96.9 | 98.2 / 94.1 |
| No MTP (one token per step) | 87.4 / 87.3 | 78.7 / 78.6 | 74.5 / 74.3 |

The output with MTP is the same as without it, token for token, in every run. MTP only proposes tokens, and the
full model checks every one of them.

**Prompt reading** (time to the first token), on the needle-test prompts below:

| | 4K tokens | 32K tokens |
|---|---|---|
| Q2_0 | ~1,430 tok/s | ~1,460 tok/s |
| IQ3_XXS | ~1,260 tok/s | ~1,240 tok/s |
| IQ3_S | ~1,320 tok/s | ~1,140-1,190 tok/s |

The first prompt after the Mac starts is slower, while macOS reads the model file (IQ3_S: 413 tok/s on the first 4K
prompt, cold).

**For comparison**, llama.cpp on the same Mac with Q2_0: 56.6 tok/s decode on the story, and 887 / 809 tok/s on 4K / 32K prompts.

### M4 Mac mini, 32 GB, experts streamed from the SSD

The arena was 11.5-14.7 GB. Tokens per second, one token per step (no MTP), medians of 5 interleaved runs:

| | Q2_0 | IQ3_XXS | llama.cpp, CPU only (Q2_0 / IQ3_XXS) |
|---|---|---|---|
| Decode, short question (200 tokens) | **14.1** | **10.0** | 6.3 / 4.1 |
| Decode, story (200 tokens) | **13.5** | **9.2** | 6.3 / 4.1 |
| Decode after a 6K-token document | **13.1** | **7.7** | 5.1 / 2.8 |
| Prompt, 4K tokens | **93.6** | **65.8** | 21.2 / 8.7 |

- The greedy answers equal the M5 Max's (model in memory) on 10 of 10 test prompts, for both packs.
- No memory was swapped out during the runs; the least memory left to macOS was 7.1 GB.
- MTP was slower here (Q2_0 story 12.1-12.3 with it against 12.3-12.8 without). The draft layer's 1.2 GB do more as expert cache.

**Why `--spec 2`:** measured with MTP on the same two prompts.

| Window | Q2_0 (prompt1 / story) | IQ3_XXS (prompt1 / story) |
|---|---|---|
| `--spec 2` | 111 / 106 | 102 / 95 |
| `--spec 3` | 116 / 92 | 108 / 95 |
| `--spec 4` | 111 / 81 | 108 / 83 |
| `--spec 5-6` | slower | slower |

`--spec 2` is the best default on mixed text. An explicit `--spec N` in the config's `args` is kept.

### Quality

- **The Q8_0 hyper-connection read is the default on Mac.** It keeps a Q8_0 copy of the hyper-connection matrices instead of BF16. `STRATA_HC_Q8=0` in the config's `env` gives the exact BF16 read.
- **How the KL was measured:** teacher-forced over 28 prompts (code, reasoning, chat, six languages, documents), about 6,000 positions per set. The reference is BF16.
  - On the model's own text: mean KL 0.0007, top-1 agreement 98.9%.
  - On another model's text: mean KL 0.0007, top-1 agreement 99.1%.
  - Perplexity of the text is unchanged (×0.9995 and ×1.0006).
  - For scale, IQ3_XXS against Q2_0 is a KL of 0.094.
  - The Mac engine is deterministic: two runs give identical numbers.
- **The fast prompt path** (the prompt kernels) against reading the prompt through decode windows: mean KL 0.0004, top-1 99.4-99.6%.
- **Needles:** a code word hidden in 8K, 16K and 32K tokens of text, at 10/50/90% depth. Found 9 of 9 with Q2_0 and with IQ3_XXS, both with Q8_0 and with BF16. With the fast prompt path (4K-32K) the answers are the same.
- **Same answers as on a PC:**
  - against the CUDA engine on a Tesla P100 (IQ3_XXS and IQ3_S), the first 200 tokens of a reply are identical on the short question;
  - on the story, the first IQ3_S tokens differ at token 1, a near-tie (margin 0.12). Following the CUDA engine's tokens, 4 of 200 picks differ, all at margins 0.06-0.21;
  - against the HIP engine on a Radeon AI PRO R9700 (Q2_0), 36 of 39 first tokens are identical.
  - The differences are near-ties between the two best tokens, from a different floating-point order.

## Limits (this version)

- **Images (vision):** not on Mac yet. Setup does not offer them.
- **Context:** measured up to 32K (needles, KL). Setup recommends 32K below 96 GB and 64K from 96 GB. Up to the model's 262K can be set, but it is not measured.
- **One request at a time.** `--parallel` (batch slots) is not ported. Several clients queue.
- **Not ported:** the control vector (`--control-vector-scaled`) and the PC tuning (`--calibrate`).
- **Streaming:** measured at 32 GB only. The model's file must be on the internal SSD; an external disk is untested and likely much slower.
- **Families:** the original Qwen3.8-Flash-Next is measured. The Swift and Coder releases and the Unsloth files are not set up on Mac yet.
- **Engine only from source.** There is no ready-made Mac engine in the release zips.

## Switches (for testing; all off unless set, the defaults are above)

| Variable | Effect |
|---|---|
| `STRATA_HC_Q8=0` | The BF16 hyper-connection read (exact) instead of Q8_0 |
| `STRATA_METAL_PROMPT=window` | Read the prompt through decode windows (the slow, exact fallback; ~190 tok/s) |
| `STRATA_METAL_PROMPT_CHUNK=N` | Prompt chunk size (default 2048) |
| `STRATA_METAL_MTP_Q8=0` | The draft layer's matrices in BF16 instead of Q8_0 |
| `STRATA_METAL_MTP_CHAIN=0` | One GPU submission per draft step instead of one per round |
| `STRATA_METAL_FUSE=0`, `STRATA_METAL_LAZY_FOLD=0` | Turn off two decode fusions (same tokens, slower) |
| `STRATA_METAL_PARTIAL=1`, `STRATA_METAL_ROUTE=1` | Two decode variants that measured no faster here (same tokens) |
| `STRATA_METAL_STREAM=1` / `=0` | Stream the experts from the SSD even when the pack fits / never (the default decides by size) |
| `STRATA_METAL_KEEP_HOST_WEIGHTS=1` | Streaming: keep the host code's own copies of the weights (2-3 GB) instead of giving them back to macOS |
| `STRATA_METAL_TRACE=1`, `STRATA_METAL_PROFILE=1`, `STRATA_METAL_KPROF=T` | Per-window traces, a time breakdown, per-kernel GPU times |

## How it works

The code is in `metal/` (the engine's side in `metal/src/engine`, the kernels in `metal/kernels`). In short:
- **Host code:** the PC host code compiles unchanged for macOS. `metal/include/metal_compat` maps the CUDA runtime calls it makes onto a small Metal runtime (`metal/src/gpu`).
- **The decode and prompt work** run in Metal kernels (`metal/kernels`, `metal/src/engine`):
  - one command buffer per verify window;
  - the experts are read in place from the mapped model file;
  - the hyper-connections, the GDN state and the QSA attention stay on the GPU;
  - a whole MTP draft round runs in one command buffer.
- **The PC builds are byte-identical** with this code present. Every shared file's change is behind `STRATA_GPU_METAL` or `__APPLE__`, and the CUDA and HIP objects are compared after each change.
