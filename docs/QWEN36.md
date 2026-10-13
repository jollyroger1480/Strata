# Qwen3.6-35B-A3B: pictures and several requests at once

Strata runs [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (35B mixture of experts, 3B active, 256 experts, top-8) from
[Unsloth's GGUFs with the MTP layer](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) (UD-Q4_K_XL, UD-Q6_K_XL, Q8_0) on its own
engine path (`src/qwen35`; `strata` switches to it from the file's architecture). Pick it in setup (`--family qwen36`). Besides text it
reads **pictures** and serves **several requests at once**, the same two options the Flash-Next models have.

Measured on an AMD Radeon AI PRO R9700 32 GB (HIP, Linux, UD-Q4_K_XL, every expert in VRAM) and on an RTX 3060 12 GB (CUDA, 20 threads,
UD-Q4_K_XL in the hybrid mode: 22-37 of 256 experts per layer in VRAM, the rest on the CPU), llama.cpp 972b5fc on the same files.

## Pictures

The vision encoder is Unsloth's `mmproj-F16.gguf` (0.9 GB, pinned revision, SHA-256 checked by setup): a 27-layer ViT with a 2x2 merger,
the same projector family as the Flash-Next models' (`qwen3vl_merger`; only its output width differs: 2048 here, 2560 there). Strata runs
it with the same helper (`strata-vision`, llama.cpp's `mtmd`), so the encoder choices are the Flash-Next ones
([Images](DETAILS.md#images-vision)): **setup asks "Do you want images?"**, the default answer is yes on an NVIDIA card of 16 GB or more
(GPU encoder, ~1.2 GB of VRAM at up to 1,024 image tokens) and on AMD under Linux (CPU encoder, no VRAM); `--vision yes|no|gpu|cpu`
and `--vision-tokens N` work as there. A card under 16 GB is recommended the CPU encoder (the GPU one takes room from the experts);
`--vision gpu` is kept as asked, with a note.

| Encoder | Time per picture (640x400 = 260 tokens / 800x450 = 350 tokens) | VRAM |
| --- | --- | --- |
| GPU, RTX 3060 | 0.10 s / 0.15 s | 1,179 MiB at up to 1,024 tokens (1,337 MiB at 2,048), CUDA context included |
| CPU, 32 threads (R9700 box) | 0.9 s / 1.1 s | none |
| CPU, 10 threads (3060 box) | 1.6 s / 2.1 s | none |

A picture becomes up to 1,024 tokens of the context (the CPU default is 300). The engine puts the encoder's rows where the prompt has
`<|image_pad|>` tokens and gives every row its 2-D position, interleaved M-RoPE exactly as llama.cpp does for this model (rotary sections
11 / 11 / 10, a picture's rows at (t, h, w) = (p, p + row, p + column), the text after it continuing at p + max(columns, rows)).

**Checked against llama.cpp** (its `mtmd` and its own graph, the encoder on the CPU on both sides, the same prompt ids, the answer
llama.cpp gives fed to both so the logits are compared step by step, 24 steps; the "floor" is llama.cpp's CPU backend against its own
HIP run, the noise two correct engines have on this model):

| Picture and question | Prompt tokens | First token | Mean KL, llama.cpp HIP vs Strata | Floor (llama.cpp CPU vs HIP) |
| --- | ---: | --- | ---: | ---: |
| 640x400, text in the image | 288 | same | 0.029 | 0.056 |
| 640x400, list the shapes | 294 | same | 0.035 | 0.039 |
| 800x450, text in the image | 378 | same | 0.021 | 0.031 |
| 800x450, describe the shapes | 381 | same | 0.0005 | 0.0019 |
| 1024x1024, the title (1,024 image tokens, read in 3 chunks) | 1,042 | same | 0.0022 | 0.0026 |
| two pictures in one request, the text in each | 655 | differs (a near tie: the floor's KL at this step is 0.16 too) | 0.014 | 0.022 |

The answers are the same text as llama.cpp's: `STRATA 42` (with llama.cpp's own spelling ` STRATA42`), `CODE 4217`, `Red circle / Blue square
/ Green triangle`, `A red square is positioned on the left ... A blue circle is positioned on the right ...`. With the picture options on,
a request without pictures gives the same tokens as without them (10 prompts x 128 greedy tokens, identical).

Pictures work with several requests at once, with the MTP drafts and with the conversation cache: the same conversation sent again (a chat
app resends its pictures) continues from the held state when its pictures are the same (they are matched by the server's key for each
picture); a different picture starts that conversation again.

## Several requests at once

`"parallel": N` in the model's config (`strata-qwen36-<size>.json`, or `--parallel N` at setup) makes the engine keep N conversations open
(`--batch N`, up to 8) and decode them together: every window carries one token of each conversation, so the dense weights, the head and
every routed expert two conversations share are read once for all of them. The server needs no change: it speaks the same engine protocol
as for Flash-Next ([BATCHING.md](BATCHING.md)). A request that arrives while another is read waits only for that read: the prompt of a
new request is read **between** the decode windows of the others, and a conversation continues in the slot that holds its start (or the
engine copies the longer held start from another slot). A 3,008-token prompt read in 0.95 s (3,200 tok/s, 6 chunks) beside three decoding
slots: each of them paused at most 160 ms (one chunk) instead of the whole read, and produced 20 tokens meanwhile (one window per chunk,
the default); `STRATA_Q36_READ_WINDOWS=4` gives them 65 tokens during a 1.14 s read (more of the time, a slower read).
A request that runs alone (the solo path, with its MTP drafts) and is moved into a batch slot when another arrives, and back when it is
alone again, gives the text it gives running alone (the server's hand-over reuses the held state).

Aggregate decode speed, N requests at once, same prompts, greedy, 128 new tokens each, wall time from the first request to the last
answer, **R9700 32 GB, UD-Q4_K_XL, llama.cpp `llama-server -np N`** (5 interleaved pairs, median; Strata `--batch 8 --mtp 0`):

| Requests at once | Strata | llama.cpp | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.9 tok/s | 73.8 | 1.14 | 5/5 |
| 2 | 173.5 | 107.4 | 1.61 | 5/5 |
| 4 | 252.0 | 165.0 | 1.53 | 5/5 |
| 8 | 325.4 | 208.1 | 1.56 | 5/5 |

Every slot's tokens equal the tokens of the same request alone (the base engine's), at 2, 4 and 8 slots, with a 2.5K-token prompt read beside
decoding slots, with slots reused for later requests, with pictures and with sampling (same seed). Through the server: four concurrent chat
requests give the texts of the same four one after the other.

**MTP drafts.** A request that has the engine to itself keeps its MTP drafts (the server sends it alone while nothing else runs). With
several requests the batch window has no drafts unless `--batch-mtp` is in the config's `args` (opt-in): every slot's window then also
carries the proposal its MTP block made (one per slot, up to the rows to spare in the 8-row window) and a confirmed proposal moves two
tokens in the window. Same prompts, 5 interleaved pairs, R9700:

| Requests at once | `--mtp 0` | `--mtp 2 --batch-mtp` | Ratio | Pairs faster |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 83.7 tok/s | 134.0 | 1.60 | 5/5 |
| 2 | 173.1 | 207.0 | 1.20 | 5/5 |
| 4 | 251.0 | 275.0 | 1.10 | 5/5 |
| 8 | 324.0 | 320.5 | 0.99 | 0/5 (no row to spare: no proposals) |

The text is the same with and without it (every slot equal to its solo tokens).

**What a slot costs.** Each slot keeps its own cache in VRAM: 1.5 GB at a context of 32K (fp32 keys and values of the 10 attention layers
and the MTP block, 45 KB per token) and 0.07 GB of recurrent state. Setup writes the context for one conversation, so N slots take N times
that from the room the experts would have: on a card that does not hold every expert (12 and 16 GB, the hybrid mode) that is room the
expert slots lose, and a batch reads more distinct experts from RAM. Setup therefore **recommends `--parallel` only where every expert still
fits** (about 3 on a 32 GB card at 32K) and says why it leaves it at one otherwise; any number you give is kept. On the RTX 3060 12 GB with
`--batch 4` the engine kept 22 of 256 experts per layer (the sessions, the picture encoder and the MTP block took the rest) and four
requests at once ran 45 tok/s together against 42 one after the other: not worth it there. The engine says what the sessions take before it
allocates them and refuses (with the numbers) a count that cannot fit.

In the hybrid mode the text of a request can differ a little between running alone and running beside others: the experts that sit on the
GPU or on the CPU change with the routing of all the requests, and the two compute with different rounding. With the slot tables fixed (`STRATA_Q36_FIXED_SLOTS=1`, a test setting) every slot equals its
solo run there too, with and without `--batch-mtp`.

## Not done / limits

- Eight slots at most (the window holds 8 rows); `--batch-mtp` shares those rows between slots and proposals.
- A sampling request in a batch decodes one token per window without proposals. `logit_bias` is not available on this engine path: the
  server refuses a request that has it (as it does with `--batch` for Flash-Next).
- A slot that stopped being alone goes back to the solo path only through the server's own hand-over; the MTP block's cache has holes for
  the tokens a slot decoded in batch windows (the drafts of a later solo stretch are a little worse; the text is not affected).
- Intel Arc (SYCL) does not run this family.

## Testing

The gates are in `tools/qwen35/` and talk to the engine through its serve protocol (no server needed); run them under the box's GPU lock:

- `q36_solo_gate.py` + `q36_cmp_gate.py`: the 10 parity prompts x 128 greedy tokens through `GEN`, compared with another build's.
- `q36_batch_gate.py --slots N --waves --long 2500`: every slot's tokens equal its solo tokens, slots reused, a long prompt read beside
  decoding slots. `q36_proto_edge.py`: `max_new` 1, `BSTOP`, a busy slot, `STOP` during a prompt read, held-state reuse, sampled requests,
  the context's end. `q36_interleave_probe.py`: what the decoding slots see while a long prompt is read.
- `q36_vision_cmp.py` (with `q36_mtmd_dump.cpp` built against llama.cpp): the picture comparison above; `q36_vision_proto.py`: pictures through
  `GENI` / `BGENI` (solo, in a batch, the conversation cache); `q36_vision_serve.py` and `q36_serve_par.py` / `q36_serve_handoff.py`:
  the same through the server.
- `q36_par_bench.py`: the aggregate-throughput A/B against `llama-server -np N`.
