<h1 align="center">Strata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more), or an Apple Silicon Mac · Windows, Linux or Mac · free and open source<br>
<sub>Strata 0.2.0-beta · engine 0.1.45.3</sub></p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

Strata comes in three parts. Most people only need the first.

- **The app:** a desktop program. Install it, pick a model, chat. It has the engine inside.
- **The Terminal:** the `strata` command, for people who like a terminal and for scripts and AI agents.
- **The engine:** the program that talks to your graphics card. The app and the Terminal start it for you.

## Install

### The app (easiest)

Download the installer for your system from the [releases page](https://github.com/Niko1221/Strata/releases/latest):

| System | File | Notes |
| --- | --- | --- |
| **Windows 10 / 11** | the `.exe` installer | Installs for your user. It does not need administrator rights. |
| **Linux** | the `.AppImage` | Make it runnable (`chmod +x`), then start it. It needs FUSE 2 (`libfuse2`; on Ubuntu 24.04 and newer `libfuse2t64`). |
| **Mac** (Apple Silicon, macOS 15 or newer) | the `.dmg` | Open it and drag Strata onto Applications. |

The release page says which installer is for which graphics card (NVIDIA or AMD). Take the one that matches yours.
The engine is inside, so there is nothing else to install. Then:

1. Start Strata. The first screen reads your PC and recommends a model that fits.
2. Press the install button. The model is a big download (about 40-70 GB, depending on the size). If it stops, it
   continues where it stopped.
3. When the model has loaded, chat. Closing the window stops Strata and the engine.

If something does not open: Windows may say "Windows protected your PC" because the installer is not code-signed yet
(choose **More info**, then **Run anyway**). If a Mac says the app is "damaged", see [docs/MAC.md](docs/MAC.md).

### The Terminal (one line)

Linux and Mac:

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh | sh
```

Windows (PowerShell):

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

It checks what kind of PC this is, downloads the Terminal, checks the download's SHA-256, unpacks it into your home
folder (no administrator rights) and puts `strata` on your PATH. On Windows, open a new terminal window afterwards.
Then run:

```sh
strata launch
```

It asks one question at a time: which model, how much context, pictures, who may connect. Then it downloads the
engine for your graphics card (signed, and test-started before it is used) and the model, and starts serving. A
script or an AI agent uses flags instead of questions:

```sh
strata launch --model q2_0 --yes --no-tui
```

If you want to read the install script before it runs, download it first (`curl -fsSL <address> -o install.sh`),
read it, then run `sh install.sh`. More: [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md).

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

Do you want to install from the repository instead? See [Build from source](#build-from-source).

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. A Mac has no separate card: see the Mac paragraph below. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11, Linux, or a Mac with Apple Silicon (macOS 15 or newer), and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).
One engine drives the cards of one make: Strata never splits a model across NVIDIA and AMD cards.

**Apple Silicon Macs (M1 to M5):** the Mac app has the Metal engine inside. A Mac needs 16 GB of unified memory or
more to try it, and the memory decides which size fits ([docs/MAC.md](docs/MAC.md)). **There is no picture reader on
the Mac yet.**

Not tested by us, or experimental. Community members wrote and tested most of these on their own machines:

- **Windows with an AMD card:** the 0.2.0 installer and engine start, but we did not run a model with them on a real
  AMD card on Windows. [AMD cards](docs/AMD_HIP.md).
- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc** on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux. We have no Strix Halo test machine any more:
  [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

**These are older measurements.** They were made with older engines, not with the engine of this release (0.1.45.3).
The note under the tables says which engine measured which numbers.

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA table: Q2_0 measured with engine 0.1.36, the other rows with engine 0.1.26 (4K answers, 32K prompts). AMD
table: measured on 2026-10-02 with an engine older than 0.1.41 (the exact build was not recorded; Linux, 4K answers,
32K prompts). The full tables are in [DETAILS.md](docs/DETAILS.md#speed-measured). Engine 0.1.45.3 changes the speed
of several cards. The measured changes, and what they were measured on, are in the
[release notes](https://github.com/Niko1221/Strata/releases). A card with more VRAM is faster: an RTX 3090 (24 GB)
should write about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## Which model should I pick?

The app and `strata launch` recommend one for your RAM. The same model comes in several sizes, compressed more or
less. Smaller sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):**
  4- to 8-bit, closer to the full model (93-95% the same next token as Q8_0). Fast where every expert fits in RAM
  (~88 GB for UD-Q4_K_XL up to ~136 GB for Q8_0); with less RAM Strata reads part of them from the SSD while it
  answers, which is much slower (7-8.5 tokens/s for UD-Q4_K_XL on a 64 GB PC).
- **[Gyro-S and Gyro-M](docs/GYRO.md)** and **[NVFP4](docs/NVFP4.md)** are other compressions of the same model, for
  special cases. **[Qwen3.6-35B-A3B](docs/QWEN36.md)** is a smaller, different model that reads pictures and can
  answer several requests at once.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, open the app's
**Models** page, or run `strata launch --model NAME` in the Terminal. (From source: `SETUP.bat`, or
`./setup.sh --setup` on Linux.)

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the app:** **Chat**, a **Monitor** of the model and your GPU, CPU and RAM, a **Models** page and **Settings**.
  The chat can work in a folder on your PC. In its **Manual** mode it asks before it changes a file or runs a
  command. Type `/loop 10m <what to do>` to repeat a message every 10 minutes while the chat is open. The chat can
  search the web: it uses DuckDuckGo (or Bing) without an account, and you can add a search key under **Settings,
  Tools**. This is the one feature that sends something off your PC: the search words.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: use **`http://127.0.0.1:8080`** as the base URL, **without `/v1`**. Those apps add
    `/v1/messages` themselves. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`.
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
  - `strata connect` prints the exact settings for OpenCode, Codex CLI, Claude Code, oh-my-pi, pi, Hermes Agent and
    OpenClaw. `strata connect opencode --write` writes the file for you, after asking, and keeps a backup.
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** pictures are on by default for models that can read them. Attach or paste a picture in the chat, or
  send it from your app. The picture reader runs on the processor when your graphics card has less than 24 GB (a few
  seconds per picture; it takes nothing from the card's memory) and on the graphics card from 24 GB. The rule is the
  same for NVIDIA and AMD. You can turn pictures off per model (in the app, or `--vision off`), or choose the reader
  (`--vision gpu` or `--vision cpu`). Why this rule: on a Radeon AI PRO R9700, the card's reader took 57 ms for a
  260-token picture and the processor's took 3.1 s. On 16 GB cards (Tesla P100 and RTX A4000, Linux) the card's reader
  cost 3-5% of text speed, because it keeps about 1.4 GB that the model would use otherwise. **The Mac has no picture
  reader yet.**
- **Two models on two cards:** in the app, turn on **Run multiple models** in Models, and give each model its own
  card. Both stay loaded, and you pick one in the chat. The cards must be of the same make.
- **From your phone or another PC:** in the Terminal, `strata launch --access lan`. It makes an API key for you and
  will not start on a network without one. Always use a key. (From source: `START-HERE.bat --setup --host 0.0.0.0
  --api-key <secret>`.)
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)), or `--parallel N` in `strata launch`. On a 12 GB card this makes
  each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.
- **No browser opens by itself.** `strata launch` opens its web page only if you add `--open`.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## The Terminal

The Terminal is the `strata` command. It uses the same models, engine and saved settings as the app.

```sh
strata launch                                  # guided screens: model, context, pictures, access, then serve
strata launch --model q2_0 --yes --no-tui     # no questions: serve, print the address, return
strata launch --plan --model q2_0              # show what would happen; nothing changes
strata launch --status                         # what is being served now
strata launch --stop                           # stop it
strata connect claude-code                     # the settings for a coding app
strata monitor --url http://127.0.0.1:8080     # watch the model: speed, GPU, memory, requests
strata update --check                          # is there a new engine? (strata update installs it)
strata doctor                                  # check the installed engine files
```

`strata --help` lists everything. The step-by-step guide is [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md), and
the whole client is described in [docs/TERMINAL.md](docs/TERMINAL.md). Starting and serving never download anything.
Only the install, `strata launch` (when you choose a model) and `strata update` do.

## Build from source

You can also install from the repository. [Download Strata](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)
and unzip it (or `git clone` it). **Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in
the Strata folder.

**Mac (Apple Silicon, M1 to M5, macOS 15 or newer):** install Apple's Command Line Tools once
(`xcode-select --install` in Terminal; Xcode itself is not needed). Then open Terminal in the Strata folder and run
**`./setup.sh`**. It builds the Mac engine on your Mac (about 7 minutes), asks the same questions, downloads the model
and starts it. Next time, `./setup.sh` starts it right away. Which size fits your Mac and how fast it is:
[docs/MAC.md](docs/MAC.md).

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata page at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

**Is it fast on your PC?** Run `BENCH.bat` (`./bench.sh`): about 5-10 minutes, it measures chat, 4K/16K prompts and an agent-style session with your own settings and writes a report to share ([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

### Optional: more speed on one card when the whole model fits in RAM

Two opt-in settings (by RubenTorney, #1904 and #1905). They help when you run one graphics card and your RAM holds
every expert (Strata's arena mode, for example Q2_0 or IQ3_XXS on a 64 GB PC). Open your config,
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)), and add:

- **Faster decode:** `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread
  instead of pausing the answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long
  documents. Answers can differ slightly from the default (they stay the same from run to run).
- **Faster prompt reading on Windows:** `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block. Windows can lock only about
  half the RAM for the graphics card; this locks the experts the card actually needs first. Measured on the RTX 5070:
  +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata (`START-HERE.bat` or `./setup.sh`) after the change. To undo, remove the two entries.

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Start the app or `strata launch` again, or run `START-HERE.bat`
  (or `./setup.sh`) again. It continues where it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says there is no engine for this PC.** In the app, open **Settings, Engine** and press **Update engine**. In
  the Terminal, run `strata update`. `strata doctor` shows which engine files are installed.
- **It says port 8080 is already in use.** Strata is already running. Look for its window.
- **A Mac says the app is "damaged".** The file is fine. [docs/MAC.md](docs/MAC.md) has the one command.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues). In the app, attach the file from **Settings, Engine, Diagnostics,
View logs, Save logs**. In the Terminal, paste the output of `strata doctor`. From source, attach `strata-<model>.log`
from the Strata folder. Found a security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Contributing

Found a bug, or want to send a change or a benchmark? Read [CONTRIBUTING.md](CONTRIBUTING.md) first. It says what to
include so that others can check your work.

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
