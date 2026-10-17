# `strata launch`: set up, serve and watch, in the terminal

One command for the whole path: choose a model, tune it, start serving, and watch it. It is the same questions as the one-click setup (`START-HERE.bat` / `setup.sh`),
in the Terminal's look, with presets, an Advanced table, an access choice and a Connect view for your coding apps. Text previews of every screen are in
[docs/terminal-previews/](terminal-previews/) (`launch-*.txt`).

## Install (one line)

```sh
curl -fsSL <release-folder>/install.sh | sh          # Linux, macOS
irm <release-folder>/install.ps1 | iex                # Windows PowerShell
```

It finds out which computer this is, downloads the Terminal archive named in the release folder's `install-manifest.txt`, checks its SHA-256, unpacks it into your
home (`~/.local/share/strata/terminal`, `%LOCALAPPDATA%\Strata\terminal`; no administrator rights), checks the release's signed feed with the Terminal it just unpacked
(`strata update --check`; if that fails it removes what it unpacked), puts `strata` on your PATH and prints `Installed. Run: strata launch`. The engine (the part that talks
to your graphics card) comes with the first model: `strata launch` fetches the one for your card from the same release folder, signed, and test-starts it before using it.
`<release-folder>` is where the release is hosted (not decided yet: set `STRATA_FEED` to try a local one; `packaging/make_local_feed.py` builds a test-signed one).

## The screens (`strata launch`)

1. **Start**: saved presets (name, model, context, parallel requests, tuned or not, last used) or "Set up from the beginning" or "Use model files I already have".
2. **Model**: the catalog of the app and `setup.py` (one source), with download size, what it needs (graphics memory + RAM) and "fits / tight / does not fit this computer".
   A warning, never a block. The n-gram table stays on your SSD and is not counted. `o` = your own files (a folder or a .gguf; checked with the fingerprint table).
   *(Cards: only with two or more usable graphics cards.)*
3. **Folder**: where model files are stored, free space shown.
4. **Context** (8K to 512K, with the extra memory each costs) and the memory format (`m`: 8-bit or 4-bit).
5. **Pictures**: only for models that can read them; says what it adds (about 0.9 GB: the picture-reader file, downloaded with the model through
   setup's Control mode, the same path the app's "Read pictures" uses). On a card under 24 GB and on AMD the picture reader runs on the CPU (nothing taken
   from the graphics card); on a bigger NVIDIA card it runs on the GPU. It needs `strata-vision` beside the engine in the engine package; the screen
   warns in plain words when the installed engine has none, and the run says "Pictures could not be switched on" instead of silently answering text only.
6. **Parallel requests**: "fit to your card" (12 GB: 1, 20: 2, 24: 3, 32+: 4) or 1 to 4; a warning if it will not fit.
7. **Tuning**: quick (about 20 minutes) / thorough (up to 2 hours) / skip (`strata tune auto`; the "answers may change very slightly" tick is off).
8. **Advanced** (optional): every engine setting with default, tuned value and your value; edit, reset, add a setting by name (checked against the engine's own list; a warning, not a refusal);
   the exact command and the matching `strata launch` flags.
9. **Access**: this computer only (127.0.0.1), your network (LAN address, key), Cloudflare quick tunnel (needs `cloudflared`), Tailscale (needs it installed and connected); port; API key
   (made for you beyond this computer: the server will not start on a network without one).
10. **Save as preset** and launch. Then downloads (with progress), engine start, and the monitor.

Keys: arrows choose, Enter goes on, Esc goes back, `?` explains the screen, `q` leaves. Every screen says in plain words what it means and what to do next.

## After the start: the monitor

The Strata Monitor with the new **Overview** (key 1: model state bar, speed, GPU load, VRAM, temperature, power, PCIe, CPU, disk, context fill, experts in VRAM, system RAM, recent requests; read from the
server's `/metrics`, the same data as the browser Monitor), the other five views as before, and **7 Connect**: the address, the key (hidden until `k`), and the setup for each app
(`c` copy, `w` write it for you after a backup, `h` check it: one prompt and one tool call sent the way that app sends them). `q` asks: keep serving in the background, or stop
(stopping unloads the model and frees the graphics memory).

In the classic Windows console (Consolas) the monitor draws ASCII sparklines and square corners instead of block glyphs the font does not have (`STRATA_GLYPHS=unicode|ascii` forces either).

## Without the screens (flags)

```sh
strata launch --preset "IQ3_XXS 128K"                                 # a saved preset (opens the monitor)
strata launch --model iq3_xxs --context 128k --parallel 4 --vision off --tune quick \
              --access lan --port 8080 --save-preset "Office" --no-tui   # serve and print the address
strata launch --model-path /data/models/IQ3_XXS --no-tui               # files you already have
strata launch --preset Office --set prefill=65536 --set STRATA_PIN_BY_NEED=1 --no-tui
strata launch --plan --model q2_0                                      # what would happen; nothing changes
strata launch --list-presets | --delete-preset NAME | --status | --stop [--preset NAME] | --show-key
```

`strata launch --help` lists everything with examples. An unknown model or flag gets a plain message with the closest match. Presets are small files in `launch-presets/` beside the profiles
folder; the run uses the shared profile `launch-<name>` and the shared tuning presets (`strata presets`), so the app sees them too. Flags next to `--preset` change that start only
(add `--save-preset` to keep them).

## Connect your apps: `strata connect`

`strata connect` lists OpenCode, Codex CLI, Claude Code, oh-my-pi (omp), pi, Hermes Agent, OpenClaw and the DeepSeek Harness (built into the app). `strata connect opencode` prints the file and
environment for the running server (address, key, model name and context filled in); `--write` writes it after asking and keeps a backup of anything it replaces; `--check` sends one prompt and one tool call
the way that app does and says whether each worked. Each configuration was tested for real against a Strata server at 32K context (the app versions are on screen); the known limits are printed in plain
words (for example: Claude Code needs `CLAUDE_CODE_MAX_CONTEXT_TOKENS`; OpenCode needs `limit.input`; Hermes refuses windows under 64K unless set up as the config does; OpenClaw needs Node 24.16+).

## Boxes the installer cannot set up

`setup.py` sets up NVIDIA (CUDA) and AMD (HIP) cards. For an Intel Arc card its Linux path is a container build that the Terminal does not run: if the model's files and their prepared pack are
already on the disk, `strata launch` writes the run settings itself and says so; otherwise it says which file is missing. Nothing is downloaded or compiled in that case.

## Limits

* The model download uses `setup.py` in its control mode (no pip, no engine fetch, no compiler), as the desktop app does; it needs internet and room for the files.
* `strata update`'s test start of an engine runs `<engine> --version`; the SYCL (Intel) package's engine does not answer that flag (exit 2), so on an Intel box the engine package has to be copied into the
  components folder until that is fixed.
* The Windows console has no way for the test scripts to drive the screens; they were checked by pilot tests and by hand.
