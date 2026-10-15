# Strata packaging: targets, recipes and locks

Revision: Phase A. This directory holds the per-target build recipes and the machine-readable
manifests. **It describes what can be built where, and it is explicit about what has *not* been
built or tested.** Nothing here is a support claim.

## The target matrix

A backend name alone is not a target. Each row is an OS × CPU ABI × backend tuple, and a row is
`qualified` only after the acceptance checklist has actually run on it.

| Target tuple | App artifact | Engine package | Status on this Mac |
| --- | --- | --- | --- |
| `macos-arm64-metal` | `.app` + `.dmg` (Tauri) | source-built `strata` | **built.** App self-test 8/8 from inside the mounted DMG; engine built from source and 17/17 kernel-parity suites pass on the real M4 GPU. Real-model inference **not run** (no pack fits 16 GiB / 21 GiB free). |
| Terminal `darwin-arm64` | self-contained `.tar.gz` (42 MB) | any | **built and verified.** Runs from a moved directory in a clean environment with no usable system Python; serving, `status`, `timeskip` and `tune` all exercised on the bundled interpreter. |
| `windows-x86_64-cuda` | NSIS `.exe` (Tauri) | `setup.py` / `tools/windows` | **not built here.** Config present; needs a Windows host with an NVIDIA GPU. |
| `windows-x86_64-hip` | NSIS `.exe` (Tauri) | `tools/hip/build_windows.bat` | **not built here.** Needs Windows + AMD. |
| `linux-x86_64-cuda` | AppImage (Tauri) | `setup.py` | **not built here.** Needs a Linux host with an NVIDIA GPU. |
| `linux-x86_64-hip` | AppImage (Tauri) | `cmake -DSTRATA_ENABLE_HIP=ON` | **not built here.** Needs Linux + ROCm. |
| `linux-x86_64-sycl` | AppImage (Tauri) | `sycl/setup_intel.py` | **not built here.** Needs Linux + Intel Arc. Documented upstream as experimental. |
| `windows-x86_64-sycl` | — | — | **gated.** No native Windows SYCL path exists upstream; it needs a separate feasibility gate, not a recipe. |
| `macos-x86_64-*` | — | — | **unsupported.** The macOS engine is Apple-Silicon only (`metal/CMakeLists.txt` requires Apple Silicon for the Metal runtime). |

**Cross-compilation is not attempted for the desktop app.** Tauri bundles are per-OS and the
Windows installer must be produced on Windows. What *is* portable, and therefore tested here, is
everything below the carrier: the Python contracts, the owner transport, the checks, and the engine
build recipes.

## The private Node runtime

The desktop bundle ships a pinned standalone Node beside the App Host, because the packaging
requirement is that an end user supplies no runtime. It is **not committed**: it is a ~117 MB
platform binary, it cannot be reviewed in a diff, and it is an input rather than source. Fetch and
verify it instead:

```sh
apps/desktop/scripts/fetch-node.sh                 # host platform
apps/desktop/scripts/fetch-node.sh --target linux-x64
apps/desktop/scripts/fetch-node.sh --check         # verify what is installed
```

The script refuses an archive whose SHA-256 does not match the digest recorded for that target. The
digests are taken from nodejs.org's own `SHASUMS256.txt` for the pinned version, so they are not
derived from the same response as the payload. **To bump the Node version: change
`STRATA_NODE_VERSION`, add the new digests, and re-run `--check`** — the script will refuse an
unknown target's archive rather than installing it unverified.

## Desktop app

```sh
cd apps/desktop
./scripts/fetch-node.sh                                                      # the sidecar (once)
npm install --prefix ../host && npm --prefix ../host run build                # the UI
cargo tauri build --bundles app                                              # -> Strata.app

# macOS DMG (headless: Tauri's default step needs a Finder session)
./scripts/make-dmg.sh

# Windows (on Windows): produces an NSIS per-user installer
cargo tauri build --bundles nsis

# Linux (on Linux): AppImage plus a deb
cargo tauri build --bundles appimage deb
```

Windows installer policy is set in `tauri.conf.json`: `installMode: currentUser`, so the installer
is **per-user and needs no elevation**. Uninstall retention (checklist P10) is a property of *where
the app keeps its data*, not of an installer flag: models, profiles and sessions live outside the
install directory (`~/.strata` and the platform state directories), so removing the application
cannot remove them. The packaging config was validated against this Tauri version's schema — an
earlier draft used `perMachine`/`oneClick`/`deleteAppDataOnUninstall`, which `tauri-utils` 2.10 does
not declare and which are rejected outright rather than ignored.

**The `dmg`/`app` targets were rebuilt and the packaged self-test re-run (8/8) after the Windows and
Linux targets were added**, so the added targets do not affect the macOS artifact.

## Strata Control and the engine in the installed app

The installer carries two things that start a model, and neither fetches anything when it runs.

**Control** (`<install>/control`, built by `apps/desktop/scripts/stage-control.py`): the same bundle as the Terminal
archive, the pinned CPython 3.12.15, the hash-locked wheels, `strata_control`, `strata_cli` and `serve`, plus a
`BUNDLE.json` with the interpreter and lock digests. The host finds it beside itself (or through `STRATA_CONTROL_DIR`),
runs it on its own interpreter (`PYTHONNOUSERSITE=1`, no `PYTHONHOME`/`PYTHONPATH`) and starts a model with
`strata profile create|use-config` + `strata runtime start`, so the engine runs under the owner runtime (loopback TCP and a
256-bit token on Windows, a private socket elsewhere; a Job Object / process group; leases) and a second client (the
Terminal) attaches to the same instance.

**The engine** is not in the installer. It is a component in the per-user component folder
(`%LOCALAPPDATA%\Strata\components`, `$XDG_DATA_HOME/strata/components`, or `STRATA_COMPONENTS_DIR`):

    components/engine/<version>-<platform>-<backend>/
        engine-component.json     schema_version, id "strata-engine", version, platform, backend, entrypoint,
                                  cuda_archs (optional), min_runtime (optional), files [{path, sha256, bytes}]
        strata[.exe]  *.dll | *.so   the binary and the GPU runtime libraries beside it

A run config names it as `"exe": "component:strata-engine"`; at start `strata_control/engine_component.py` picks the package
for this PC (platform, then the backend its GPU needs: nvidia-smi / rocminfo / sycl-ls, or `STRATA_ENGINE_BACKEND`; CUDA
architectures and `min_runtime` are checked; the newest version wins), checks every file's size and the entrypoint's
SHA-256 (`strata components engine --full` hashes everything) and puts the package folder first on the library path.
Nothing matching is `EngineMissing`: exit 5, "No engine for this PC was found in ... The app never downloads an engine
while it starts or serves", and no fetch. `packaging/package_engine.py` makes a local package from a build folder.

**No acquisition at run time.** `STRATA_NO_ACQUIRE=1` (set by the Tauri shell for the host, and for Control by the host)
makes `strata_control/acquisition.py` refuse pip, downloads and compiles; on the host side `server/acquisition.mjs`
refuses the llama.cpp and cloudflared fetches (a connector placed in `components/cloudflared/` is used as it is). Models the
user chooses to download are a separate, explicit action. `apps/host/tests/no-acquire.test.mjs` runs the host under
`tests/helpers/acquire-spy.mjs` and fails on any outbound request or installer process;
`apps/desktop/scripts/installed-flow.mjs` drives an installed app end to end (stub engine anywhere, real engine with `--components`).

## What every platform must produce before it can be called qualified

1. The app launches and the window serves the supplied UI from the host's origin.
2. The owner transport works: one owner per installation, a second client attaches, a stale owner
   can be recovered, and no client can stop a process it does not own (R03/R04).
3. Real inference on a supported pack, with the engine build hash and the model hashes recorded.
4. `docs/work/QUALIFICATION-KIT.md` run end to end, with raw evidence retained.

## Honest gaps in this directory

* There is no CI. Every recipe here is written to be run by hand; making them automated is separate
  work.
* No signing or notarization assets, and no update-feed bootstrap trust. Both need owner-supplied
  credentials (work plan D04).
* No font or asset redistribution evidence has been gathered (D05). The app currently ships the
  brand assets from the supplied UI ZIP, which is fine for an internal candidate and not sufficient
  for redistribution.
* The terminal distribution bundles a private Python runtime and its dependencies for `windows-x86_64`,
  `linux-x86_64` and `darwin-arm64` (`package_terminal.py --target ...`, built on the target OS). The interpreter is
  the pinned python-build-standalone CPython 3.12.15 (digest-checked at build time); the packages come from
  `packaging/locks/runtime-<target>-cp312.txt`, installed with `pip install --require-hashes --no-deps
  --only-binary=:all:` at BUILD time only (`lock_wheels.py` makes the lock; review its diff when a pin changes).
  Wrappers: `bin/strata.cmd` / `bin/strata.ps1` on Windows, `bin/strata` on POSIX; START-HERE.bat, SETUP.bat,
  UPDATE.bat, setup.sh and update.sh use a `python/` folder beside them when there is one.
  `smoke_terminal.py <archive>` extracts an archive into a clean environment (a path with a space, a PATH with no
  Python) and runs `--version`, `doctor`, `profile list`, `install` and `serve --engine mock`. The macOS archive
  predates the lock: build it again on a Mac to get the hash-pinned install (the lock is already here).
* `.gitignore` coverage for build output was added late. `apps/desktop/src-tauri/target` and the
  Node sidecar were briefly committed, which left ~324 MB in the git object store. The files are
  untracked now and the ignore rules are in place, but the history is not rewritten: that is a
  deliberate choice, since rewriting published history is worse than a large local repository. A
  developer cloning this branch gets the current tree, and the sidecar arrives via
  `fetch-node.sh`.
