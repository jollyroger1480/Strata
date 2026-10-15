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
* The terminal distribution **does** now bundle a private Python runtime and its dependencies
  (42 MB, built by `package_terminal.py`), and was verified running from a moved directory in a
  clean environment with no usable system Python. What remains is per-platform work: the current
  archive is `darwin-arm64` only. `PYTHON_PIN` in that script needs the matching
  python-build-standalone archive and digest for each target, and the launcher already prefers a
  bundled runtime, so a Linux or Windows archive is a pin change plus a build on that OS.
* `.gitignore` coverage for build output was added late. `apps/desktop/src-tauri/target` and the
  Node sidecar were briefly committed, which left ~324 MB in the git object store. The files are
  untracked now and the ignore rules are in place, but the history is not rewritten: that is a
  deliberate choice, since rewriting published history is worse than a large local repository. A
  developer cloning this branch gets the current tree, and the sidecar arrives via
  `fetch-node.sh`.
