# Updates: the signed feed, the component updater, the app updater

Everything that downloads software on a user's PC lives behind one explicit step: `strata update` in the Terminal, or
"Update engine" / "Update Strata" in the app's Settings. Starting a model and serving never download anything
(`STRATA_NO_ACQUIRE` stays on for them; the update step is a separate Control process that is allowed to). Engines are
separate from the app and ship more often: an app-only update downloads 0 engine bytes, an engine update downloads only the
package for that PC's GPU. **Models and saved tuning presets are never downloaded, replaced or deleted by any update.**

## The three things that are updated

| What | How | Signed with |
|---|---|---|
| Engine package (per platform / backend / variant) | Control's component updater: download with resume, verify, install beside the old one, test-start, switch | the feed key (Ed25519) |
| Control (Python runtime, `strata_control`, CLI, server) | the same updater: staged beside the running one, run once, then `components/control/active.json` points at it | the feed key |
| The desktop app | Tauri's signed updater in the shell, reached by the host through a token-protected loopback bridge | the Tauri updater key (minisign) |

Two separate private keys, both kept by the owner, neither in git, neither the same as Apple/Windows code signing:

* **feed key**: Ed25519, signs `strata-feed.json` and every engine package manifest.
* **Tauri updater key**: minisign (what `tauri signer generate` makes), signs the app installers.

## The feed (`strata-feed.json` + `strata-feed.json.sig`)

`strata-feed.json.sig` is `{"alg": "ed25519", "key_id": "<16 hex>", "value": "<base64>"}` over the **exact bytes** of
`strata-feed.json` (no canonicalisation). The feed:

```json
{
  "schema_version": 1, "feed": "strata-updates", "channel": "stable",
  "sequence": 12,                           // goes up with every feed; an older one than the last accepted is refused
  "generated_at": "2026-10-20T10:00:00Z", "expires_at": "2027-10-20T10:00:00Z",   // expires_at may be null
  "app": {                                  // exactly Tauri's static updater JSON, plus sha256/size per platform
    "version": "0.2.0", "notes": "...", "pub_date": "2026-10-20T10:00:00Z",
    "platforms": {"windows-x86_64": {"url": "app/Strata_0.2.0_x64-setup.exe", "signature": "<minisign, base64>", "sha256": "...", "size": 71000000}}
  },
  "control": {"version": "0.1.46", "packages": [{"platform": "linux-x86_64", "format": "tar.gz", "url": "control/...tar.gz",
                                                  "size": 0, "sha256": "...", "installed_bytes": 0, "entry": "strata.py"}]},
  "engine": {"packages": [{
    "version": "0.1.46", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda13",
    "archs": ["sm_75", "sm_86", "sm_89", "sm_120", "compute_120"], "min_driver": {"nvidia": "580.65"}, "min_runtime": null,
    "build_id": "be885026f96a-cuda13.0", "format": "tar.gz", "url": "engine/strata-engine-0.1.46-linux-x86_64-cuda-cuda13.tar.gz",
    "size": 505568970,                      // bytes of the archive
    "sha256": "...",                        // of the archive
    "manifest_sha256": "...",               // of the package's engine-component.json (it holds every file's hash)
    "installed_bytes": 840893419, "notes": ""}]}
}
```

`url` is relative to the feed's own location (or an absolute `https://` address; a loopback `http://` is allowed so a local test
feed works; a local feed folder may only name files inside itself).
For a GitHub release the assets are flat: pack with `--url-prefix ""` and upload every file beside the feed.

### What a PC checks, in order

1. the signature of the feed (key ids it trusts: `strata_control/update_keys.json`, plus `STRATA_UPDATE_TRUSTED_KEYS` for a test setup, which
   the CLI announces);
2. `channel`, `expires_at`, and `sequence` not lower than the last accepted feed (the downgrade guard for a replayed old feed);
3. the engine package that fits **this** PC (`feed.select_engine`): platform, backend (CUDA / HIP / SYCL / Metal as detected),
   CUDA `archs` against the card (a `compute_NN` PTX entry covers newer cards), HIP `archs` against the card's gfx name,
   `min_driver` against the NVIDIA driver (this is how CUDA 13 beats CUDA 12 on a new driver and CUDA 12 is chosen on an old one),
   `min_runtime` against Control; newest version wins, and equal versions prefer the build that needs the newer driver;
4. never older than what is installed (an update never downgrades; only `strata components rollback` goes back).

### What an engine update does

download (resume, progress) -> SHA-256 of the archive = the feed's -> unpack into `components/.updates/staging/` (names and links checked,
total size limited to the feed's `installed_bytes`) -> the manifest is the one the feed names (`manifest_sha256`) and says what the feed says
-> the manifest's own signature verifies -> every file's hash -> fits this PC -> **test-start from the staging folder through the real start
path** (isolated owner runtime and profile, `component:strata-engine`, an installed model, one-token probe, stop) -> only then rename
into `components/engine/` and re-resolve to make sure a start now picks it. Any failure leaves `engine/` exactly as it was.
The previous version is kept (`state.json` `engine.previous`); the one before it is removed. With no model installed yet (first run)
the test is `--version` only, and the report says so.

Not while an engine is serving: the check uses the activity lease `strata serve` and the owner runtime already hold. `--wait SECONDS` waits;
a terminal asks whether to wait; scripts get exit 7. The download itself may run first; the check is repeated right before the test start and
before the switch.

Saved tuning presets: when the engine identity changed (version, build id, manifest hash, backend) `tuning/recheck.json` gets a
`pending` entry (`engine_changed`, from/to identities). Presets are not opened. The tuner offers the ~2 minute check for them and calls
`preset_marks.clear(id)`; rolling back to the engine they were measured on clears the debt again.

### Commands

    strata update [--check] [--app] [--no-engine] [--no-control] [--feed URL|DIR] [--wait S] [--force] [--model-config RUN.json]
    strata components update            # the same
    strata components rollback          # back to the previous engine; the next update will not re-offer the version rolled back from (--force does)
    strata components engine            # shows "update available (...)" from the last check, no network

Exit codes: 0 done or nothing to do, 5 refused (signature, hash, downgrade, no trusted key, acquiring forbidden),
6 failed with the old version still in use (network, test start), 7 an engine is serving, 130 stopped.
`--output json` / `--output jsonl` (progress events, last line is the result) are what the app's host uses.

## How the owner makes the real keys (once)

On an offline machine, outside any git checkout:

    python packaging/feed_tool.py keygen --out D:\strata-keys          # feed-key.json (SECRET) + feed-key.pub.json
    npx tauri signer generate -w D:\strata-keys\tauri-updater.key      # asks for a password; prints the public key

Back both up (a lost feed key means every installed PC needs a build with a new key; a lost Tauri key means the same for the app).
Then, in git, once, in the commit that ships the first signed release:

* the `keys` entry from `feed-key.pub.json` into `strata_control/update_keys.json`;
* the Tauri public key into `apps/desktop/src-tauri/tauri.conf.json` (`plugins.updater.pubkey`) and into `update_keys.json` as `app_minisign_public`
  (the Terminal verifies the app installer with it).

Until then `update_keys.json` is empty and every feed is refused with "this build trusts no update key": a build with no key cannot install anything.
A key rotation is a release that carries both public keys; the old private key signs that release, the new one signs the next.

## Releasing (when the owner says go)

    python packaging/feed_tool.py sign-package <engine package folder> --key <feed-key.json>
    python packaging/feed_tool.py pack-engine  <engine package folder> --out RELEASE/engine
    python packaging/feed_tool.py pack-control <control bundle folder> --version V --platform P --out RELEASE/control
    TAURI_SIGNING_PRIVATE_KEY=<key> npx tauri build --config tauri.release.conf.json         # makes the signed installers
    python packaging/feed_tool.py build-feed --out RELEASE --sequence N --key <feed-key.json> --engine RELEASE/engine --control RELEASE/control --app-json <tauri latest.json>
    python packaging/feed_tool.py verify-feed RELEASE --pub <feed-key.pub.json>

Where the feed is hosted (GitHub release assets, a static host, ...) is the owner's decision; the default address in
`strata_cli/update_cmd.py` is a placeholder until then, and nothing in this tree uploads anything.

## Testing locally

Everything above runs against a folder or a loopback server: `tests/terminal/test_update_feed.py` (signatures, hashes, downgrades, selection),
`tests/terminal/test_updater.py` (end to end against a loopback HTTP feed with stub engines: update, test-start, switch; a failing test start
keeps the old engine; an interrupted download resumes; rollback; Control; the app installer's minisign check), `apps/host/tests/component-updates.test.mjs`
(the Settings buttons). For a real engine on a box: `phaseA-tests/split/s6/` (prepare + driver).
