# Engine component manifest (schema 2)

One format for every engine package: Windows CUDA 13 and 12, Windows HIP, Linux CUDA, Linux HIP, Linux SYCL and macOS
Metal. Control (`strata_control/engine_component.py`) reads it, checks it, and only then starts the engine.
`packaging/package_engine.py` writes it. Packages are local and unsigned for now; publishing and signing are S6.

## Folder

    <components>/engine/<version>-<platform>-<backend>/
        engine-component.json
        strata | strata.exe              the entrypoint
        lib*.so* | *.dll | *.dylib       the GPU runtime libraries it loads from beside itself
        BUILD.json  LICENSE*             optional records
        rocm/**  strata-vision.exe  ...  anything else the build needs, listed in the manifest like the rest

`<components>` is `%LOCALAPPDATA%\Strata\components`, `$XDG_DATA_HOME/strata/components` or `STRATA_COMPONENTS_DIR`.
A package folder is immutable: a new build is a new folder.

## `engine-component.json`

| field | type | meaning |
|---|---|---|
| `schema_version` | 2 | schema 1 packages (no `archs`, `min_driver`, `build_id`, `signature`; `cuda_archs` instead of `archs`) still load |
| `id`, `name` | `"strata-engine"` | the component name; anything else is refused |
| `version` | string | the engine release, e.g. `0.1.45`; the newest compatible version wins |
| `platform` | string | `windows-x86_64`, `linux-x86_64`, `macos-arm64` (must match this PC) |
| `backend` | string | `cuda`, `hip`, `sycl`, `metal`, `cpu`; must be a combination `BACKEND_MATRIX` commits to, and the one this PC's GPU needs |
| `archs` | list of strings | what the build contains. CUDA: `sm_86` (machine code for that card), `compute_89` (PTX, runs on that generation or newer). HIP: `gfx1201`. SYCL and Metal: a descriptive tag (`intel-arc`, `apple-gpu`). Required for every non-`cpu` backend. Enforced for CUDA against `nvidia-smi` compute capability; recorded for the others |
| `min_driver` | `{vendor: version}` | the oldest driver the build runs on, e.g. `{"nvidia": "580.65"}`. Enforced for NVIDIA where `nvidia-smi` answers; an unreadable driver is not a refusal |
| `build_id` | string | which engine build this is: source commit and toolkit, e.g. `be885026f96a-cuda13.0`. Required |
| `min_runtime` | string, optional | the oldest Control that may start it |
| `entrypoint` | string | relative path of the binary; must be in `files` |
| `files` | list | `{path, sha256, bytes}` for **every** file of the package except the manifest; paths are plain relative POSIX paths that stay inside the folder; no duplicates |
| `notes` | string, optional | free text |
| `signature` | `null` or `{alg, key_id, value}` | see below |

Example (shortened):

```json
{"schema_version": 2, "id": "strata-engine", "name": "strata-engine", "version": "0.1.45",
 "platform": "linux-x86_64", "backend": "cuda", "archs": ["sm_86"], "min_driver": {"nvidia": "580.65"},
 "build_id": "be885026f96a-cuda13.0", "entrypoint": "strata",
 "files": [{"path": "strata", "sha256": "<64 hex>", "bytes": 191000000}], "signature": null}
```

## What Control checks before a start

In this order, each refusal names the cause; none of them starts anything and none of them downloads anything:

1. the manifest parses strictly (types, schema, paths, archs, driver, signature shape);
2. the platform and backend are this PC's;
3. `min_runtime`, CUDA archs vs. the card, `min_driver` vs. the driver;
4. the **signature policy** (below);
5. **every file** exists, is a regular file inside the folder (no symlinks), has the listed size, **and matches its SHA-256**
   (a start hashes the whole package; the listing commands `strata components engine` hash only the entrypoint unless given `--full`);
6. on POSIX the entrypoint is executable.

If a package fails (5) or (4) and no other package fits this PC, the start ends with `EngineIntegrityError`
(exit code 5): "The engine package failed its integrity check, so it was not started: 0.1.45 linux-x86_64-cuda:
libcudart.so.13 does not match its recorded SHA-256. Repair or reinstall the engine component from the Terminal". A package
for another GPU or an older driver is `EngineMissing` with its reason. A tampered newer package never falls back silently
to nothing: a good older package is used and the tampered one is listed under `problems` in `strata components engine`.

## Signature (S6)

`signature` is `null` today. S6 fills it and registers the verifier:

    engine_component.register_signature_verifier(fn)       # fn(payload: bytes, signature: dict, component) -> None, raises if bad
    engine_component.signing_payload(manifest_dict)         # canonical bytes the signature covers

The payload is the manifest without its `signature` member, as canonical JSON (sorted keys, `,` and `:` separators,
UTF-8). It contains every file hash, so one signature covers the whole package. Policy:

* unsigned package: loads, reported `unsigned`;
* signed, no verifier registered: loads, reported `unverified` (never "verified");
* signed, verifier registered: must verify, otherwise refused;
* `STRATA_REQUIRE_SIGNED_COMPONENTS=1`: unsigned and unverifiable packages are refused (the setting S6 turns on for released builds).

## Building a package

    python packaging/package_engine.py --src <build folder> --backend cuda --version 0.1.45 \
        --out <components>/engine --arch sm_86 --arch compute_89 --min-driver nvidia=580.65 \
        [--build-id <id>] [--platform linux-x86_64] [--include 'rocm/**' --include strata-vision.exe | --tree]
    python packaging/package_engine.py --verify <package folder>       # full hash check

Without `--include`/`--tree` it takes the entrypoint, the libraries beside it, `BUILD.json` and `LICENSE*`. The build id
defaults to the checkout's commit (`-dirty` when the tree has changes). It prints the size, file count and the SHA-256 of
the manifest file (which holds every file's hash), used as the package's digest in notes.
