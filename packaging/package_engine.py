#!/usr/bin/env python3
"""Make a local, unpublished engine package for the component folder (strata_control/engine_component.py).

    python packaging/package_engine.py --src <folder with strata[.exe] and its libraries> --backend cuda \
        --version 0.1.45 [--platform linux-x86_64] --out <components>/engine [--cuda-arch sm_86 ...]

The result is one folder `<out>/<version>-<platform>-<backend>/` holding the engine binary, the shared libraries
next to it, BUILD.json when the source has one, and `engine-component.json` (every file's SHA-256 and size). It
is the thing the app resolves; it is not signed, not published, and nothing here uploads it. Publishing and
signing are stream S6.

Files taken from --src: the entrypoint, every `*.dll` / `*.so` / `*.so.*` / `*.dylib` in the same folder (the GPU
runtime libraries the engine loads from beside itself), `BUILD.json`, and `LICENSE*` when present. Nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strata_control import engine_component as ec  # noqa: E402
from strata_control.components import BACKEND_MATRIX, host_platform  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def wanted(path: Path) -> bool:
    name = path.name.lower()
    return (name.endswith((".dll", ".so", ".dylib")) or ".so." in name or name == "build.json"
            or name.startswith("license"))


def build(src: Path, backend: str, version: str, target: str, out: Path, *, entry: str | None = None,
          cuda_archs: tuple[str, ...] = (), min_runtime: str | None = None, notes: str = "") -> Path:
    if backend not in BACKEND_MATRIX:
        raise SystemExit(f"unknown backend {backend!r}")
    entrypoint = entry or ("strata.exe" if target.startswith("windows") else "strata")
    if not (src / entrypoint).is_file():
        raise SystemExit(f"{src / entrypoint} does not exist")
    folder = out / f"{version}-{target}-{backend}"
    if folder.exists():
        raise SystemExit(f"{folder} already exists; remove it first (packages are immutable)")
    staging = out / f".{folder.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        names = [entrypoint] + sorted(p.name for p in src.iterdir() if p.is_file() and p.name != entrypoint and wanted(p))
        files = []
        for name in names:
            shutil.copy2(src / name, staging / name)
            files.append({"path": name, "sha256": sha256(staging / name), "bytes": (staging / name).stat().st_size})
        manifest = {"schema_version": ec.SCHEMA_VERSION, "id": ec.COMPONENT_ID, "version": version,
                    "platform": target, "backend": backend, "entrypoint": entrypoint, "files": files}
        if min_runtime:
            manifest["min_runtime"] = min_runtime
        if cuda_archs:
            manifest["cuda_archs"] = list(cuda_archs)
        if notes:
            manifest["notes"] = notes
        (staging / ec.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        ec.verify(ec.read_manifest(staging), full=True)
        staging.rename(folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--backend", required=True, choices=sorted(BACKEND_MATRIX))
    ap.add_argument("--version", required=True)
    ap.add_argument("--platform", default=None, help="default: this machine")
    ap.add_argument("--out", required=True, type=Path, help="the component folder's engine/ directory")
    ap.add_argument("--entrypoint", default=None)
    ap.add_argument("--cuda-arch", action="append", default=[], help="e.g. sm_86; repeat for several")
    ap.add_argument("--min-runtime", default=None)
    ap.add_argument("--notes", default="")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    folder = build(a.src, a.backend, a.version, a.platform or host_platform(), a.out, entry=a.entrypoint,
                   cuda_archs=tuple(a.cuda_arch), min_runtime=a.min_runtime, notes=a.notes)
    total = sum(p.stat().st_size for p in folder.iterdir())
    print(f"{folder}  ({total / (1 << 20):.1f} MiB, {len(list(folder.iterdir()))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
