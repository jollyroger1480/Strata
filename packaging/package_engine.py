#!/usr/bin/env python3
"""Make a local, unpublished engine package for the component folder (strata_control/engine_component.py).

    python packaging/package_engine.py --src <folder with strata[.exe] and its libraries> --backend cuda \
        --version 0.1.45 --out <components>/engine --arch sm_86 --arch compute_89 \
        --min-driver nvidia=580.65 [--build-id <id>] [--platform linux-x86_64] [--tree] [--include GLOB ...]

    python packaging/package_engine.py --verify <components>/engine/<package folder>     # full hash check, exit 0 / 1

The result is one folder `<out>/<version>-<platform>-<backend>/` holding the engine binary, the libraries it loads,
BUILD.json when the source has one, and `engine-component.json` (schema 2: every file's SHA-256 and size, the arch
list, the oldest driver, the engine build id, and an empty `signature` for S6). It is the thing Control resolves,
checks and starts; it is not signed, not published, and nothing here uploads it. Signing is stream S6
(`engine_component.register_signature_verifier`); `--sign-with` is deliberately absent until then.

Files taken from --src, by default: the entrypoint, every `*.dll` / `*.so` / `*.so.*` / `*.dylib` / `*.metallib`
beside it, `BUILD.json`, and `LICENSE*`. `--include GLOB` (repeatable, relative to --src, `**` allowed) adds more, for
example `strata-vision.exe`, `strata-device.exe` or `rocm/**`. `--tree` takes every regular file under --src (the HIP
package ships a `rocm/` tree of kernels). Symlinks are never followed or copied.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
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
    return (name.endswith((".dll", ".so", ".dylib", ".metallib")) or ".so." in name or name == "build.json"
            or name.startswith("license"))


def git_build_id(repo: Path = ROOT) -> str:
    """The source commit of the engine build, `-dirty` when the tree had changes. `unknown` outside a checkout."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head + ("-dirty" if dirty else "")) if head else "unknown"


def select(src: Path, entrypoint: str, *, tree: bool, include: list[str]) -> list[str]:
    """Relative POSIX paths to package, entrypoint first, the rest sorted."""
    chosen: set[str] = set()
    for path in sorted(src.rglob("*") if (tree or include) else src.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(src).as_posix()
        if rel == ec.MANIFEST:
            continue
        top_level = "/" not in rel
        if tree or (top_level and wanted(path)) or any(fnmatch.fnmatch(rel, pattern) for pattern in include):
            chosen.add(rel)
    chosen.discard(entrypoint)
    return [entrypoint] + sorted(chosen)


def build(src: Path, backend: str, version: str, target: str, out: Path, *, entry: str | None = None,
          archs: tuple[str, ...] = (), min_driver: dict | None = None, build_id: str | None = None,
          min_runtime: str | None = None, notes: str = "", tree: bool = False, include: list[str] | None = None,
          variant: str = "") -> Path:
    if backend not in BACKEND_MATRIX:
        raise SystemExit(f"unknown backend {backend!r}")
    if backend != "cpu" and not archs:
        raise SystemExit(f"a {backend} package needs at least one --arch (sm_86, gfx1201, ...)")
    entrypoint = entry or ("strata.exe" if target.startswith("windows") else "strata")
    if not (src / entrypoint).is_file():
        raise SystemExit(f"{src / entrypoint} does not exist")
    folder = out / (f"{version}-{target}-{backend}" + (f"-{variant}" if variant else ""))
    if folder.exists():
        raise SystemExit(f"{folder} already exists; remove it first (packages are immutable)")
    staging = out / f".{folder.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        files = []
        for rel in select(src, entrypoint, tree=tree, include=include or []):
            destination = staging / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, destination)
            files.append({"path": rel, "sha256": sha256(destination), "bytes": destination.stat().st_size})
        manifest = {"schema_version": ec.SCHEMA_VERSION, "id": ec.COMPONENT_ID, "name": ec.COMPONENT_ID,
                    "version": version, "platform": target, "backend": backend, "archs": list(archs),
                    "min_driver": dict(min_driver or {}), "build_id": build_id or git_build_id(),
                    "entrypoint": entrypoint, "files": files, "signature": None}
        if variant:
            manifest["variant"] = variant
        if min_runtime:
            manifest["min_runtime"] = min_runtime
        if notes:
            manifest["notes"] = notes
        (staging / ec.MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        if os.name != "nt":
            (staging / entrypoint).chmod(0o755)
        ec.verify(ec.read_manifest(staging), full=True)
        staging.rename(folder)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder


def package_sha256(folder: Path) -> str:
    """One digest for the whole package: SHA-256 of the manifest file (it holds every file's hash)."""
    return sha256(folder / ec.MANIFEST)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", type=Path, metavar="FOLDER", help="hash-check an existing package and exit")
    ap.add_argument("--src", type=Path)
    ap.add_argument("--backend", choices=sorted(BACKEND_MATRIX))
    ap.add_argument("--version")
    ap.add_argument("--platform", default=None, help="default: this machine")
    ap.add_argument("--out", type=Path, help="the component folder's engine/ directory")
    ap.add_argument("--entrypoint", default=None)
    ap.add_argument("--arch", "--cuda-arch", dest="arch", action="append", default=[],
                    help="sm_86 (code for that card), compute_89 (PTX), gfx1201, ...; repeat for several")
    ap.add_argument("--min-driver", action="append", default=[], metavar="VENDOR=VERSION",
                    help="the oldest driver, e.g. nvidia=580.65; repeat for several vendors")
    ap.add_argument("--variant", default="", help="tag when one backend has several builds (cuda13, cuda12); goes in the folder name")
    ap.add_argument("--build-id", default=None, help="default: the source commit of this checkout")
    ap.add_argument("--min-runtime", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--tree", action="store_true", help="every regular file under --src, recursively")
    ap.add_argument("--include", action="append", default=[], metavar="GLOB", help="more files, relative to --src")
    a = ap.parse_args(argv)
    if a.verify:
        try:
            component = ec.read_manifest(a.verify)
            ec.verify(component, full=True)
        except ec.ComponentProblem as exc:
            print(f"{a.verify}: REFUSED: {exc}")
            return 1
        print(f"{a.verify}: ok ({len(component.files)} files, manifest sha256 {package_sha256(a.verify)})")
        return 0
    for needed in ("src", "backend", "version", "out"):
        if getattr(a, needed) is None:
            ap.error(f"--{needed} is required")
    drivers = {}
    for item in a.min_driver:
        vendor, _, version = item.partition("=")
        if not vendor or not version:
            ap.error(f"--min-driver {item!r} is not VENDOR=VERSION")
        drivers[vendor] = version
    a.out.mkdir(parents=True, exist_ok=True)
    folder = build(a.src, a.backend, a.version, a.platform or host_platform(), a.out, entry=a.entrypoint,
                   archs=tuple(a.arch), min_driver=drivers, build_id=a.build_id, min_runtime=a.min_runtime,
                   notes=a.notes, tree=a.tree, include=a.include, variant=a.variant)
    entries = [p for p in folder.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in entries)
    print(f"{folder}  ({total / (1 << 20):.1f} MiB, {len(entries)} files, manifest sha256 {package_sha256(folder)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
