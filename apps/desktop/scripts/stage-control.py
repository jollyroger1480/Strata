#!/usr/bin/env python3
"""Stage the Control component the installer ships: src-tauri/resources/control.

    python apps/desktop/scripts/stage-control.py                       # build the terminal bundle, stage it
    python apps/desktop/scripts/stage-control.py --archive X.zip       # stage an archive you already built
    python apps/desktop/scripts/stage-control.py --wheelhouse DIR      # build with no network (wheels from DIR)
    python apps/desktop/scripts/stage-control.py --check               # verify what is staged

What lands in `control/`: the pinned CPython 3.12.15 (python-build-standalone, digest-checked), the hash-locked wheels
(`packaging/locks/runtime-<target>-cp312.txt`, installed with --require-hashes at build time), strata_control,
strata_cli, serve, tools/strata_tokenizer.py + gguf_reader.py, and BUNDLE.json (version, interpreter and lock digests,
the digest of the archive it came from). It is the same bundle as the Terminal archive (packaging/package_terminal.py):
one Python runtime-core for both products. The app then never runs pip, npm, git or a compiler to start or serve.

This script is a BUILD step. It may download the interpreter and the wheels (verified); the app never does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "src-tauri" / "resources" / "control"
sys.path.insert(0, str(ROOT / "packaging"))
import package_terminal as pt  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(work: Path, wheelhouse: str | None) -> Path:
    command = [sys.executable, str(ROOT / "packaging" / "package_terminal.py"), "--out", str(work)]
    if wheelhouse:
        command += ["--wheelhouse", wheelhouse]
    subprocess.run(command, check=True, cwd=str(ROOT))
    archives = sorted(work.glob("strata-terminal-*.zip")) + sorted(work.glob("strata-terminal-*.tar.gz"))
    if len(archives) != 1:
        raise SystemExit(f"expected one archive in {work}, found {archives}")
    return archives[0]


def extract(archive: Path, destination: Path) -> Path:
    """Unpack the archive; returns the folder that holds python/ and strata.py."""
    scratch = destination.parent / (destination.name + ".incoming")
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (scratch / member).resolve()
                if scratch.resolve() not in target.parents and target != scratch.resolve():
                    raise SystemExit(f"unsafe member in {archive.name}: {member}")
            zf.extractall(scratch)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(scratch, filter="data")
    (inner,) = [p for p in scratch.iterdir() if p.is_dir()]
    return inner


#: What the app's installer needs on top of the Terminal bundle: the model catalog, GPU probe and install job. The list and the copy
#: live in packaging/package_terminal.py (the Terminal archive carries the same files, for `strata launch`).
INSTALLER_FILES, INSTALLER_TOOLS, INSTALLER_DATA = pt.INSTALLER_FILES, pt.INSTALLER_TOOLS, pt.INSTALLER_DATA
local_imports = pt.local_imports


def add_installer(folder: Path) -> dict:
    """Copy the installer's files from the source tree into the staged Control (after the Terminal bundle is in place)."""
    return pt.add_installer(folder, ROOT)


def write_bundle_record(folder: Path, archive: Path) -> dict:
    manifest_path = archive.with_name(archive.name.rsplit(".", 2 if archive.name.endswith(".tar.gz") else 1)[0] + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    target = manifest.get("target") or pt.host_target()
    record = {
        "schema_version": 1, "component": "strata-control",
        "version": (folder / "VERSION").read_text(encoding="utf-8").strip(),
        "target": target,
        "python": {"version": pt.PYTHON_VERSION, "build": pt.PBS_RELEASE, "sha256": pt.PYTHON_PINS[target]["sha256"]},
        "lock": manifest.get("lock"),
        "installed": manifest.get("installed"),
        "archive": {"name": archive.name, "sha256": sha256(archive), "bytes": archive.stat().st_size},
        "acquires_at_runtime": False,
        "installer": add_installer(folder),
    }
    (folder / "BUNDLE.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return record


def check(folder: Path) -> int:
    target = pt.host_target()
    python = folder / pt.PYTHON_PINS[target]["python"]
    problems = []
    for need in (python, folder / "strata.py", folder / "strata_control", folder / "serve" / "server.py", folder / "BUNDLE.json",
                 folder / "setup.py", folder / "tools" / "strata_mcp.py", folder / "tools" / "iq_pack.py", folder / "data" / "draft_vocab.bin"):
        if not need.exists():
            problems.append(f"missing {need}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    env = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "STRATA_NO_ACQUIRE": "1", "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    done = subprocess.run([str(python), str(folder / "strata.py"), "--version"], capture_output=True, text=True, env=env,
                          cwd=str(folder), timeout=60)
    print(done.stdout.strip() or done.stderr.strip())
    if done.returncode:
        return done.returncode
    # the installer: its catalog, a GPU probe and its imports work on the bundled Python alone (nothing is installed or fetched)
    probe = ("import sys; sys.path.insert(0, %r); import importlib.util as u; s = u.spec_from_file_location('s', %r); m = u.module_from_spec(s); "
             "s.loader.exec_module(m); import iq_pack, strata_pack, pack_index, mtp_fetch; "
             "print('installer ok: %%d models, %%d families' %% (len(m.MODELS), len(m.FAMILIES)))") % (str(folder / "tools"), str(folder / "setup.py"))
    env["STRATA_CONTROL_INSTALL"] = "1"
    again = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, env=env, cwd=str(folder), timeout=120)
    print(again.stdout.strip() or again.stderr.strip()[-600:])
    return again.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="stage this terminal archive instead of building one")
    ap.add_argument("--wheelhouse", help="build with the wheels in this folder (no network)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="verify the staged component and exit")
    a = ap.parse_args()
    if a.check:
        return check(a.out)
    with tempfile.TemporaryDirectory(prefix="strata-stage-control-") as work:
        archive = a.archive.resolve() if a.archive else build_archive(Path(work), a.wheelhouse)
        keep = [p for p in a.out.iterdir() if p.name == ".gitignore"] if a.out.exists() else []
        if a.out.exists():
            for child in a.out.iterdir():
                if child.name != ".gitignore":
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
        a.out.mkdir(parents=True, exist_ok=True)
        inner = extract(archive, a.out)
        record = write_bundle_record(inner, archive)
        for child in inner.iterdir():
            shutil.move(str(child), str(a.out / child.name))
        shutil.rmtree(inner.parent, ignore_errors=True)
        print(f"staged {record['component']} {record['version']} ({record['target']}) into {a.out}")
    return check(a.out)


if __name__ == "__main__":
    raise SystemExit(main())
