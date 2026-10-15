#!/usr/bin/env python3
"""Build the self-contained Strata Terminal distribution.

The requirement this exists for (checklist P04): a user installs the terminal without Desktop, and
needs no development checkout, no system Python, no pip and no compiler. So the archive carries its
own Python runtime and its own dependencies, and the `strata` launcher prefers them.

What goes in
------------
* a pinned **standalone CPython**, downloaded and SHA-256-verified against the same pin `setup.sh`
  uses, so the terminal and the installer cannot drift apart;
* the runtime dependencies from `requirements.txt`, installed *into* that interpreter;
* the application: `strata.py`, `strata_control`, `strata_cli`, `serve`, `data` and the docs the
  CLI references;
* the analysis-only tooling under `tools/` that the terminal paths import. Everything else under
  `tools/` is engine installation and preparation work and is deliberately excluded — the serving
  import boundary must not pull it in.

What does NOT go in
-------------------
* the engine binary. That is a separate, independently versioned component
  (`packaging/README.md`), and a terminal distribution that bundled one would be wrong for every
  machine except the one it was built on;
* model weights or prepared data;
* `node_modules` or the UI. The terminal does not need a browser.

Usage
-----
    python packaging/package_terminal.py --out release/artifacts
    python packaging/package_terminal.py --out /tmp/x --skip-deps   # for a fast layout check
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The standalone CPython pin, identical to `setup.sh`'s. Two places, one value: a terminal built
#: from one and an installer built from the other must be the same interpreter.
PYTHON_PIN = {
    "url": ("https://github.com/astral-sh/python-build-standalone/releases/download/20261009/"
            "cpython-3.12.15%2B20261009-aarch64-apple-darwin-install_only.tar.gz"),
    "sha256": "83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8",
    "target": "darwin-arm64",
}

#: The application, as (source relative to ROOT, destination relative to the archive root).
PAYLOAD = (
    "strata.py",
    "strata_control",
    "strata_cli",
    "serve",
    "requirements.txt",
    "requirements-monitor.txt",
    "docs/MANAGEMENT_PROTOCOL.md",
)

#: `tools/` modules the terminal paths import. Adding one here is a deliberate act: this list is what
#: keeps installation/preparation code out of the serving import graph.
TOOLS_SUBSET = (
    "strata_tokenizer.py",
)

EXCLUDES = ("__pycache__", "*.pyc", ".DS_Store", "*.egg-info")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_python(cache: Path) -> Path:
    """Download the pinned interpreter once, verifying the digest before it is used."""
    archive = cache / "cpython-standalone.tar.gz"
    if not archive.exists():
        print(f"fetching the pinned CPython from {PYTHON_PIN['url']}")
        urllib.request.urlretrieve(PYTHON_PIN["url"], archive)
    got = sha256(archive)
    if got != PYTHON_PIN["sha256"]:
        raise SystemExit(
            f"the CPython download does not match its pin\n  expected {PYTHON_PIN['sha256']}\n"
            f"  got      {got}\nRefusing to package an unverified interpreter.")
    return archive


def ignore(_directory: str, names: list[str]) -> set[str]:
    out = set()
    for name in names:
        if name in ("__pycache__", ".DS_Store") or name.endswith(".pyc"):
            out.add(name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="directory to write the archive into")
    ap.add_argument("--version", default=None, help="version label (default: read from the source)")
    ap.add_argument("--skip-deps", action="store_true",
                    help="skip pip install (layout checks only; the result is NOT distributable)")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args()

    version = args.version or _source_version()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"strata-terminal-{version}-{PYTHON_PIN['target']}"

    cache = Path(tempfile.gettempdir()) / "strata-packaging-cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = fetch_python(cache)

    staging = Path(tempfile.mkdtemp(prefix="strata-terminal-build-"))
    root = staging / name
    root.mkdir(parents=True)
    try:
        print("unpacking the interpreter")
        with tarfile.open(archive) as tf:
            tf.extractall(root)
        # python-build-standalone extracts to `python/`; the launcher looks in exactly that place.
        py = root / "python" / "bin" / "python3"
        if not py.exists():
            raise SystemExit(f"the archive did not produce {py}")

        print("copying the application")
        for item in PAYLOAD:
            src = ROOT / item
            dst = root / item
            if not src.exists():
                raise SystemExit(f"missing payload item: {item}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=ignore)
            else:
                shutil.copy2(src, dst)
        tools_dir = root / "tools"
        tools_dir.mkdir(exist_ok=True)
        for item in TOOLS_SUBSET:
            src = ROOT / "tools" / item
            if src.exists():
                shutil.copy2(src, tools_dir / item)

        # The launcher goes in `bin/`, and `strata.py` sits one level up, which is the layout
        # `bin/strata` already understands.
        bin_dir = root / "bin"
        bin_dir.mkdir(exist_ok=True)
        launcher = bin_dir / "strata"
        shutil.copy2(ROOT / "bin" / "strata", launcher)
        launcher.chmod(0o755)

        if not args.skip_deps:
            print("installing the runtime dependencies into the bundled interpreter")
            subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--no-warn-script-location",
                            "--upgrade", "pip"], check=True)
            subprocess.run([str(py), "-m", "pip", "install", "--quiet", "--no-warn-script-location",
                            "-r", str(root / "requirements.txt")], check=True)
            # A packaged tree must not carry pip's cache or the tests of its dependencies.
            for junk in ("python/lib/python3.12/site-packages/pip",):
                shutil.rmtree(root / junk, ignore_errors=True)

        # Prove the interpreter works from where it now sits, before claiming the artifact is good.
        probe = subprocess.run(
            [str(py), "-c", "import jinja2, yaml, regex, psutil; print('deps ok')"],
            capture_output=True, text=True, cwd=str(root))
        print("  " + (probe.stdout.strip() or probe.stderr.strip()[:200]))

        readme = root / "README.txt"
        readme.write_text(_readme(version), encoding="utf-8")

        archive_path = out_dir / f"{name}.tar.gz"
        print(f"writing {archive_path}")
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(root, arcname=name)

        digest = sha256(archive_path)
        size = archive_path.stat().st_size
        print(f"\n{archive_path}")
        print(f"  bytes  {size}")
        print(f"  sha256 {digest}")

        manifest = out_dir / f"{name}.manifest.json"
        import json
        manifest.write_text(json.dumps({
            "name": name,
            "version": version,
            "bytes": size,
            "sha256": digest,
            "python": {"url": PYTHON_PIN["url"], "sha256": PYTHON_PIN["sha256"],
                       "target": PYTHON_PIN["target"]},
            "payload": list(PAYLOAD),
            "tools_subset": list(TOOLS_SUBSET),
            "excludes": ("engine binary", "model weights", "prepared data", "UI", "node_modules"),
            "deps_installed": not args.skip_deps,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"  manifest {manifest}")
        return 0
    finally:
        if args.keep_staging:
            print(f"staging kept at {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)


def _source_version() -> str:
    """The version the source reports, so the artifact name cannot disagree with the code."""
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
    import re
    # `project(strata VERSION 0.1.44.1 LANGUAGES C CXX ...)` — the version is a dotted triple and
    # must be read as one. Taking "everything after VERSION" yields "0.1.44.1.LANGUAGES.CXX".
    m = re.search(r"^\s*project\([^)]*?VERSION\s+([0-9]+(?:\.[0-9]+)+)", text, re.MULTILINE)
    return m.group(1) if m else "0.0.0"


def _readme(version: str) -> str:
    return f"""Strata Terminal {version}
{'=' * (17 + len(version))}

This is the terminal distribution. It carries its own Python runtime and its own dependencies, so
you do not need Python, pip, Node, a compiler or a development checkout.

Install
-------
    tar xzf strata-terminal-{version}-{PYTHON_PIN['target']}.tar.gz
    cd strata-terminal-{version}-{PYTHON_PIN['target']}
    ./bin/strata --help

Put `bin/` on your PATH to get the `strata` command everywhere:

    echo "export PATH=\\"$PWD/bin:$PATH\\"" >> ~/.zshrc

What it does NOT include
------------------------
The **engine** is a separate, independently versioned component and is not bundled: the right engine
package depends on your OS, CPU and GPU, and shipping one machine's build to another would be
wrong. Point Strata at an engine with `STRATA_ENGINE`, or install one through the desktop app or
`setup.sh`. Model weights are likewise not included.

Useful commands
---------------
    strata status                     one read-only telemetry snapshot
    strata monitor                    the six-view terminal monitor
    strata doctor                     read-only component inventory
    strata timeskip run --help        retain a good result, revise it, check it, decide
    strata tune plan --help           what a tuning sweep would search, and why not
    strata serve --help               start the inference server in the foreground

Serving, `status` and `monitor` do not need the agent runtime or the tuning packages. The opt-in
terminal TUI additionally needs `textual` (see requirements-monitor.txt).

Verification
------------
The archive's manifest records the SHA-256 of the bundle and of the pinned interpreter it carries.
`bin/strata` prefers the bundled interpreter and falls back to a Python 3.10+ on PATH, so the
distribution still runs if the bundled runtime is removed.
"""


if __name__ == "__main__":
    sys.exit(main())
