"""Finding the engine package for this machine in a local component folder. There is no download path.

The engine (the C++ `strata` binary and its GPU runtime libraries) is a separate, independently versioned
component. The desktop app and the terminal start whatever is installed in the component folder; nothing here, and
nothing on the start path, fetches an engine. A missing or unusable engine is a typed error that says what to do
(`EngineMissing`, exit code 5, the same family as `acquisition.AcquisitionRefused`) and never an attempt to get one.

Layout (``components_dir()``)::

    <components>/engine/<version>-<platform>-<backend>/
        engine-component.json        the manifest (below)
        strata.exe | strata          the entrypoint (path from the manifest)
        *.dll | *.so                 the GPU runtime libraries, hashed in the manifest too
        BUILD.json                   optional, the engine's own build record

Manifest (``engine-component.json``)::

    {"schema_version": 1, "id": "strata-engine", "version": "0.1.45", "platform": "windows-x86_64",
     "backend": "cuda", "entrypoint": "strata.exe", "min_runtime": "0.1.45",
     "cuda_archs": ["sm_86"],                      # optional: GPU architectures the build contains
     "files": [{"path": "strata.exe", "sha256": "...", "bytes": 1}, ...]}

Checks, in order, and each one names what failed:

* the manifest parses strictly; file paths are relative, plain, and stay inside the component folder;
* the platform and backend are combinations the project commits to (`components.BACKEND_MATRIX`);
* the backend is the one this machine's GPU needs (detected, or asked for);
* every listed file exists with the listed size (always, it is cheap), and the entrypoint's SHA-256 matches;
  `verify(full=True)` hashes every file (a few hundred MB of libraries: used by Repair and by tests);
* the runtime that is asking is new enough for `min_runtime`.

Nothing here imports anything outside the standard library and `strata_control`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from strata_control import acquisition
from strata_control.targets import BACKEND_MATRIX, backend_supported, host_platform

SCHEMA_VERSION = 1
MANIFEST = "engine-component.json"
COMPONENT_ID = "strata-engine"
ENV_DIR = "STRATA_COMPONENTS_DIR"
ENV_BACKEND = "STRATA_ENGINE_BACKEND"

#: the version of the control/runtime that is asking; the app reads it from VERSION when it has one.
RUNTIME_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"


class EngineMissing(acquisition.AcquisitionRefused):
    """No usable engine component for this machine. Nothing was fetched and nothing will be."""

    code = "engine_component_missing"
    exit_code = 5

    def __init__(self, reason: str, root: Path | None = None, *, problems: list[dict] | None = None):
        self.reason = reason
        self.root = root
        self.problems = problems or []
        where = f" in {root}" if root else ""
        RuntimeError.__init__(
            self,
            f"No engine for this PC was found{where}: {reason}. The app never downloads an engine while it "
            f"starts or serves. Install the engine component for your GPU from the Terminal "
            f"(`strata install`) or copy an engine package into the component folder.")
        self.what = "the engine component"
        self.command = "strata install"


class ComponentProblem(ValueError):
    """One engine package is unusable. The message says why; the caller decides what to do next."""


@dataclass
class EngineComponent:
    id: str
    version: str
    platform: str
    backend: str
    entrypoint: str
    files: tuple[tuple[str, str, int], ...]      # (relative path, sha256, bytes)
    root: Path
    min_runtime: str | None = None
    cuda_archs: tuple[str, ...] = ()
    notes: str = ""

    @property
    def exe(self) -> Path:
        return self.root / self.entrypoint

    def to_dict(self) -> dict:
        return {"id": self.id, "version": self.version, "platform": self.platform, "backend": self.backend,
                "entrypoint": self.entrypoint, "exe": str(self.exe), "root": str(self.root),
                "min_runtime": self.min_runtime, "cuda_archs": list(self.cuda_archs),
                "file_count": len(self.files), "bytes": sum(item[2] for item in self.files)}


# ----------------------------------------------------------------------------------------------- where

def components_dir(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The per-user component folder. ``STRATA_COMPONENTS_DIR`` wins (the app and the tests set it)."""
    environ = os.environ if environ is None else environ
    explicit = environ.get(ENV_DIR)
    if explicit:
        return Path(explicit)
    windows = os.name == "nt" if windows is None else windows
    home = Path.home() if home is None else home
    if windows:
        return Path(environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")) / "Strata" / "components"
    return Path(environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")) / "strata" / "components"


def engine_dir(root: Path | None = None) -> Path:
    return (root or components_dir()) / "engine"


# ----------------------------------------------------------------------------------------------- manifest

def _version_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else 0 for part in re.split(r"[.\-+]", value))


def _plain_relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\0" in path:
        raise ComponentProblem("a file path is empty")
    for flavour in (PurePosixPath(path), PureWindowsPath(path)):
        if flavour.is_absolute() or flavour.drive or flavour.root:
            raise ComponentProblem(f"file path {path!r} is absolute")
        if ".." in flavour.parts:
            raise ComponentProblem(f"file path {path!r} leaves the component folder")
    return path.replace("\\", "/")


def read_manifest(folder: Path) -> EngineComponent:
    path = folder / MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ComponentProblem(f"{MANIFEST} is missing") from None
    except (OSError, ValueError) as exc:
        raise ComponentProblem(f"{MANIFEST} is not valid JSON ({type(exc).__name__})") from None
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ComponentProblem(f"{MANIFEST} has an unsupported schema_version")
    if data.get("id") != COMPONENT_ID:
        raise ComponentProblem(f"{MANIFEST} is not an engine component (id {data.get('id')!r})")
    for key in ("version", "platform", "backend", "entrypoint"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ComponentProblem(f"{MANIFEST} has no {key}")
    if data["backend"] not in BACKEND_MATRIX:
        raise ComponentProblem(f"unknown backend {data['backend']!r}")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ComponentProblem(f"{MANIFEST} lists no files")
    parsed = []
    for item in files:
        if not isinstance(item, dict):
            raise ComponentProblem("a file entry is not an object")
        rel = _plain_relative(item.get("path"))
        digest = str(item.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ComponentProblem(f"{rel} has no valid sha256")
        size = item.get("bytes")
        if type(size) is not int or size < 0:
            raise ComponentProblem(f"{rel} has no valid size")
        parsed.append((rel, digest, size))
    entry = _plain_relative(data["entrypoint"])
    if entry not in {item[0] for item in parsed}:
        raise ComponentProblem(f"the entrypoint {entry} is not in the file list")
    archs = data.get("cuda_archs") or []
    if not isinstance(archs, list) or not all(isinstance(a, str) for a in archs):
        raise ComponentProblem("cuda_archs must be a list of strings")
    return EngineComponent(
        id=data["id"], version=data["version"], platform=data["platform"], backend=data["backend"],
        entrypoint=entry, files=tuple(parsed), root=folder, min_runtime=data.get("min_runtime"),
        cuda_archs=tuple(archs), notes=str(data.get("notes") or ""))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(component: EngineComponent, *, full: bool = False) -> None:
    """Raise ComponentProblem unless the files are what the manifest says. `full` hashes every file."""
    root = component.root.resolve()
    for rel, digest, size in component.files:
        target = component.root / rel
        try:
            info = target.lstat()
        except OSError:
            raise ComponentProblem(f"{rel} is missing") from None
        if target.is_symlink() or not target.is_file():
            raise ComponentProblem(f"{rel} is not a regular file")
        try:
            target.resolve().relative_to(root)
        except ValueError:
            raise ComponentProblem(f"{rel} resolves outside the component folder") from None
        if info.st_size != size:
            raise ComponentProblem(f"{rel} is {info.st_size} bytes, the manifest says {size}")
        if (full or rel == component.entrypoint) and _sha256(target) != digest:
            raise ComponentProblem(f"{rel} does not match its recorded SHA-256")
    if os.name != "nt" and not os.access(component.exe, os.X_OK):
        raise ComponentProblem(f"{component.entrypoint} is not executable")


def runtime_version() -> str | None:
    try:
        return RUNTIME_VERSION_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ----------------------------------------------------------------------------------------------- the GPU

def detect_backend(environ=None) -> tuple[str | None, str]:
    """(backend, how) for this machine: the engine family its GPU needs. Bounded, read-only probes of tools that are
    already installed; ``STRATA_ENGINE_BACKEND`` overrides (the tests and the support desk use it)."""
    environ = os.environ if environ is None else environ
    forced = environ.get(ENV_BACKEND, "").strip().lower()
    if forced:
        if forced not in BACKEND_MATRIX:
            raise ComponentProblem(f"{ENV_BACKEND}={forced!r} is not one of {', '.join(sorted(BACKEND_MATRIX))}")
        return forced, "environment"
    if host_platform().startswith("macos"):
        return "metal", "platform"
    probes = (("cuda", ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader")),
              ("hip", ("rocminfo",) if os.name != "nt" else ("hipInfo",)),
              ("sycl", ("sycl-ls",)))
    for backend, command in probes:
        tool = shutil.which(command[0])
        if not tool:
            continue
        try:
            done = subprocess.run([tool, *command[1:]], capture_output=True, timeout=8, text=True,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and done.stdout.strip():
            if backend == "hip" and "gfx" not in done.stdout:
                continue
            if backend == "sycl" and "level_zero" not in done.stdout.lower() and "gpu" not in done.stdout.lower():
                continue
            return backend, command[0]
    return None, "no GPU tool answered"


def cuda_archs_present(environ=None) -> list[str]:
    """``sm_XX`` for each NVIDIA card nvidia-smi reports; empty when it cannot say."""
    tool = shutil.which("nvidia-smi")
    if not tool:
        return []
    try:
        done = subprocess.run([tool, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                              timeout=8, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    found = []
    for line in done.stdout.split():
        match = re.fullmatch(r"(\d+)\.(\d+)", line.strip())
        if match:
            found.append(f"sm_{match.group(1)}{match.group(2)}")
    return found


# ----------------------------------------------------------------------------------------------- resolve

def scan(root: Path | None = None) -> tuple[list[EngineComponent], list[dict]]:
    """Every readable engine package in the folder, and the ones that were not (with the reason)."""
    folder = engine_dir(root)
    found, problems = [], []
    try:
        entries = sorted(item for item in folder.iterdir() if item.is_dir() and not item.name.startswith("."))
    except OSError:
        return [], []
    for entry in entries[:64]:
        try:
            found.append(read_manifest(entry))
        except ComponentProblem as exc:
            problems.append({"folder": str(entry), "problem": str(exc)})
    return found, problems


def resolve(root: Path | None = None, *, backend: str | None = None, target: str | None = None,
            full: bool = False, runtime: str | None = None) -> EngineComponent:
    """The engine to start on this machine, or `EngineMissing`. Never downloads, never creates the folder."""
    base = root or components_dir()
    target = target or host_platform()
    found, problems = scan(base)
    if not found and not problems:
        raise EngineMissing("the component folder holds no engine package", base)
    wanted = backend
    how = "requested"
    if wanted is None:
        wanted, how = detect_backend()
    if wanted is None:
        # no GPU tool answered: only a CPU engine can be started
        wanted = "cpu"
    usable, reasons = [], []
    archs_here = cuda_archs_present() if wanted == "cuda" else []
    have_runtime = runtime if runtime is not None else runtime_version()
    for item in found:
        label = f"{item.version} {item.platform}-{item.backend}"
        if item.platform != target:
            reasons.append(f"{label} is for {item.platform}, this PC is {target}")
            continue
        ok, why = backend_supported(item.backend, target)
        if not ok:
            reasons.append(f"{label}: {why}")
            continue
        if item.backend != wanted:
            reasons.append(f"{label} is a {item.backend} engine, this PC needs {wanted} ({how})")
            continue
        if item.min_runtime and have_runtime and _version_key(have_runtime) < _version_key(item.min_runtime):
            reasons.append(f"{label} needs control {item.min_runtime} or newer, this is {have_runtime}")
            continue
        if item.cuda_archs and archs_here and not set(archs_here) & set(item.cuda_archs):
            reasons.append(f"{label} has code for {', '.join(item.cuda_archs)}, this PC's GPU is "
                           f"{', '.join(archs_here)}")
            continue
        try:
            verify(item, full=full)
        except ComponentProblem as exc:
            reasons.append(f"{label}: {exc}")
            continue
        usable.append(item)
    if not usable:
        raise EngineMissing("; ".join(reasons + [p["folder"] + ": " + p["problem"] for p in problems])
                            or "nothing matched", base, problems=problems)
    return max(usable, key=lambda item: _version_key(item.version))


#: A run config can name its engine as "component:strata-engine" instead of a path: the server then asks the
#: component folder for the engine that fits this PC when it starts (a config written before an engine update keeps
#: working after it, and a config copied to another PC finds that PC's engine).
REFERENCE_PREFIX = "component:"


def is_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith(REFERENCE_PREFIX)


def expand(value: str, cfg: dict | None = None, *, root: Path | None = None) -> str:
    """The engine path for `value`: unchanged when it is a path, resolved when it is a component reference."""
    if not is_reference(value):
        return value
    name = value[len(REFERENCE_PREFIX):]
    if name not in ("", COMPONENT_ID):
        raise EngineMissing(f"the run config names an unknown component {name!r}", root or components_dir())
    backend = (cfg or {}).get("backend")
    return str(resolve(root, backend=backend if backend in BACKEND_MATRIX else None).exe)


def status(root: Path | None = None, *, backend: str | None = None, full: bool = False) -> dict:
    """What `strata components engine` and the app's host report. Always answers; never raises for a missing engine."""
    base = root or components_dir()
    found, problems = scan(base)
    try:
        detected, how = detect_backend()
    except ComponentProblem as exc:
        detected, how = None, str(exc)
    report = {"schema_version": 1, "components_dir": str(base), "platform": host_platform(),
              "backend": backend or detected, "backend_source": "requested" if backend else how,
              "installed": [item.to_dict() for item in found], "problems": problems,
              "engine": None, "usable": False, "notice": None, "acquires": False}
    try:
        chosen = resolve(base, backend=backend, full=full)
        report["engine"] = chosen.to_dict()
        report["usable"] = True
    except EngineMissing as exc:
        report["notice"] = str(exc)
    return report
