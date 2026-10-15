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

Manifest (``engine-component.json``, schema 2; schema 1 packages still load). packaging/README.md has the full table::

    {"schema_version": 2, "id": "strata-engine", "name": "strata-engine", "version": "0.1.45",
     "platform": "linux-x86_64", "backend": "cuda", "archs": ["sm_86", "compute_89"],
     "min_driver": {"nvidia": "580.65"}, "build_id": "be885026f9-cuda13.0", "min_runtime": "0.1.45",
     "entrypoint": "strata", "files": [{"path": "strata", "sha256": "...", "bytes": 1}, ...],
     "signature": null}                      # S6 fills it; verify_signature() is the hook

Checks, in order, and each one names what failed:

* the manifest parses strictly; file paths are relative, plain, and stay inside the component folder;
* the platform and backend are combinations the project commits to (`components.BACKEND_MATRIX`);
* the backend is the one this machine's GPU needs (detected, or asked for);
* the signature (none yet; a verifier registered by S6 is called, and `STRATA_REQUIRE_SIGNED_COMPONENTS=1` refuses
  an unsigned or unverifiable package);
* every listed file exists with the listed size, and every file's SHA-256 matches (`verify(full=True)`, which is what
  a start does; the quick listing/status path hashes only the entrypoint);
* the runtime that is asking is new enough for `min_runtime`, and the GPU driver (when it can be read) for
  `min_driver`. A package that fails an integrity check is `EngineIntegrityError`: nothing is started.

Nothing here imports anything outside the standard library and `strata_control`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from strata_control import acquisition
from strata_control.targets import BACKEND_MATRIX, backend_supported, host_platform

SCHEMA_VERSION = 2
READABLE_SCHEMAS = (1, 2)           # schema 1 (S3's first packages: no archs/min_driver/build_id/signature) still loads
MANIFEST = "engine-component.json"
COMPONENT_ID = "strata-engine"
ENV_DIR = "STRATA_COMPONENTS_DIR"
ENV_BACKEND = "STRATA_ENGINE_BACKEND"
ENV_REQUIRE_SIGNED = "STRATA_REQUIRE_SIGNED_COMPONENTS"

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


class EngineIntegrityError(EngineMissing):
    """A package is installed but is not what its manifest says (a changed or missing file, a bad signature). Nothing
    was started. The message names the file; the fix is to repair or reinstall the component, never to run it anyway."""

    code = "engine_component_integrity"

    def __init__(self, reason: str, root: Path | None = None, *, problems: list[dict] | None = None):
        self.reason = reason
        self.root = root
        self.problems = problems or []
        RuntimeError.__init__(
            self,
            f"The engine package failed its integrity check, so it was not started: {reason}. Repair or reinstall "
            f"the engine component from the Terminal (`strata install`); the app never downloads one while it runs.")
        self.what = "the engine component"
        self.command = "strata install"


class ComponentProblem(ValueError):
    """One engine package is unusable. The message says why; the caller decides what to do next."""


class IntegrityProblem(ComponentProblem):
    """The package's files or signature do not match its manifest (as opposed to a package for another PC)."""


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
    archs: tuple[str, ...] = ()
    notes: str = ""
    min_driver: dict = field(default_factory=dict)       # {"nvidia": "580.65"}: the oldest driver the build runs on
    build_id: str = ""                                    # which engine build this is (source commit + toolkit)
    variant: str = ""                                     # optional tag when one backend has several builds (cuda13, cuda12)
    signature: dict | None = None                         # S6: {"alg", "key_id", "value"} over signing_payload()
    raw: dict = field(default_factory=dict, repr=False)   # the manifest as read, for the signature check

    @property
    def exe(self) -> Path:
        return self.root / self.entrypoint

    @property
    def cuda_archs(self) -> tuple[str, ...]:
        return tuple(item for item in self.archs if item.startswith(("sm_", "compute_")))

    @property
    def hip_archs(self) -> tuple[str, ...]:
        return tuple(item for item in self.archs if item.startswith("gfx"))

    def to_dict(self) -> dict:
        return {"id": self.id, "version": self.version, "platform": self.platform, "backend": self.backend,
                "entrypoint": self.entrypoint, "exe": str(self.exe), "root": str(self.root),
                "min_runtime": self.min_runtime, "archs": list(self.archs), "min_driver": dict(self.min_driver),
                "build_id": self.build_id, "variant": self.variant, "signed": bool(self.signature), "file_count": len(self.files), "bytes": sum(item[2] for item in self.files)}


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
    if not isinstance(data, dict) or data.get("schema_version") not in READABLE_SCHEMAS:
        raise ComponentProblem(f"{MANIFEST} has an unsupported schema_version")
    schema = data["schema_version"]
    if data.get("id") != COMPONENT_ID or data.get("name", COMPONENT_ID) != COMPONENT_ID:
        raise ComponentProblem(f"{MANIFEST} is not an engine component (id {data.get('id')!r})")
    for key in ("version", "platform", "backend", "entrypoint"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ComponentProblem(f"{MANIFEST} has no {key}")
    if schema >= 2 and (not isinstance(data.get("build_id"), str) or not data["build_id"].strip()):
        raise ComponentProblem(f"{MANIFEST} has no build_id")
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
    if len({item[0] for item in parsed}) != len(parsed):
        raise ComponentProblem(f"{MANIFEST} lists a file twice")
    archs = data.get("archs", data.get("cuda_archs")) or []        # schema 1 called the list cuda_archs
    if not isinstance(archs, list) or not all(isinstance(a, str) and a for a in archs):
        raise ComponentProblem("archs must be a list of strings")
    if schema >= 2 and data["backend"] != "cpu" and not archs:
        raise ComponentProblem(f"{MANIFEST} lists no archs for a {data['backend']} engine")
    driver = data.get("min_driver") or {}
    if not isinstance(driver, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in driver.items()):
        raise ComponentProblem("min_driver must map a vendor to a version string")
    signature = data.get("signature")
    if signature is not None and not (isinstance(signature, dict) and isinstance(signature.get("alg"), str)
                                      and isinstance(signature.get("value"), str)):
        raise ComponentProblem("signature must be null or {alg, key_id, value}")
    return EngineComponent(
        id=data["id"], version=data["version"], platform=data["platform"], backend=data["backend"],
        entrypoint=entry, files=tuple(parsed), root=folder, min_runtime=data.get("min_runtime"),
        archs=tuple(archs), notes=str(data.get("notes") or ""), min_driver=dict(driver),
        build_id=str(data.get("build_id") or ""), variant=str(data.get("variant") or ""), signature=signature,
        raw=data)


# ----------------------------------------------------------------------------------------------- signature (S6)

#: A signature verifier is ``fn(payload: bytes, signature: dict, component) -> None``; it raises
#: ``IntegrityProblem`` (or any exception) when the signature does not hold. S6 registers one that checks the project's
#: key; until then there is none, unsigned packages load (unless ``STRATA_REQUIRE_SIGNED_COMPONENTS=1``), and a package
#: that carries a signature nobody can check is reported as "signature not verified", never as verified.
_verifier: Callable[[bytes, dict, "EngineComponent"], None] | None = None


def register_signature_verifier(fn: Callable[[bytes, dict, "EngineComponent"], None] | None) -> None:
    global _verifier
    _verifier = fn


def signing_payload(manifest: dict) -> bytes:
    """The bytes a signature covers: the manifest without its ``signature`` member, as canonical JSON (sorted keys,
    no spaces, UTF-8). The file hashes are inside it, so one signature covers every file."""
    body = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_signature(component: "EngineComponent", environ=None) -> str:
    """"verified", "unsigned" or "unverified" (signed, no verifier yet). Raises IntegrityProblem for a bad signature,
    and for an unsigned or unverifiable package when ``STRATA_REQUIRE_SIGNED_COMPONENTS`` is set."""
    environ = os.environ if environ is None else environ
    required = environ.get(ENV_REQUIRE_SIGNED, "").strip().lower() in ("1", "true", "yes", "on")
    if not component.signature:
        if required:
            raise IntegrityProblem("the package is not signed and this PC requires signed components")
        return "unsigned"
    if _verifier is None:
        if required:
            raise IntegrityProblem("the package is signed but this build cannot check signatures yet")
        return "unverified"
    try:
        _verifier(signing_payload(component.raw), component.signature, component)
    except IntegrityProblem:
        raise
    except Exception as exc:
        raise IntegrityProblem(f"the signature does not verify ({type(exc).__name__}: {exc})") from None
    return "verified"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(component: EngineComponent, *, full: bool = False) -> None:
    """Raise IntegrityProblem unless the files are what the manifest says. `full` hashes every file; without it only
    the entrypoint is hashed (sizes are always checked)."""
    root = component.root.resolve()
    for rel, digest, size in component.files:
        target = component.root / rel
        try:
            info = target.lstat()
        except OSError:
            raise IntegrityProblem(f"{rel} is missing") from None
        if target.is_symlink() or not target.is_file():
            raise IntegrityProblem(f"{rel} is not a regular file")
        try:
            target.resolve().relative_to(root)
        except ValueError:
            raise IntegrityProblem(f"{rel} resolves outside the component folder") from None
        if info.st_size != size:
            raise IntegrityProblem(f"{rel} is {info.st_size} bytes, the manifest says {size}")
        if (full or rel == component.entrypoint) and _sha256(target) != digest:
            raise IntegrityProblem(f"{rel} does not match its recorded SHA-256")
    if os.name != "nt" and not os.access(component.exe, os.X_OK):
        raise IntegrityProblem(f"{component.entrypoint} is not executable")


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
    found = _drm_vendor_backend()
    if found:
        return found, "sysfs pci vendor"
    return None, "no GPU tool answered"


def _drm_vendor_backend() -> str | None:
    """Linux without the vendor tools on PATH (a HIP box that runs ROCm from Python wheels has no rocminfo): the PCI
    vendor of the display adapters decides. NVIDIA 0x10de -> cuda, AMD 0x1002 -> hip, Intel 0x8086 -> sycl, in that
    order. Read-only sysfs, nothing is run."""
    if not os.path.isdir("/sys/class/drm"):
        return None
    vendors = set()
    try:
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
            if "-" in card.name:
                continue
            try:
                vendors.add((card / "device" / "vendor").read_text().strip().lower())
            except OSError:
                continue
    except OSError:
        return None
    for vendor, backend in (("0x10de", "cuda"), ("0x1002", "hip"), ("0x8086", "sycl")):
        if vendor in vendors:
            return backend
    return None


def cuda_archs_present(environ=None) -> list[str]:
    """``sm_XX`` for each NVIDIA card nvidia-smi reports; empty when it cannot say."""
    found = []
    for line in _smi_query("compute_cap"):
        match = re.fullmatch(r"(\d+)\.(\d+)", line.strip())
        if match:
            found.append(f"sm_{match.group(1)}{match.group(2)}")
    return found


def hip_archs_present(kfd_root: str = "/sys/class/kfd/kfd/topology/nodes") -> list[str]:
    """``gfxNNNN`` for each AMD GPU the Linux kernel driver lists (``gfx_target_version`` is MMmmss: gfx1201 = 120001,
    gfx1103 = 110003, gfx90a = 90010); empty when it cannot say (Windows, no amdgpu/kfd, no permission)."""
    found = []
    try:
        nodes = sorted(Path(kfd_root).iterdir())
    except OSError:
        return []
    for node in nodes:
        try:
            for line in (node / "properties").read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(" ")
                if key == "gfx_target_version" and value.strip().isdigit() and int(value) > 0:
                    number = int(value)
                    found.append(f"gfx{number // 10000}{number // 100 % 100:x}{number % 100:x}")
        except OSError:
            continue
    return found


def _smi_query(field_name: str) -> list[str]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return []
    try:
        done = subprocess.run([tool, f"--query-gpu={field_name}", "--format=csv,noheader"], capture_output=True,
                              timeout=8, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in done.stdout.splitlines() if line.strip()] if done.returncode == 0 else []


def nvidia_driver_version() -> str | None:
    """The NVIDIA driver version nvidia-smi reports (the oldest one when several cards disagree); None when unreadable."""
    versions = _smi_query("driver_version")
    return min(versions, key=_version_key) if versions else None


def arch_covers(archs, present: list[str]) -> bool:
    """Whether a build with `archs` (``sm_86`` = machine code for that card, ``compute_89`` = PTX that any card of
    that generation or newer can JIT) can run on at least one of the `present` cards (``sm_XX``)."""
    cards = {int(item[3:]) for item in present if item.startswith("sm_") and item[3:].isdigit()}
    if not cards:
        return True              # cannot tell: not a refusal
    for arch in archs:
        kind, _, number = arch.partition("_")
        if not number.isdigit():
            continue
        if kind == "sm" and int(number) in cards:
            return True
        if kind == "compute" and any(card >= int(number) for card in cards):
            return True
    return False


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
    """The engine to start on this machine, or `EngineMissing` / `EngineIntegrityError`. Never downloads, never creates
    the folder. `full=True` hashes every file (a start does); the default is the quick check the listings use."""
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
    usable, reasons, broken = [], [], []
    archs_here = cuda_archs_present() if wanted == "cuda" else []
    driver_here = nvidia_driver_version() if wanted == "cuda" else None
    hip_here = hip_archs_present() if wanted == "hip" else []
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
        if item.cuda_archs and archs_here and not arch_covers(item.cuda_archs, archs_here):
            reasons.append(f"{label} has code for {', '.join(item.cuda_archs)}, this PC's GPU is "
                           f"{', '.join(archs_here)}")
            continue
        if item.hip_archs and hip_here and not set(item.hip_archs) & set(hip_here):
            reasons.append(f"{label} has code for {', '.join(item.hip_archs)}, this PC's GPU is "
                           f"{', '.join(hip_here)}")
            continue
        need = item.min_driver.get("nvidia")
        if need and driver_here and _version_key(driver_here) < _version_key(need):
            reasons.append(f"{label} needs NVIDIA driver {need} or newer, this PC has {driver_here}")
            continue
        try:
            verify_signature(item)
            verify(item, full=full)
        except ComponentProblem as exc:
            reasons.append(f"{label}: {exc}")
            broken.append(f"{label}: {exc}")
            continue
        usable.append(item)
    if not usable:
        everything = reasons + [p["folder"] + ": " + p["problem"] for p in problems]
        if broken and len(broken) == len(reasons):       # every package that fits this PC failed its integrity check
            raise EngineIntegrityError("; ".join(broken), base, problems=problems)
        raise EngineMissing("; ".join(everything) or "nothing matched", base, problems=problems)
    # newest version first; with equal versions the build for the newer driver (the CUDA 13 build beside the CUDA 12 one)
    return max(usable, key=lambda item: (_version_key(item.version), _version_key(item.min_driver.get("nvidia", "0"))))


#: A run config can name its engine as "component:strata-engine" instead of a path: the server then asks the
#: component folder for the engine that fits this PC when it starts (a config written before an engine update keeps
#: working after it, and a config copied to another PC finds that PC's engine).
REFERENCE_PREFIX = "component:"


def is_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith(REFERENCE_PREFIX)


def expand(value: str, cfg: dict | None = None, *, root: Path | None = None, full: bool = True) -> str:
    """The engine path for `value`: unchanged when it is a path, resolved when it is a component reference. This is the
    start path, so every file's SHA-256 is checked (`full`): a changed file refuses the start instead of running."""
    if not is_reference(value):
        return value
    name = value[len(REFERENCE_PREFIX):]
    if name not in ("", COMPONENT_ID):
        raise EngineMissing(f"the run config names an unknown component {name!r}", root or components_dir())
    backend = (cfg or {}).get("backend")
    return str(resolve(root, backend=backend if backend in BACKEND_MATRIX else None, full=full).exe)


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
