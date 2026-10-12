"""setup.py on a Mac: Apple Silicon, the Metal engine (metal/), built from source on the Mac.

    ./setup.sh [options]          (setup.py hands a macOS run to this file; the PC paths are not used)

What it does, each step skipped when it is already done:

  1. checks the Mac: Apple Silicon, the chip, unified memory, macOS version, the Command Line Tools, free disk
  2. recommends a model by memory (recommend, never force: an explicit --model is kept, with what it risks)
  3. installs the Python packages into .venv (cmake and ninja come from pip: no Homebrew, no sudo)
  4. builds the Metal engine from source: metal-cpp (Apple's headers) and llama.cpp's ggml at the pinned commit
  5. downloads the model (setup.py's downloader: resumable, the pinned revision)
  6. prepares the model: the native pack (tools/iq_pack.py) and the MTP draft layer (tools/mtp_fetch.py, mtp_pack.py)
  7. writes strata-<model>.json and run-<model>.sh with the Metal defaults (--spec 2, the Q8_0 hyper-connection
     read) and starts the server (serve/server.py), the same as on a PC

Everything shared - the model table, the downloader, the config and run-script writers, the server - is setup.py's.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

# Apple's metal-cpp headers for the macOS 26 SDK (https://developer.apple.com/metal/cpp/), pinned by SHA-256
METAL_CPP_URL = "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip"
METAL_CPP_SHA256 = "4df3c078b9aadcb516212e9cb03004cbc5ce9a3e9c068fa3144d021db585a3a4"
BUILD_DIR = ROOT / "build-metal"
EXE = BUILD_DIR / "strata"
MIN_MACOS = (15, 0)        # MTLResidencySet (the engine keeps every buffer resident through one)

# The Mac's packs.  shard1_gb: the file the GPU maps (resident in unified memory while the model runs); the n-gram
# table (shard 2, 28.8 GB) is read from the SSD a few rows at a time.  tested: measured on the M5 Max (docs/MAC.md).
MAC_MODELS = {
    "Q2_0": {"shard1_gb": 37.6, "tested": True},
    "IQ3_XXS": {"shard1_gb": 47.0, "tested": True},
    # its gate/up experts are IQ4_XS: the Metal expert kernels have no IQ4_XS variant yet (the engine stops at load)
    "IQ3_S": {"shard1_gb": 54.8, "tested": False, "missing": "the Metal expert kernels for its IQ4_XS experts"},
}
# beside the mapped file: the prompt kernels' buffers (chunks of 2,048 rows), the KV cache, the draft layer, the
# Q8_0 copies of the hyper-connection matrices and the macOS desktop
OVERHEAD_GB = 8.0
# the share of the unified memory Metal lets one process keep resident (recommendedMaxWorkingSetSize): 107.5 of 128 GB
# measured on the M5 Max; 3/4 is the conservative rule for the other sizes
GPU_SHARE = 0.75
CONTEXTS = [8192, 32768, 65536, 131072]
TESTED_CONTEXT = 32768     # needles and the KL harness ran to 32K on the Mac


# ------------------------------------------------------------------------------------------------ the Mac
def _sysctl(name: str, run=subprocess.run) -> str:
    try:
        return run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def mac_info(run=subprocess.run) -> dict:
    """The chip, its unified memory, macOS, the architecture and the Command Line Tools."""
    chip = _sysctl("machdep.cpu.brand_string", run) or "unknown chip"
    try:
        ram = int(_sysctl("hw.memsize", run) or 0) / 2 ** 30   # GB as Apple sells them (128 GB = 2^37 bytes)
    except ValueError:
        ram = 0.0
    try:
        ver = run(["sw_vers", "-productVersion"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        ver = ""
    try:
        clt = run(["xcode-select", "-p"], capture_output=True, text=True, timeout=10)
        clt_path = clt.stdout.strip() if clt.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        clt_path = ""
    return {"chip": chip, "ram_gb": ram, "macos": ver, "arm64": platform.machine() == "arm64",
            "clt": clt_path, "perf_cores": int(_sysctl("hw.perflevel0.physicalcpu", run) or 0)}


def macos_tuple(ver: str) -> tuple:
    try:
        return tuple(int(x) for x in ver.split(".")[:2])
    except ValueError:
        return (0, 0)


def mac_problems(info: dict) -> list[str]:
    """What stops the setup on this Mac (each with what to do)."""
    p = []
    if not info["arm64"]:
        p.append(f"this Mac has an Intel CPU ({info['chip']}): the Metal engine needs Apple Silicon (M1 or newer)")
    if macos_tuple(info["macos"]) < MIN_MACOS:
        p.append(f"macOS {info['macos'] or '?'}: the Metal engine needs macOS {MIN_MACOS[0]} or newer (System Settings "
                 "> General > Software Update)")
    if not info["clt"]:
        p.append("the Command Line Tools are missing (the compiler the engine is built with): run "
                 "`xcode-select --install`, then this setup again")
    return p


def gpu_budget_gb(ram_gb: float) -> float:
    return ram_gb * GPU_SHARE


def model_need_gb(model: str) -> float:
    return MAC_MODELS[model]["shard1_gb"] + OVERHEAD_GB


def recommend_model(ram_gb: float) -> tuple[str, list[str]]:
    """The recommended pack and the notes that go with it.  The best quality whose mapped file fits Metal's share of
    the memory, preferring the measured packs; nothing fits: Q2_0, with the SSD-streaming note."""
    budget = gpu_budget_gb(ram_gb)
    fits = [m for m in ("IQ3_XXS", "Q2_0") if model_need_gb(m) <= budget]
    if MAC_MODELS["IQ3_S"]["tested"] and ram_gb >= 128 and model_need_gb("IQ3_S") <= budget:
        fits.insert(0, "IQ3_S")
    if fits:
        return fits[0], []
    return "Q2_0", [f"{ram_gb:.0f} GB of memory: no pack fits Metal's share of it ({budget:.0f} GB) with the model "
                    "mapped whole. The SSD-streaming mode for 32-48 GB Macs is not in this build yet (docs/MAC.md); "
                    "Q2_0 may still start, and pages from the SSD while it runs (much slower)"]


def model_note(model: str, ram_gb: float) -> list[str]:
    """What a chosen pack risks on this Mac (said, never refused)."""
    notes = []
    if model_need_gb(model) > gpu_budget_gb(ram_gb):
        notes.append(f"{model} maps {MAC_MODELS[model]['shard1_gb']:.0f} GB and needs ~{model_need_gb(model):.0f} GB of "
                     f"the {gpu_budget_gb(ram_gb):.0f} GB Metal keeps resident on this Mac: kept as you chose; it may page "
                     "from the SSD or fail to load")
    if MAC_MODELS[model].get("missing"):
        notes.append(f"{model} does not run on this Metal engine yet ({MAC_MODELS[model]['missing']} are missing): the "
                     "engine will stop when it loads; kept as you chose (for testing a newer engine)")
    elif not MAC_MODELS[model]["tested"]:
        notes.append(f"{model} has not been measured on a Mac yet")
    return notes


def recommend_context(ram_gb: float) -> int:
    return 32768 if ram_gb < 96 else 65536


# ------------------------------------------------------------------------------------------------ the engine
def metal_cpp() -> Path:
    """Apple's metal-cpp headers in third_party/metal-cpp (one zip, pinned)."""
    dst = ROOT / "third_party" / "metal-cpp"
    if (dst / "Metal" / "Metal.hpp").exists():
        return dst
    z = ROOT / "third_party" / "metal-cpp_26.zip"
    S.download(METAL_CPP_URL, z, "metal-cpp (Apple's Metal headers)")
    S.verify_sha256(z, z.stat().st_size, METAL_CPP_SHA256)
    tmp = ROOT / "third_party" / "_metal_cpp"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(tmp)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(tmp / "metal-cpp"), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)
    return dst


def tool(name: str) -> str:
    """cmake / ninja from the venv's pip packages (no Homebrew)."""
    p = Path(sys.executable).parent / name
    return str(p) if p.exists() else (shutil.which(name) or name)


def engine_current() -> bool:
    try:
        meta = json.loads((BUILD_DIR / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return EXE.exists() and meta.get("version") == S.source_version() and meta.get("commit") == source_commit()


def source_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_engine(llama: Path, jobs: int | None = None) -> Path:
    """The Metal engine from source: cmake + ninja (pip's), Apple's clang from the Command Line Tools."""
    if engine_current():
        S.ok(f"Metal engine: {EXE} (built for this source)")
        return EXE
    mcpp = metal_cpp()
    S.say("  Building the Metal engine from source (5-15 minutes the first time) ...")
    BUILD_DIR.mkdir(exist_ok=True)
    log = BUILD_DIR / "build.log"
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    cfg = [tool("cmake"), "-S", str(ROOT / "metal"), "-B", str(BUILD_DIR), "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
           f"-DCMAKE_MAKE_PROGRAM={tool('ninja')}", f"-DMETAL_CPP_DIR={mcpp}", f"-DSTRATA_GGML_DIR={llama}"]
    build = [tool("ninja"), "-C", str(BUILD_DIR), "strata"] + (["-j", str(jobs)] if jobs else [])
    with open(log, "w", encoding="utf-8") as f:
        for cmd in (cfg, build):
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
            if rc != 0:
                S.fail(f"the Metal engine did not build (see {log})",
                       "the last lines of the log name the problem; the Command Line Tools must be up to date "
                       "(softwareupdate --list)")
    (BUILD_DIR / "BUILD.json").write_text(json.dumps({"version": S.source_version(), "commit": source_commit(),
                                                      "backend": "metal"}, indent=1), encoding="utf-8")
    S.ok(f"Metal engine built: {EXE}")
    return EXE


# ------------------------------------------------------------------------------------------------ the config
def engine_args(pack: Path, shards: list, rt: Path, ctx: int) -> list[str]:
    """The Metal defaults: --spec 2 (measured best on the M5 Max, docs/MAC.md) and the Q8_0 hyper-connection read
    (the engine's default on Metal; STRATA_HC_Q8=0 in the config's env gives the exact BF16 read)."""
    return ["--pack", str(pack), "--native", str(shards[0]), "--ple-gguf", str(shards[1]),
            "--expert-profile", str(ROOT / "data" / "expert-profile.bin"), "--expert-cache", "64",
            "--prefill", "auto", "--spec", "2", "--mtp", str(rt), "--max-context", str(ctx)]


def mac_config(exe: Path, args: list, pack: Path, model_name: str, tag: str, port: int) -> dict:
    return {"exe": str(exe), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
            "model_name": model_name, "log": str(ROOT / f"strata-{tag.lower()}.log"), "lib_dirs": [], "port": port,
            "backend": "metal"}


def mac_configs() -> list[Path]:
    out = []
    for p in S.installed_configs():
        try:
            if json.loads(p.read_text(encoding="utf-8-sig")).get("backend") == "metal":
                out.append(p)
        except (OSError, ValueError):
            pass
    return out


def start(cfg_path: Path, port: int | None, open_browser: bool = True) -> int:
    """serve/server.py with this config (setup.py's start without the NVIDIA / AMD card checks)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    missing = [p for p in [cfg["exe"], *[a for a in cfg["args"] if a.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run ./setup.sh --setup to repair it")
    cfg_path.touch()
    if "--mtp" in cfg["args"][:-1]:
        S.refresh_draft_vocab(Path(cfg["args"][cfg["args"].index("--mtp") + 1]), cfg.get("draft_vocab", "cjk"))
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if open_browser and cfg.get("open_browser") is not False:
        cmd.append("--open")
    S.say()
    S.say(f"  Starting {cfg.get('model_name', 'the model')} on Metal: the engine maps the model file and is ready in about "
          "half a minute (longer the first time after a restart, while macOS reads the file).")
    S.say("  Closing this window stops the model.")
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", choices=["qwen"], default="qwen",
                    help="qwen = Qwen3.8-Flash-Next (the only family measured on a Mac so far)")
    ap.add_argument("--model", choices=list(MAC_MODELS))
    ap.add_argument("--context", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--api-key")
    ap.add_argument("--draft-vocab", choices=list(S.DRAFT_VOCABS))
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None)
    ap.add_argument("--data-dir")
    ap.add_argument("--models-dir")
    ap.add_argument("--gguf-dir")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--no-start", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--build", action="store_true", help="build the engine again even when it is current")
    a, rest = ap.parse_known_args(argv)
    if rest:
        S.warn("not used on a Mac: " + " ".join(rest))
    S.say("Strata on a Mac (Apple Silicon, Metal)")
    data, elsewhere = S.data_folder(a.data_dir)
    roots = [data, *elsewhere]
    models_root = Path(a.models_dir) if a.models_dir else data / "models"

    have = mac_configs()
    if have and not (a.setup or a.model or a.check or a.no_start or a.update or a.build):
        return start(have[0], a.port, a.browser is not False)

    # ---- 1. the Mac
    S.step(1, "checking your Mac")
    info = mac_info()
    S.ok(f"{info['chip']}, {info['ram_gb']:.0f} GB unified memory, macOS {info['macos'] or '?'}")
    problems = mac_problems(info)
    for p in problems:
        S.warn(p)
    free = S.free_gb(data if data.exists() else ROOT)
    S.ok(f"free disk: {free:.0f} GB at {data}")
    if a.check:
        return 1 if problems else 0
    if problems:
        S.fail(problems[0])

    # ---- 2. the choices
    S.step(2, "your choices")
    rec, notes = recommend_model(info["ram_gb"])
    for n in notes:
        S.warn(n)
    if a.model:
        model = a.model
    else:
        names = [m for m in MAC_MODELS if not MAC_MODELS[m].get("missing")]   # --model offers the others
        for i, m in enumerate(names, 1):
            S.say(f"  {i}) {m:8} {S.MODELS[m]['about']}  (maps {MAC_MODELS[m]['shard1_gb']:.0f} GB"
                  + ("" if MAC_MODELS[m]["tested"] else ", not measured on a Mac") + ")"
                  + ("   <- recommended" if m == rec else ""))
        model = names[int(S.ask("Which size?", [str(i) for i in range(1, len(names) + 1)],
                                str(names.index(rec) + 1), a.yes)) - 1]
    for n in model_note(model, info["ram_gb"]):
        S.warn(n)
    ctx = a.context or recommend_context(info["ram_gb"])
    if ctx > TESTED_CONTEXT:
        S.say(f"  context {ctx}: measured on the Mac up to {TESTED_CONTEXT} (needles found at 8K-32K); longer is kept")
    S.ok(f"{model}, context {ctx}")
    fam = S.FAMILIES[a.family]
    tag = fam["tag"] + model
    need = S.MODELS[model]["download_gb"] + 6
    if free < need:
        S.warn(f"about {need:.0f} GB are needed on the disk ({free:.0f} GB free): kept as you chose; it may run out")

    # ---- 3. Python packages (cmake and ninja among them: the engine build's tools)
    S.step(3, "Python packages")
    S.pip_install(S.requirement_lines() if S.REQUIREMENTS.exists() else S.PY_PACKAGES, "numpy, jinja2, ..., cmake, ninja")

    # ---- 4. the engine
    S.step(4, "the Strata engine (Metal)")
    llama = S.get_llama_cpp()
    S.ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:7]} (ggml, gguf-py)")
    if a.build:
        (BUILD_DIR / "BUILD.json").unlink(missing_ok=True)
    exe = build_engine(llama)
    if a.update:
        return 0

    # ---- 5. the model
    S.step(5, f"downloading {fam['title']} {model}")
    models_dir = Path(a.gguf_dir) if a.gguf_dir else models_root / tag
    models_dir.mkdir(parents=True, exist_ok=True)
    shards = [models_dir / S.model_file(fam, model, i) for i in (1, 2)]
    for s in shards:
        if s.exists() and S.done(s):
            S.ok(f"{s.name} already downloaded")
            continue
        S.download(fam["hf"].format(q=model) + s.name, s)
    S.check_shards(shards)
    S.ok("model files present")

    # ---- 6. the pack and the MTP draft layer
    S.step(6, "preparing the model for Strata")
    pack = S.find_in(roots, f"packs/{tag.lower()}") or data / "packs" / tag.lower()
    env = dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py"))
    if not (pack / "native_experts.txt").exists() or not (pack / "tokenizer" / "vocab.json").exists():
        S.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack)], env=env)
    S.ok(f"model prepared: {pack}")
    mtp = (S.find_in(roots, "mtp/mtp-q2_0.gguf") or data / "mtp" / "mtp-q2_0.gguf").parent
    rt = mtp / "rt"
    if not (mtp / "mtp-q2_0.gguf").exists():
        S.say("  The MTP draft layer (speculative decoding) comes from the original Qwen checkpoint: only its ~5 GB of")
        S.say("  MTP tensors are downloaded.")
        S.run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)],
              env={**env, "STRATA_SOURCE": S.model_source()})
        S.run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
               "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
    rt.mkdir(parents=True, exist_ok=True)              # the Metal engine reads mtp-q2_0.gguf beside it (no rt pack)
    S.refresh_draft_vocab(rt, a.draft_vocab or "cjk")
    S.ok(f"MTP draft layer: {mtp / 'mtp-q2_0.gguf'}")

    # ---- 7. the config, the start script
    S.step(7, "writing the start script")
    port = a.port or 8080
    cfg = mac_config(exe, engine_args(pack, shards, rt, ctx), pack, f"{fam['name']}-{model.lower()}", tag, port)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.draft_vocab:
        cfg["draft_vocab"] = a.draft_vocab
    if a.browser is not None:
        cfg["open_browser"] = a.browser
    cfg_path = ROOT / f"strata-{tag.lower()}.json"
    S.write_config(cfg_path, cfg)
    script = S.write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    S.ok(f"start script: {script.name}")
    S.say()
    S.say("All set.")
    S.say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1")
    S.say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    S.say(f"  Next time:        just run ./setup.sh (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port, a.browser is not False)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
