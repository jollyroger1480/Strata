"""What `strata launch` does after the choices are made: get the engine and the model, write the run config, tune, start, open access.

Every step reports plain-word events through one callback (`emit`), so the TUI (progress screen), `--no-tui` (printed lines) and the
tests see the same thing. Nothing here prints or reads the keyboard. The steps reuse what is already in the Terminal:

* the engine comes from the signed feed (`strata update`, the one explicit step that downloads);
* the model comes from setup.py in the app's own installer mode (STRATA_CONTROL_INSTALL), the same job the desktop app runs;
* the run config the engine uses is a private copy of the installed one with this launch's choices applied; the profile
  `launch-<name>` (the shared profiles store) points at it, and carries host, port and parallel;
* tuning is `strata tune auto`, its result lives in the shared presets store and is applied at every start;
* start and stop are `strata runtime start|stop` (a background service: it keeps serving after this terminal closes).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from strata_cli.launch import access as acc
from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog, CatalogUnavailable

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = os.name == "nt"


class LaunchError(RuntimeError):
    """A step could not be done. `text` says what it means; `next` says what to do."""

    def __init__(self, text: str, next: str = "", code: int = 6):
        super().__init__((text + " " + next).strip())
        self.text, self.next, self.code = text, next, code


class Cancelled(LaunchError):
    def __init__(self):
        super().__init__("Stopped. Nothing was changed on your computer beyond files already downloaded.", "", 130)


@dataclass
class Event:
    stage: str                      # engine | model | config | tune | start | access | done
    text: str
    pct: float | None = None
    level: str = "info"             # info | warn | error
    data: dict = field(default_factory=dict)


@dataclass
class Result:
    profile: str
    urls: list[str]
    key: str | None
    tune_word: str | None
    notices: list[str]
    config: str
    tunnel: str | None = None


STAGES = ("engine", "model", "config", "tune", "start", "access")


def tune_words(result: dict | None, code: int | None = None) -> tuple[str, str]:
    """(short word, plain sentence) for a tuner result: faster / already fastest / busy PC / failed / stopped."""
    if not result:
        return "failed", "The tuner did not finish, so the standard settings run. Nothing was changed."
    out = result.get("outcome")
    if out == "improved":
        gain = result.get("gain_pct")
        return "faster", (f"Done: answers are about {gain:.0f}% faster with the tuned settings." if isinstance(gain, (int, float))
                          else "Done: the tuned settings are faster.")
    if out == "default_confirmed":
        return "already fastest", "Your computer already runs at its best with the standard settings. Nothing needed changing."
    if out == "inconclusive":
        return "busy PC", ("The computer was too busy to tell a real gain from noise, so the standard settings run. "
                           "Close other programs and tune again later (strata tune auto).")
    if out == "cancelled":
        return "stopped", "Tuning was stopped. Your settings were not changed."
    return "failed", (result.get("message") or "Tuning could not run here.") + " The standard settings run."


# ----------------------------------------------------------------------------------------------- run config

def read_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def desc_matches(desc: dict, s: S.LaunchSettings) -> bool:
    """Does the installed model's config already carry the context, memory format and picture choice of these settings?"""
    if s.context and desc.get("context") != s.context:
        return False
    want_kv = s.kv or ("int8" if (s.context or 32768) > 8192 else "fp16")
    if desc.get("kv") not in (None, want_kv) and (s.context or 32768) > 8192:
        return False
    images = desc.get("images") or "off"
    if (s.vision == "off") != (images == "off"):
        return False
    if s.vision in ("gpu", "cpu") and images != s.vision:
        return False
    return True


def files_present(cfg: dict) -> list[str]:
    """The model files a run config names that are missing (empty list: all there)."""
    missing = []
    for tok in cfg.get("args") or []:
        if isinstance(tok, str) and tok.lower().endswith(".gguf") and not Path(tok).exists():
            missing.append(tok)
    for flag in ("--pack", "--mtp"):
        args = [str(a) for a in cfg.get("args") or []]
        if flag in args and args.index(flag) + 1 < len(args) and not Path(args[args.index(flag) + 1]).exists():
            missing.append(args[args.index(flag) + 1])
    return missing


def derive_config(base: dict, s: S.LaunchSettings, *, bool_flags: frozenset[str], log: str | None = None) -> dict:
    """The private run config of this launch: the installed one plus the choices that are not setup's (host, port, key, advanced)."""
    from strata_control import tune_knobs
    cfg = json.loads(json.dumps(base))
    cfg["host"] = acc.host_for(s.access)
    cfg["port"] = int(s.port)
    if s.api_key:
        cfg["api_key"] = s.api_key
    else:
        cfg.pop("api_key", None)
    cfg.setdefault("exe", "component:strata-engine")
    if log:
        cfg["log"] = log
    if s.extra:
        a_changes, e_changes = S.as_argv_changes(s.extra, bool_flags)
        cfg["args"] = tune_knobs.apply_to_argv([str(x) for x in cfg.get("args") or []], a_changes)
        if e_changes:
            cfg["env"] = tune_knobs.apply_to_env(dict(cfg.get("env") or {}), e_changes)
    return cfg


def write_private(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".launch-cfg-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write((json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    if WINDOWS:
        try:
            from strata_control import private_files
            private_files.restrict_to_current_user(path, directory=False)
        except Exception:                                    # noqa: BLE001
            pass


def launch_dir() -> Path:
    from strata_control import paths
    d = paths.state_base() / "launch"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------------------------------- the runner

class Runner:
    def __init__(self, s: S.LaunchSettings, *, catalog: Catalog | None = None, emit: Callable[[Event], None] | None = None,
                 cancel: threading.Event | None = None, feed: str | None = None, python: str | None = None,
                 components_dir: str | None = None, hardware: dict | None = None):
        self.s = s
        self._catalog = catalog
        self.emit_cb = emit or (lambda e: None)
        self.cancel = cancel or threading.Event()
        self.feed = feed
        self.python = python or sys.executable
        self.components_dir = components_dir
        self.hardware_override = hardware
        self.children: list[subprocess.Popen] = []
        self.notices: list[str] = []
        self.tune_result: dict | None = None
        self.config_path: Path | None = None

    # ---- small helpers
    def emit(self, stage: str, text: str, pct: float | None = None, level: str = "info", **data: Any) -> None:
        self.emit_cb(Event(stage, text, pct, level, data))

    def check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled()

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            try:
                self._catalog = Catalog(hardware=self.hardware_override)
            except CatalogUnavailable as e:
                raise LaunchError(str(e), "", 5) from None
        return self._catalog

    @property
    def name(self) -> str:
        return S.slug(self.s.name or (self.s.model or "model") + "-" + S.context_label(self.s.context).lower())

    @property
    def profile(self) -> str:
        return "launch-" + self.name

    def _env(self) -> dict[str, str]:
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        env.pop("STRATA_NO_ACQUIRE", None)                  # launch is the explicit step that may download
        if self.components_dir:
            env["STRATA_COMPONENTS_DIR"] = self.components_dir
        return env

    def _spawn(self, argv: list[str], *, env: dict | None = None, cwd: str | None = None) -> subprocess.Popen:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env or self._env(), cwd=cwd, bufsize=1, **kw)
        self.children.append(p)
        return p

    def _stop_children(self) -> None:
        """End only the processes this runner started (by the handles it holds)."""
        for p in self.children:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=15)
                except Exception:                            # noqa: BLE001
                    try:
                        p.kill()
                    except Exception:                        # noqa: BLE001
                        pass

    def _cli(self, *args: str) -> list[str]:
        return [self.python, str(ROOT / "strata.py"), *args]

    # ---- 1. the engine
    def ensure_engine(self) -> None:
        self.emit("engine", "Checking the engine for this computer…")
        from strata_control import engine_component as ec
        root = Path(self.components_dir) if self.components_dir else None
        report = ec.status(root)
        if report.get("usable"):
            eng = report["engine"]
            self.emit("engine", f"Engine {eng['version']} ({eng['backend']}) is installed.", 100)
            return
        feed = self.feed or os.environ.get("STRATA_UPDATE_FEED")
        self.emit("engine", "No engine is installed yet. Getting the one for your graphics card…", 0)
        argv = self._cli("update", "--no-control", "--output", "jsonl")
        if feed:
            argv += ["--feed", feed]
        if self.components_dir:
            argv += ["--components-dir", self.components_dir]
        p = self._spawn(argv)
        last = None
        tail: list[str] = []
        for line in p.stdout:                                # type: ignore[union-attr]
            self.check_cancel()
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                tail.append(line)
                continue
            if ev.get("event") == "progress":
                self.emit("engine", f"Downloading the engine… {ev.get('percent', 0)}%", float(ev.get("percent", 0)))
            elif ev.get("event") in ("phase", "waiting"):
                self.emit("engine", str(ev.get("message") or ""))
            last = ev
        code = p.wait()
        if code != 0:
            why = ""
            if isinstance(last, dict):
                why = str((last.get("error") or {}).get("message") or last.get("message") or "")
            why = why or (tail[-1] if tail else "")
            raise LaunchError("The engine could not be installed." + (f" {why}" if why else ""),
                              "Check your internet connection, or point Strata at an update source with --feed, then try again.", 6)
        self.emit("engine", "The engine is installed.", 100)

    # ---- 2. the model
    def setup_args(self) -> list[str]:
        cat, s = self.catalog, self.s
        fam, model = s.family, s.model
        args = ["--yes", "--no-start", "--family", str(fam), "--model", str(model)]
        ctx = s.context
        if ctx:
            args += ["--context", str(ctx)]
        args += ["--vision", {"off": "no", "on": "yes"}.get(s.vision, s.vision)]
        if s.kv:
            args += ["--kv", s.kv]
        if fam in ("qwen", "coder"):
            args += ["--experimental-speed-projection", "off"]
        data = s.data_dir
        if data:
            args += ["--data-dir", str(data)]
        if s.model_path:
            p = Path(s.model_path)
            args += ["--gguf-dir", str(p if p.is_dir() else p.parent)]
        if s.gpus:
            args += ["--gpus", ",".join(str(g) for g in s.gpus)] if len(s.gpus) > 1 else ["--gpu", str(s.gpus[0])]
        return args

    def installed_state(self) -> tuple[Path | None, dict | None]:
        tag = self.catalog.tag_of(self.s.family, self.s.model)
        path = self.catalog.installed_config(tag)
        if path is None:
            return None, None
        desc = self.catalog.strata.describe_config(path)
        return path, desc

    def ensure_model(self) -> Path:
        s, cat = self.s, self.catalog
        if s.model_path and not (s.family and s.model):
            self.resolve_own_files()
        if not (s.family and s.model):
            raise LaunchError("No model was chosen.", "Choose a model, or use --model NAME (strata launch --help lists them).", 2)
        path, desc = self.installed_state()
        if path is not None and desc is not None and desc_matches(desc, s) and not files_present(read_config(path)):
            self.emit("model", f"{s.family} {s.model} is already installed with these choices.", 100)
            return path
        title = f"{s.family} {s.model}"
        self.emit("model", f"Getting {title} ready (a download may be large; it continues where it stopped if interrupted)…", 0)
        argv = [self.python, str(cat.src / "setup.py"), *self.setup_args()]
        env = self._env()
        env.update(STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(cat.root), PYTHONNOUSERSITE="1")
        for k in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            env.pop(k, None)
        env["PYTHONPATH"] = str(cat.src)
        p = self._spawn(argv, env=env, cwd=str(cat.src))
        lines: list[str] = []
        folder = str(cat.data_dir() / "models" / cat.tag_of(s.family, s.model))
        total = float(next((r.download_gb for r in cat.rows() if r.family == s.family and r.model == s.model), 0) or 0)
        for line in p.stdout:                                # type: ignore[union-attr]
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            line = line.rstrip()
            if not line:
                continue
            lines.append(line)
            lines[:] = lines[-400:]
            prog = cat.mcp.install_progress(lines)
            prog = cat.mcp.combined_progress(prog, {"folder": folder, "total_gb": total})
            dl = prog.get("download")
            if dl and not s.model_path and dl.get("total_gb", 0) >= 0.05:
                self.emit("model", f"Downloading {dl['done_gb']:.1f} of {dl['total_gb']:.1f} GB", float(dl["percent"]), step=prog.get("step"))
            elif prog.get("step"):
                self.emit("model", f"Setup step {prog['step']}", None, step=prog.get("step"))
        code = p.wait()
        if code != 0:
            err = ""
            for line in reversed(lines):
                if line.strip().startswith("[X]"):
                    err = line.strip()[3:].strip()
                    break
            hint = ""
            if err:
                for i, line in enumerate(lines):
                    if err in line and i + 1 < len(lines) and lines[i + 1].startswith("       "):
                        hint = lines[i + 1].strip()
            raise LaunchError("The model could not be installed." + (f" {err}." if err else ""),
                              hint or "Run it again: finished steps are skipped and a download continues where it stopped.", 6)
        path, _desc = self.installed_state()
        if path is None:
            raise LaunchError("Setup finished but no run config was written for this model.",
                              "Run `strata launch` again; if it repeats, the setup output above says why.", 6)
        self.emit("model", f"{title} is ready.", 100)
        return path

    def resolve_own_files(self) -> None:
        """"Use model files I already have": the fingerprint table says which catalog model the files are."""
        from strata_control import models
        s = self.s
        try:
            plan = models.plan_import(str(Path(s.model_path).expanduser().resolve()))
        except Exception as e:                               # noqa: BLE001
            raise LaunchError(f"These model files cannot be used: {e}.", "Pick the folder with every shard of one model, or a .gguf file of it.", 5) from None
        sel = plan.get("candidate_selection") or {}
        fam, model = sel.get("family"), sel.get("model")
        if not (fam and model) or plan.get("issues"):
            issues = "; ".join(plain_issue(str(x)) for x in (plan.get("issues") or [])[:3]) or "they are not a model Strata knows"
            raise LaunchError(f"These files cannot be used as they are ({issues}).",
                              "Choose the matching model in the list, or check that every shard is in the folder.", 5)
        s.family, s.model = fam, model
        self.emit("model", f"These files are {fam} {model}.", None)

    # ---- 3. config and profile
    def build_config(self, base_path: Path) -> Path:
        from strata_control import tune_knobs
        base = read_config(base_path)
        cfg = derive_config(base, self.s, bool_flags=tune_knobs.BOOL_FLAGS,
                            log=str(launch_dir() / f"{self.name}.engine.log"))
        out = launch_dir() / f"{self.name}.json"
        write_private(out, cfg)
        self.config_path = out
        return out

    def ensure_profile(self, config: Path) -> None:
        from strata_control import profiles
        name = self.profile
        parallel = self.s.parallel if self.s.parallel else self.catalog_parallel()
        sets = [f"host={acc.host_for(self.s.access)}", f"port={int(self.s.port)}"] + ([f"parallel={parallel}"] if parallel else [])
        try:
            current, _rev = profiles.load(name)
        except profiles.ProfileError:
            profiles.create(name, str(config))
        else:
            if current.get("source_config") != str(config.resolve()):
                proposed, rev = profiles.propose_source(name, str(config))
                profiles.save(name, proposed, expected_revision=rev)
        # the overrides are replaced as a set: a launch's host/port/parallel are exactly these
        cur, rev = profiles.load(name)
        want = {k: v for k, v in (profiles.parse_assignment(x) for x in sets)}
        if cur.get("overrides") != want:
            profiles.save(name, {**cur, "overrides": want}, expected_revision=rev)

    def catalog_parallel(self) -> int | None:
        try:
            comp = self.catalog.computer()
            return S.fit_parallel(comp.main_vram_gb) if comp.usable_gpus else None
        except Exception:                                    # noqa: BLE001
            return None

    # ---- 4. tuning
    def tune(self) -> None:
        s = self.s
        if s.tune == "skip":
            return
        from strata_control import presets
        if s.tuned_preset:
            try:
                presets.find(s.tuned_preset)
                self.emit("tune", "This preset is already tuned for this computer.", 100)
                return
            except presets.PresetError:
                pass
        mode = "thorough" if s.tune == "thorough" else "quick"
        status_file = launch_dir() / f"{self.name}.tune.json"
        try:
            status_file.unlink()
        except OSError:
            pass
        argv = self._cli("tune", "auto", self.profile, "--mode", mode, "--output", "json", "--status-file", str(status_file),
                         "--name", (s.name or self.name))
        if s.tune_allow_small:
            argv.append("--allow-small-differences")
        self.emit("tune", "Tuning for the best speed. Close other heavy programs while it measures.", 0)
        p = self._spawn(argv)
        out_lines: list[str] = []
        reader = threading.Thread(target=lambda: out_lines.extend(p.stdout.readlines()), daemon=True)   # type: ignore[union-attr]
        reader.start()
        while p.poll() is None:
            if self.cancel.is_set():
                self._stop_children()
                raise Cancelled()
            st = _read_json(status_file)
            if st:
                done, planned = st.get("runs_done") or 0, st.get("runs_planned") or 0
                pct = (100.0 * done / planned) if planned else None
                left = st.get("minutes_left")
                self.emit("tune", st.get("message") or "Measuring…", pct, speeds=st.get("speeds") or [],
                          round=(done, planned), minutes_left=left)
            time.sleep(1.0)
        reader.join(timeout=3)
        result = None
        for line in reversed(out_lines):
            try:
                result = json.loads(line).get("result")
                break
            except (ValueError, AttributeError):
                continue
        self.tune_result = result
        word, sentence = tune_words(result, p.returncode)
        self.tune_word = word
        preset = (result or {}).get("preset") or {}
        if preset.get("id"):
            s.tuned_preset, s.tuned_at = preset["id"], time.strftime("%Y-%m-%d")
        elif result and result.get("outcome") == "default_confirmed":
            s.tuned_at = time.strftime("%Y-%m-%d")
        self.emit("tune", sentence, 100, level="info" if word in ("faster", "already fastest") else "warn", word=word)

    tune_word: str | None = None

    # ---- 5. start
    def start(self) -> None:
        from strata_control import runtime
        from strata_control.profiles import ProfileError
        name = self.profile
        st = runtime.status(name)
        if st.get("state") in ("ready", "starting"):
            self.emit("start", "A server for this preset is already running; restarting it with these choices…")
            try:
                runtime.stop(name, timeout=120)
            except ProfileError as e:
                raise LaunchError(f"The running server could not be stopped: {e}.", "Run `strata launch --stop` first.", 6) from None
        self.emit("start", "Starting the engine and loading the model. The first start can take a few minutes.", None)
        done = threading.Event()
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["r"] = runtime.start(name, wait=600)
            except Exception as e:                           # noqa: BLE001
                box["e"] = e
            finally:
                done.set()

        threading.Thread(target=go, daemon=True).start()
        t0 = time.monotonic()
        while not done.wait(1.0):
            if self.cancel.is_set():
                try:
                    runtime.stop(name, timeout=60)
                except Exception:                            # noqa: BLE001
                    pass
                raise Cancelled()
            self.emit("start", f"Loading the model… {int(time.monotonic() - t0)} s", None)
        if "e" in box:
            raise LaunchError(f"The server could not start: {box['e']}", _log_hint(self.config_path), 6)
        r = box["r"]
        if not r.get("ok"):
            raise LaunchError(f"The server did not become ready ({r.get('state')}: {r.get('error') or r.get('message') or 'see the engine log'}).",
                              _log_hint(self.config_path), 6)
        self.emit("start", "The model is loaded and answering.", 100)

    # ---- 6. access
    def open_access(self) -> str | None:
        s = self.s
        missing = acc.missing_helper(s.access)
        tunnel = None
        if s.access == "cloudflare":
            if missing:
                self.notices.append(missing + " The server is running on this computer meanwhile.")
                self.emit("access", missing, level="warn")
            else:
                self.emit("access", "Opening a Cloudflare quick tunnel…")
                tunnel, msg = acc.start_tunnel(self.profile, int(s.port))
                if tunnel is None:
                    self.notices.append(msg)
                self.emit("access", msg, level="info" if tunnel else "warn", url=tunnel)
        elif missing:
            self.notices.append(missing)
            self.emit("access", missing, level="warn")
        else:
            self.emit("access", "Ready: " + ", ".join(acc.urls_for(s)), 100)
        return tunnel

    # ---- all of it
    def plan(self) -> dict:
        s = self.s
        warns = [n.line() for n in (*S.check_access(s),)]
        return {"profile": self.profile, "settings": s.to_dict(with_key=False), "warnings": warns,
                "flags": S.flags_text(s), "steps": list(STAGES)}

    def run(self, *, start: bool = True) -> Result:
        try:
            self.ensure_engine()
            self.check_cancel()
            base = self.ensure_model()
            self.check_cancel()
            self.emit("config", "Writing this launch's settings…")
            cfg = self.build_config(base)
            self.ensure_profile(cfg)
            self.emit("config", "Settings are saved.", 100)
            self.tune()
            self.check_cancel()
            if not start:
                return Result(self.profile, [], self.s.api_key, self.tune_word, self.notices, str(cfg))
            self.start()
            tunnel = self.open_access()
            urls = acc.urls_for(self.s, tunnel_url=tunnel)
            res = Result(self.profile, urls, self.s.api_key, self.tune_word, self.notices, str(cfg), tunnel)
            self.emit("done", "Strata is serving.", 100, urls=urls)
            return res
        except Cancelled:
            self._stop_children()
            raise
        except LaunchError:
            self._stop_children()
            raise

    def stop(self) -> dict:
        from strata_control import runtime
        out = runtime.stop(self.profile, timeout=180)
        acc.stop_tunnel(self.profile)
        return out


ISSUE_WORDS = {
    "multiple_model_groups": "the folder holds more than one model",
    "incomplete_or_invalid_shard_set": "some shards are missing",
    "unrecognized_or_unsupported_model_variant": "this is not a model size Strata knows",
    "unsupported_or_missing_architecture": "this is not a model Strata can run",
    "unsupported_or_incomplete_tokenizer_metadata": "the file has no usable tokenizer",
    "missing_or_unexpected_expert_quantization": "its quantization is not one Strata knows",
}


def plain_issue(code: str) -> str:
    return ISSUE_WORDS.get(code, code.replace("_", " "))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _log_hint(config: Path | None) -> str:
    log = None
    if config:
        try:
            log = read_config(config).get("log")
        except (OSError, ValueError):
            log = None
    return f"The engine's log has the reason: {log}" if log else "The engine's log has the reason (strata logs, or the log named in the run config)."
