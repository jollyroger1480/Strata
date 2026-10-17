"""The model catalog, the computer's figures and the fit words for `strata launch`.

There is ONE catalog: setup.py's MODELS / FAMILIES / CONTEXTS, which the desktop app reads through tools/strata_mcp.py
(`Strata.model_table`, `Strata.hardware`, `Strata.recommend`; the app page's own fit arithmetic is
apps/host/src/lib/native-fit.ts `fitOn`). This module loads that same file from the installer folder and ports `fitOn` line by
line; it holds no table of its own. Without the installer files (a Terminal bundle made before launch existed) it says so in
words instead of guessing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata_cli.launch import settings as S


class CatalogUnavailable(RuntimeError):
    """The installer files (setup.py, tools/strata_mcp.py) are not there; the message is for the user."""


# --- apps/host/src/lib/native-fit.ts, ported (same constants, same rules) ------------------------------------------------
DENSE_GB = 5.0
SYSTEM_ROOM_GB = 10.0
DENSE_EXTRA_GB = 1.0


def _round1(n: float) -> float:
    return round(n * 10) / 10


def memory_needed(v: dict) -> float:
    if isinstance(v.get("memory_gb"), (int, float)):
        return float(v["memory_gb"])
    if v.get("mem_total") and isinstance(v.get("ram_needed_gb"), (int, float)):
        return float(v["ram_needed_gb"])
    return _round1(float(v.get("experts_gb") or v.get("ram_needed_gb") or 0) + DENSE_EXTRA_GB)


def fit_on(v: dict, ram_gb: float | None, vram_gb: float | None) -> dict[str, Any]:
    """native-fit.ts `fitOn`: where the weights go on this computer. Returns memoryGb, vramUseGb, ramUseGb, spillGb, known."""
    memory = memory_needed(v)
    vram = float(vram_gb or 0)
    if not ram_gb or ram_gb <= 0:
        return {"memoryGb": memory, "vramUseGb": 0.0, "ramUseGb": 0.0, "spillGb": 0.0, "known": False}
    experts = max(0.0, memory - DENSE_EXTRA_GB)
    on_gpu = max(0.0, min(experts, vram - DENSE_GB))
    ram_room = max(0.0, float(ram_gb) - SYSTEM_ROOM_GB)
    spill = max(0.0, experts - on_gpu - ram_room)
    in_ram = max(0.0, experts - on_gpu - spill)
    vram_use = min(vram, DENSE_GB + on_gpu) if vram > 0 else 0.0
    return {"memoryGb": memory, "vramUseGb": _round1(vram_use),
            "ramUseGb": _round1(in_ram + (DENSE_EXTRA_GB if vram <= 0 else 0)), "spillGb": _round1(spill), "known": True}


# --- the verdict words ---------------------------------------------------------------------------------------------------
FITS, TIGHT, NO = "fits", "tight", "no"


def verdict(on_this_pc: str | None) -> tuple[str, str]:
    """(category, plain words) from the catalog's own `on_this_pc` text (setup.py `--check`'s rules, through strata_mcp).

    Never a block: `no` still lets the person try, and the screen says so."""
    text = (on_this_pc or "").strip()
    low = text.lower()
    if not text:
        return TIGHT, "not checked (this computer's memory could not be read)"
    if low.startswith("does not fit"):
        return NO, "too big (can still try)"
    if "low-ram mode" in low:
        return FITS, "fits, a bit slower"
    if "part of its experts" in low or "read from the ssd" in low:
        base = "slow here" in low
        return (TIGHT, "tight: much read from SSD") if base else (FITS, "fits, part read from SSD")
    if low.startswith("tight"):
        return TIGHT, "tight: close other programs"
    if "fits" in low:
        return FITS, "fits"
    return TIGHT, text


@dataclass
class ModelRow:
    family: str
    family_title: str
    model: str
    id: str                       # the run config's tag (strata-<id>.json)
    about: str
    download_gb: float
    need_vram_gb: float
    need_ram_gb: float
    spill_gb: float
    verdict: str
    words: str
    experimental: bool
    images: bool
    installed: bool
    recommended: bool = False
    mem_total: bool = False
    memory_gb: float = 0.0

    @property
    def title(self) -> str:
        return f"{self.family_title} {self.model}"


@dataclass
class Computer:
    ram_gb: float | None = None
    gpus: list[dict] = field(default_factory=list)
    disk_free_gb: float | None = None
    os: str = ""
    avx2: bool | None = None
    raw: dict = field(default_factory=dict)

    @property
    def usable_gpus(self) -> list[dict]:
        """The cards that count (strata_control/gpu_fit.py: an integrated chip or a tiny card is ignored), the largest first."""
        from strata_control import gpu_fit
        return gpu_fit.usable_cards(self.gpus)[0]

    @property
    def ignored_gpus(self) -> list[dict]:
        from strata_control import gpu_fit
        return gpu_fit.usable_cards(self.gpus)[1]

    @property
    def main_vram_gb(self) -> float:
        return max((float(g.get("vram_gb") or 0) for g in self.usable_gpus), default=0.0)

    def line(self, data_dir: str | None = None) -> str:
        g = self.usable_gpus
        if g and g[0].get("unified_gb"):                       # a Mac: one pool of memory, the chip is the graphics
            disk = f"{self.disk_free_gb:,.0f} GB free" if self.disk_free_gb is not None else ""
            return " · ".join(x for x in (g[0].get("name", "Apple silicon"), f"{self.ram_gb:.0f} GB unified memory" if self.ram_gb else "memory unknown", disk) if x)
        card = (f"{g[0].get('name', 'GPU').replace('NVIDIA GeForce ', '').replace('AMD Radeon ', '')} {g[0].get('vram_gb', 0):.0f} GB"
                + (f" x{len(g)}" if len(g) > 1 else "")) if g else "no usable graphics card found"
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM unknown"
        disk = f"{self.disk_free_gb:,.0f} GB free" if self.disk_free_gb is not None else ""
        return " · ".join(x for x in (card, ram, disk) if x)


def installer_folder(explicit: str | os.PathLike | None = None) -> Path:
    """Where setup.py and tools/ are: this Terminal's own folder (the bundle, or a source checkout)."""
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[2]


def install_root(explicit: str | os.PathLike | None = None) -> Path:
    """Where run configs of installed models live: STRATA_INSTALL_DIR, else beside the components folder (what the app uses)."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("STRATA_INSTALL_DIR")
    if env:
        return Path(env)
    from strata_control import engine_component
    return engine_component.components_dir().parent / "install"


# Intel Arc cards by PCI id: (name, memory in GB).  The kernel does not tell an ordinary user how much memory the card has, so the common
# cards are listed; an id that is not here is shown as "Intel Arc" with no size and is treated as a small card.  (A770: the 8 GB model shares
# the id of the 16 GB one, so the smaller figure is used.)
INTEL_ARC = {"0x56a1": ("Arc A750", 8.0), "0x56a2": ("Arc A580", 8.0), "0x56a0": ("Arc A770", 8.0), "0x56a5": ("Arc A380", 6.0),
             "0x56a6": ("Arc A310", 4.0), "0xe20b": ("Arc B580", 12.0), "0xe20c": ("Arc B570", 10.0), "0xe223": ("Arc Pro B70", 32.0)}


def intel_arc_cards(root: str = "/sys/class/drm") -> list[dict]:
    """The Intel Arc cards on this Linux computer, read from the kernel's device list (empty elsewhere)."""
    cards: list[dict] = []
    try:
        for dev in sorted(Path(root).glob("card[0-9]/device")):
            if (dev / "vendor").read_text().strip() != "0x8086":
                continue
            did = (dev / "device").read_text().strip().lower()
            if did not in INTEL_ARC:
                continue
            name, gb = INTEL_ARC[did]
            cards.append({"vendor": "intel", "name": "Intel " + name, "vram_gb": gb, "usable": True, "index": len(cards)})
    except OSError:
        pass
    return cards


class Catalog:
    """setup.py's tables through strata_mcp's `Strata` object (the app's own catalog code)."""

    def __init__(self, src: str | os.PathLike | None = None, root: str | os.PathLike | None = None,
                 hardware: dict | None = None):
        self.src = installer_folder(src)
        self.root = install_root(root)
        mcp = self.src / "tools" / "strata_mcp.py"
        if not mcp.is_file() or not (self.src / "setup.py").is_file():
            raise CatalogUnavailable(
                "This Terminal does not carry the model list. Install it again with the one-line installer, "
                "or run `strata launch` from a full Strata folder (the folder with setup.py).")
        self.root.mkdir(parents=True, exist_ok=True)
        name = "strata_mcp_for_launch"
        mod = sys.modules.get(name)
        if mod is None:
            if str(self.src) not in sys.path:
                sys.path.insert(0, str(self.src))
            spec = importlib.util.spec_from_file_location(name, mcp)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            real = sys.stdout
            sys.stdout = sys.stderr           # the module must not print into a screen
            try:
                spec.loader.exec_module(mod)
            finally:
                sys.stdout = real
        self.mcp = mod
        self.strata = mod.Strata(self.root, self.src)
        self._hardware_override = hardware

    # ---- the computer
    def hardware(self, fresh: bool = False) -> dict:
        if self._hardware_override is not None:
            hw = dict(self._hardware_override)
            hw.setdefault("disk_free_gb", self.strata.disk_free(self.strata.data_dir()))
            return hw
        hw = dict(self.strata.hardware(fresh=fresh))
        hw["gpus"] = self._admit_old_nvidia(list(hw.get("gpus") or []))
        if not hw.get("gpus") and not str(hw.get("os", "")).startswith("Windows"):
            arc = intel_arc_cards()
            if arc:
                hw["gpus"] = arc                       # the shared hardware reader knows NVIDIA and AMD only; an Arc is added here
        return hw

    @staticmethod
    def _admit_old_nvidia(gpus: list[dict]) -> list[dict]:
        """Pascal / Volta cards (P100, P40, V100: compute capability 6.0-7.0) run on the experimental CUDA 12 engine. setup.py admits them when they
        are the only kind of graphics card in the computer (`old_gpus_opt_in`: "it is the only kind of NVIDIA GPU in this PC"); the shared hardware
        reader does not, so the screens said "no usable graphics card found" and left out the Cards step on a computer with several V100s.
        A newer NVIDIA card or a usable AMD card keeps the old cards out of the way, exactly as in setup."""
        def arch(g: dict) -> int:
            try:
                return int(g.get("arch"))
            except (TypeError, ValueError):
                return 0
        old = [g for g in gpus if g.get("vendor") == "nvidia" and 60 <= arch(g) <= 70 and not g.get("usable", True)]
        if not old:
            return gpus
        if any(g.get("usable", True) for g in gpus):
            return gpus
        return [({**g, "usable": True, "problem": None,
                  "note": "experimental: the CUDA 12 engine (docs/OLDER_GPUS.md)"} if g in old else g) for g in gpus]

    def computer(self, fresh: bool = False) -> Computer:
        hw = self.hardware(fresh)
        return Computer(ram_gb=hw.get("ram_gb"), gpus=list(hw.get("gpus") or []), disk_free_gb=hw.get("disk_free_gb"),
                        os=str(hw.get("os") or ""), avx2=(hw.get("cpu") or {}).get("avx2"), raw=hw)

    # ---- the tables
    def tables(self):
        models, families, contexts, source = self.strata.tables()
        return models, families, list(contexts), source

    def contexts(self) -> list[int]:
        return [int(c) for c in self.tables()[2]]

    def recommendation(self, hw: dict | None = None) -> dict:
        return self.strata.recommend(hw or self.hardware())

    def rows(self, hw: dict | None = None) -> list[ModelRow]:
        hw = hw or self.hardware()
        comp = Computer(ram_gb=hw.get("ram_gb"), gpus=list(hw.get("gpus") or []))
        vram = comp.main_vram_gb
        rec = self.recommendation(hw)
        out: list[ModelRow] = []
        for fam in self.strata.model_table(hw):
            for size in fam["sizes"]:
                f = fit_on(size, hw.get("ram_gb"), vram)
                cat, words = verdict(size.get("on_this_pc"))
                out.append(ModelRow(
                    family=fam["family"], family_title=fam["title"], model=size["model"], id=size["id"],
                    about=size["about"], download_gb=float(size["download_gb"]), need_vram_gb=f["vramUseGb"],
                    need_ram_gb=f["ramUseGb"], spill_gb=f["spillGb"], verdict=cat, words=words,
                    experimental=bool(size.get("experimental") or fam.get("experimental")), images=bool(size.get("images")),
                    installed=bool(size.get("installed")), mem_total=bool(size.get("mem_total")),
                    memory_gb=float(size.get("memory_gb") or 0),
                    recommended=(rec.get("family") == fam["family"] and rec.get("model") == size["model"])))
        return out

    def resolve(self, model: str, family: str | None = None) -> tuple[str, str]:
        """`iq3_xxs` or `IQ3_XXS` (and an optional family) -> (family, model); raises ValueError with the closest match."""
        models, families, _ctx, _src = self.tables()
        names = list(models)
        match = next((n for n in names if n.lower() == model.lower()), None)
        if match is None:
            near = S.closest(model, names)
            raise ValueError(f"There is no model called '{model}'." + (f" Did you mean '{near}'?" if near else "")
                             + " Models: " + ", ".join(names) + ".")
        if family is not None and family not in families:
            near = S.closest(family, list(families))
            raise ValueError(f"There is no model family called '{family}'." + (f" Did you mean '{near}'?" if near else "")
                             + " Families: " + ", ".join(families) + ".")
        offered = [f for f in families if match in self.strata.sizes_of(models, f)]
        if family is not None:
            if family not in offered:
                raise ValueError(f"{families[family]['title']} has no size {match}. It offers: "
                                 + ", ".join(self.strata.sizes_of(models, family)) + ".")
            return family, match
        if not offered:
            raise ValueError(f"{match} is not offered by any model family here.")
        return ("qwen" if "qwen" in offered else offered[0]), match

    def tag_of(self, family: str, model: str) -> str:
        _m, families, _c, _s = self.tables()
        return (families[family].get("tag", "") + model).lower()

    def kv_gb(self, tokens: int, kv: str = "int8") -> float:
        """The conversation memory for this many tokens, in GB (setup.py's UD_FIT_KV_BYTES: 13 attention layers)."""
        mod = self.strata.setup_module()
        table = getattr(mod, "UD_FIT_KV_BYTES", None) or {"fp16": 2112, "int8": 1056, "q4_0": 576, "k8v4": 816}
        return tokens * 13 * table.get(kv, 1056) / 1e9

    def max_context(self, family: str, model: str, hw: dict | None = None) -> int | None:
        """The recommended context for this computer and model (setup.py: VRAM rule, then the RAM rule)."""
        hw = hw or self.hardware()
        if hw.get("mac") and hw.get("ram_gb"):                 # a Mac: the context by unified memory (strata_control/mac_fit.py)
            mac = self.strata.mac()
            if mac is not None:
                return int(mac.recommend_context(float(hw["ram_gb"])))
        usable = [g for g in hw.get("gpus") or [] if g.get("usable", True)]
        vram = max((float(g.get("vram_gb") or 0) for g in usable), default=12.0)
        rec = 32768 if vram < 14 else 65536 if vram < 20 else 131072
        if usable and all(g.get("vendor") == "intel" for g in usable):
            rec = 8192 if vram < 10 else 16384 if vram < 20 else 32768       # an Arc keeps the session and the experts in the same memory
        mod = self.strata.setup_module()
        models = self.tables()[0]
        if mod is not None and models[model].get("budget"):
            try:
                if mod.unsloth_budget(model, hw.get("ram_gb") or 0.0):
                    rec = 8192 if vram < 14 else 32768
            except Exception:                                  # noqa: BLE001
                pass
        if mod is not None and hasattr(mod, "ram_ctx") and hw.get("ram_gb"):
            try:
                rec = min(rec, int(mod.ram_ctx(model, hw["ram_gb"], False)))
            except Exception:                                  # noqa: BLE001
                pass
        return rec

    # ---- installed models
    def installed_configs(self) -> list[dict]:
        out = []
        for path in self.strata.configs():
            d = self.strata.describe_config(path)
            d["path"] = str(path)
            out.append(d)
        return out

    def installed_config(self, tag: str) -> Path | None:
        p = self.root / f"strata-{tag.lower()}.json"
        return p if p.is_file() else None

    def data_dir(self) -> Path:
        return self.strata.data_dir()

    def disk_free(self, path: str | os.PathLike) -> float | None:
        return self.strata.disk_free(Path(path))
