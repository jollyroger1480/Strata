"""What kind of terminal this is, for the few places that must change what they draw.

The classic Windows console window (conhost: the default window of Windows PowerShell and cmd on Windows 10) uses Consolas, which
has no glyphs for the eighth-block sparkline characters, the rounded box corners or the check mark: they show as empty boxes.
Windows Terminal, VS Code's terminal, ConEmu and every Linux/macOS terminal with a modern font have them. `STRATA_GLYPHS=ascii|unicode`
overrides the guess.
"""

from __future__ import annotations

import os
import sys


def legacy_console() -> bool:
    """True for the classic Windows console (no Windows Terminal, no VS Code, no ConEmu, no ssh session into it)."""
    forced = os.environ.get("STRATA_GLYPHS", "").strip().lower()
    if forced in ("ascii", "legacy"):
        return True
    if forced == "unicode":
        return False
    if os.name != "nt" and sys.platform != "win32":
        return False
    env = os.environ
    return not (env.get("WT_SESSION") or env.get("TERM_PROGRAM") or env.get("ConEmuANSI") or env.get("ANSICON")
                or env.get("TERM", "").startswith(("xterm", "screen", "tmux", "vt")))


def console_columns() -> int | None:
    """The visible width of the Windows console window in columns (not its buffer), or None when it cannot be read."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class COORD(ctypes.Structure):
            _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]

        class RECT(ctypes.Structure):
            _fields_ = [("Left", ctypes.c_short), ("Top", ctypes.c_short), ("Right", ctypes.c_short), ("Bottom", ctypes.c_short)]

        class INFO(ctypes.Structure):
            _fields_ = [("size", COORD), ("cursor", COORD), ("attr", wintypes.WORD), ("window", RECT), ("maxsize", COORD)]

        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        info = INFO()
        if not ctypes.windll.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(info)):
            return None
        return int(info.window.Right - info.window.Left + 1)
    except Exception:                                        # noqa: BLE001
        return None
