"""Who can reach the server: this computer, the local network, a Cloudflare quick tunnel, Tailscale.

Strata starts the server; this module only works out the addresses, checks that the helper programs exist (and says in plain
words how to install them when they do not), and runs/stops the one helper Strata starts itself: the cloudflared quick tunnel.
Nothing here changes a firewall, a router or a Tailscale setting. The cloudflared output is filtered: only the public address
is ever kept or shown (a login or certificate line is dropped, never printed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from strata_cli.launch.settings import ACCESS_KINDS, LaunchSettings

TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
WINDOWS = os.name == "nt"


def host_for(kind: str) -> str:
    """The address the server listens on: loopback for this computer only, every address for the rest.

    Cloudflare's tunnel connects from the same computer, so it too can stay on loopback."""
    return "127.0.0.1" if kind in ("local", "cloudflare") else "0.0.0.0"


def lan_ips() -> list[str]:
    """This computer's private IPv4 addresses, the one used for the default route first."""
    found: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.2)
        s.connect(("10.255.255.255", 1))          # no packet is sent; it picks the interface of the default route
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        import psutil
        for infos in psutil.net_if_addrs().values():
            for i in infos:
                if i.family == socket.AF_INET:
                    found.append(i.address)
    except Exception:                                      # noqa: BLE001
        pass
    private = [a for a in dict.fromkeys(found) if _is_private(a) and not a.startswith("127.") and not a.startswith("169.254.")]
    return private


def _is_private(address: str) -> bool:
    try:
        a, b = (int(x) for x in address.split(".")[:2])
    except ValueError:
        return False
    return a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168) or (a == 100 and 64 <= b <= 127)


def find_cloudflared() -> str | None:
    found = shutil.which("cloudflared")
    if found:
        return found
    for guess in ("/usr/local/bin/cloudflared", "/opt/homebrew/bin/cloudflared", r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
                  r"C:\Program Files\cloudflared\cloudflared.exe"):
        if Path(guess).is_file():
            return guess
    return None


def find_tailscale() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for guess in ("/usr/bin/tailscale", "/usr/local/bin/tailscale", "/Applications/Tailscale.app/Contents/MacOS/Tailscale",
                  r"C:\Program Files\Tailscale\tailscale.exe"):
        if Path(guess).is_file():
            return guess
    return None


def _run(argv: list[str], timeout: float = 8.0) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return r.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def tailscale_status() -> dict:
    """{"installed", "running", "ip", "name"}: the tailnet address of this computer when Tailscale is up."""
    exe = find_tailscale()
    out = {"installed": bool(exe), "running": False, "ip": None, "name": None}
    if not exe:
        return out
    ip = _run([exe, "ip", "-4"]).strip().splitlines()
    if ip and re.fullmatch(r"\d+\.\d+\.\d+\.\d+", ip[0].strip()):
        out["running"], out["ip"] = True, ip[0].strip()
        try:
            data = json.loads(_run([exe, "status", "--json"]) or "{}")
            name = (data.get("Self") or {}).get("DNSName")
            out["name"] = name.rstrip(".") if isinstance(name, str) and name else None
        except ValueError:
            pass
    return out


INSTALL_HELP = {
    "cloudflared": {
        "linux": "Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ "
                 "(on Debian/Ubuntu download the .deb and run: sudo dpkg -i cloudflared-linux-amd64.deb). No account is needed for a quick tunnel.",
        "windows": "Install cloudflared: winget install --id Cloudflare.cloudflared  (no account is needed for a quick tunnel).",
        "darwin": "Install cloudflared: brew install cloudflared  (no account is needed for a quick tunnel).",
    },
    "tailscale": {
        "linux": "Install Tailscale: curl -fsSL https://tailscale.com/install.sh | sh, then run: sudo tailscale up",
        "windows": "Install Tailscale from https://tailscale.com/download, sign in, and leave it running.",
        "darwin": "Install Tailscale from https://tailscale.com/download, sign in, and leave it running.",
    },
}


def install_help(tool: str) -> str:
    plat = "windows" if WINDOWS else "darwin" if sys.platform == "darwin" else "linux"
    return INSTALL_HELP[tool][plat]


@dataclass
class AccessOption:
    kind: str
    label: str
    url: str
    note: str
    ready: bool          # the helper it needs is there (or none is needed)
    warn: bool = False


def options(port: int, *, tunnel_url: str | None = None) -> list[AccessOption]:
    """The four ways to share, with the address each would give and what is missing for it."""
    ips = lan_ips()
    lan = ips[0] if ips else None
    cf = find_cloudflared()
    ts = tailscale_status()
    out = [AccessOption("local", "This computer only", f"http://127.0.0.1:{port}",
                        "for apps on this computer (OpenCode, Codex, Claude Code...)", True)]
    out.append(AccessOption("lan", "My local network", f"http://{lan or '<network address>'}:{port}",
                            "other computers at home or in the office" if lan else
                            "other computers on your network; no network address was found on this computer", bool(lan), warn=not lan))
    out.append(AccessOption("cloudflare", "Cloudflare tunnel", tunnel_url or "https://<random>.trycloudflare.com",
                            "from anywhere, no router setup · cloudflared found" if cf else "cloudflared is not installed",
                            bool(cf), warn=not cf))
    if ts["running"]:
        out.append(AccessOption("tailscale", "Tailscale", f"http://{ts['name'] or ts['ip']}:{port}",
                                f"your Tailscale devices · tailnet address {ts['ip']}", True))
    else:
        out.append(AccessOption("tailscale", "Tailscale", f"http://<tailnet address>:{port}",
                                "Tailscale is installed but not connected (run: tailscale up)" if ts["installed"] else
                                "Tailscale is not installed", False, warn=True))
    return out


def option_for(kind: str, port: int) -> AccessOption:
    return next(o for o in options(port) if o.kind == kind)


def missing_helper(kind: str) -> str | None:
    """Plain words about what to install before this access kind works, or None when it is ready."""
    if kind == "cloudflare" and not find_cloudflared():
        return "cloudflared is not installed on this computer. " + install_help("cloudflared")
    if kind == "tailscale":
        ts = tailscale_status()
        if not ts["installed"]:
            return "Tailscale is not installed on this computer. " + install_help("tailscale")
        if not ts["running"]:
            return "Tailscale is installed but not connected. Run: tailscale up"
    if kind == "lan" and not lan_ips():
        return "No local network address was found on this computer. Check that it is connected to your network."
    return None


def urls_for(s: LaunchSettings, *, tunnel_url: str | None = None) -> list[str]:
    """The addresses a client can use for these choices (the first is the main one)."""
    base = [f"http://127.0.0.1:{s.port}"]
    if s.access == "lan":
        base += [f"http://{ip}:{s.port}" for ip in lan_ips()[:1]]
    elif s.access == "cloudflare" and tunnel_url:
        base = [tunnel_url] + base
    elif s.access == "tailscale":
        ts = tailscale_status()
        if ts["running"]:
            base += [f"http://{ts['name'] or ts['ip']}:{s.port}"] + ([f"http://{ts['ip']}:{s.port}"] if ts["name"] else [])
    return base


def new_key() -> str:
    import secrets
    return "sk-strata-" + secrets.token_urlsafe(24)


def mask_key(key: str | None) -> str:
    if not key:
        return "off"
    return key[:10] + "•" * 8 if len(key) > 12 else "•" * 8


# ----------------------------------------------------------------------------------------------- the quick tunnel

def _state_dir() -> Path:
    from strata_control import paths
    d = paths.state_base() / "launch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _tunnel_state(name: str) -> Path:
    from strata_cli.launch.settings import slug
    return _state_dir() / f"{slug(name)}.tunnel.json"


def start_tunnel(name: str, port: int, *, wait: float = 45.0) -> tuple[str | None, str]:
    """Start `cloudflared tunnel --url` for the local server; returns (public url or None, plain message).

    The process is detached (it keeps running after this terminal closes, like the server) and recorded with its start token so
    `stop_tunnel` ends exactly this process and nothing else. Only the public address is read from its output."""
    exe = find_cloudflared()
    if not exe:
        return None, missing_helper("cloudflare") or "cloudflared is not installed."
    existing = tunnel_info(name)
    if existing and existing.get("url") and existing.get("port") == port:
        return existing["url"], "The tunnel was already running."
    stop_tunnel(name)
    log = _state_dir() / f"{Path(name).name}.tunnel.log"
    argv = [exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]
    with open(log, "wb") as out:
        kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW} if WINDOWS
              else {"start_new_session": True})
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, **kw)
    from strata_control import procs
    token = procs.start_token(proc.pid)
    url = None
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline and url is None and proc.poll() is None:
        try:
            m = TUNNEL_URL.search(log.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            m = None
        if m:
            url = m.group(0)
            break
        time.sleep(0.4)
    state = {"pid": proc.pid, "token": token, "cmd": list(procs.command_line(proc.pid) or argv), "url": url, "port": port}
    _tunnel_state(name).write_text(json.dumps(state), encoding="utf-8")
    try:
        log.write_bytes(b"")                       # keep no log text: the address is all anyone needs from it
    except OSError:
        pass
    if url is None:
        stop_tunnel(name)
        return None, ("The Cloudflare tunnel did not come up in time. Check your internet connection and try again; "
                      "the server itself is running and still works on this computer.")
    return url, "The Cloudflare tunnel is up. Anyone with the address can reach the server, so keep the API key on."


def tunnel_info(name: str) -> dict | None:
    try:
        state = json.loads(_tunnel_state(name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    from strata_control import procs
    if isinstance(state, dict) and procs.is_same(int(state.get("pid", 0)), state.get("token") or "", state.get("cmd")):
        return state
    return None


def stop_tunnel(name: str) -> bool:
    """End this launch's tunnel, only if the recorded process is still exactly the one Strata started."""
    state = tunnel_info(name)
    ok = False
    if state:
        from strata_control import procs
        outcome = procs.end_one({"pid": state["pid"], "token": state["token"], "cmd": state["cmd"]}, grace=5.0)
        ok = outcome in ("ended", "gone")
    try:
        _tunnel_state(name).unlink()
    except OSError:
        pass
    return ok


def filter_line(line: str) -> str:
    """For any helper output shown to a person: drop login/certificate/token lines, keep the rest."""
    low = line.lower()
    if any(w in low for w in ("token", "cert", "login", "secret", "authorize", "credential")):
        return ""
    return line
