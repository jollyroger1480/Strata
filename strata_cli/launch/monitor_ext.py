"""The Strata Monitor as `strata launch` opens it: the same monitor plus a Connect view and a quit question.

Everything about the monitor itself (the views, colours, polling, key hints) is `strata_cli.monitor.MonitorApp`, unchanged.
This subclass adds
* key 7: the Connect view (the address, the API key hint and copy-ready setup for each app, with `c` copy, `w` write the config for
  you after a backup, `h` check that the app's settings reach a server that answers a prompt and a tool call correctly);
* q: asks whether to keep serving in the background or stop (stopping frees the graphics memory).
"""

from __future__ import annotations

import threading

from rich.text import Text
from textual.binding import Binding
from textual.widgets import Static

from strata_cli.launch import check, harness
from strata_cli.launch.screens import ConfirmScreen
from strata_cli.launch.widgets import ACCENT, BAD, MUTE, WARN, Glyphs
from strata_cli.monitor import MonitorApp
from strata_cli.render import VIEWS, safe


class LaunchMonitor(MonitorApp):
    BINDINGS = MonitorApp.BINDINGS + [Binding("7", "connect", "Connect view", show=False)]

    def __init__(self, endpoint, snapshot, *, info: dict, **kw) -> None:
        super().__init__(endpoint, snapshot, **kw)
        self.info = info
        self.connect_enabled = True
        self.connect_open = False
        self.connect_index = 0
        self.show_key = False
        self.check_results: dict[str, list[check.Check] | str] = {}
        self.glyphs = Glyphs()
        self.stop_requested = False
        self.notice = ""

    # ---- quit asks
    def action_quit(self) -> None:
        if self.closing:
            return

        def done(choice: str | None) -> None:
            if choice == "keep":
                self.closing = True
                self.exit()
            elif choice == "stop":
                self.stop_requested = True
                self.closing = True
                self.exit()

        self.push_screen(ConfirmScreen("Leave the monitor?", "Keeping it serving lets your apps go on using the model. "
                                       "Stopping unloads the model and frees the graphics memory.",
                                       [("keep", "Keep serving in the background"), ("stop", "Stop serving and free the memory"),
                                        ("cancel", "Stay in the monitor")]), done)

    # ---- the Connect view
    def _harness(self) -> harness.Harness:
        return harness.HARNESSES[self.connect_index % len(harness.HARNESSES)]

    def action_connect(self) -> None:
        self.connect_open = not self.connect_open
        self.notice = ""
        self._paint()

    def _set_view(self, index: int) -> None:
        self.connect_open = False
        super()._set_view(index)

    def action_next_row(self) -> None:
        if self.connect_open:
            self.connect_index = (self.connect_index + 1) % len(harness.HARNESSES)
            self.notice = ""
            self._paint()
        else:
            super().action_next_row()

    def action_previous_row(self) -> None:
        if self.connect_open:
            self.connect_index = (self.connect_index - 1) % len(harness.HARNESSES)
            self.notice = ""
            self._paint()
        else:
            super().action_previous_row()

    def action_filter_clear(self) -> None:
        if self.connect_open:
            self.connect_open = False
            self._paint()
        else:
            super().action_filter_clear()

    def action_request_detail(self) -> None:
        if not self.connect_open:
            super().action_request_detail()

    def action_next_view(self) -> None:
        if self.connect_open:
            self.connect_open = False
            self._paint()
        super().action_next_view()

    def on_key(self, event) -> None:
        if not self.connect_open:
            return
        key = event.key
        if key == "c":
            self._copy()
        elif key == "w":
            self._write()
        elif key == "k":
            self.show_key = not self.show_key
            self._paint()
        elif key == "h":
            self._check()
        else:
            return
        event.stop()

    def _rendered(self) -> dict:
        h = self._harness()
        key = self.info.get("key") or ""
        return harness.render(h, harness.target(self.info["url"], key, self.info.get("model") or "strata", self.info.get("context")))

    def _copy(self) -> None:
        out = self._rendered()
        text = "\n\n".join(f"# {f['path']}\n{f['text']}" for f in out["files"])
        if out["env"]:
            text += "\n\n" + "\n".join(f"{k}={v}" for k, v in out["env"].items())
        try:
            self.copy_to_clipboard(text)
            self.notice = "Copied (if your terminal allows it)."
        except Exception:                                    # noqa: BLE001
            self.notice = "! Your terminal does not allow copying; use: strata connect " + self._harness().id
        self._paint()

    def _write(self) -> None:
        h = self._harness()
        out = self._rendered()
        if not out["files"]:
            self.notice = "! Nothing to write for this app."
            self._paint()
            return

        def done(choice: str | None) -> None:
            if choice != "write":
                return
            lines = []
            for f in out["files"]:
                try:
                    r = harness.write_file(f)
                    lines.append(f"{r['action']} {r['path']}" + (f" (backup {r['backup']})" if r["backup"] else ""))
                except OSError as e:
                    lines.append(f"could not write {f['path']}: {e}")
            self.notice = "; ".join(lines)
            self._paint()

        self.push_screen(ConfirmScreen(f"Write {h.title}'s settings?", "A file that exists is copied to a backup first.\n"
                                       + "\n".join(f"  {f['path']}" for f in out["files"]),
                                       [("write", "Write it"), ("no", "Not now")]), done)

    def _check(self) -> None:
        h = self._harness()
        self.check_results[h.id] = "running"
        self.notice = "Checking: one prompt and one tool call. This can take a minute on a big model."
        self._paint()
        key = self.info.get("key") or ""
        url = self.info["url"]
        model = self.info.get("model") or "strata"

        def work() -> None:
            res = check.run_checks(h.id, url, key, model)
            self.check_results[h.id] = res
            try:
                self.call_from_thread(self._paint)
            except Exception:                                # noqa: BLE001
                pass

        threading.Thread(target=work, daemon=True).start()

    # ---- painting
    def _paint(self, size=None) -> None:
        if not getattr(self, "connect_open", False):
            return super()._paint(size)
        size = self.screen.size if size is None else size
        width = max(40, size.width - 2)
        self.query_one("#content", Static).update(self.connect_text(width))
        idle = "c copy   w write it for me (keeps a backup)   h check it   1-7 views   q quit (asks: keep serving / stop)"
        msg = self.notice or idle
        self.query_one("#message", Static).update(Text(msg, style=WARN if msg.startswith("!") else ""))

    def connect_text(self, width: int) -> Text:
        g = self.glyphs
        state = {"connected": "[LIVE]", "stale": "[STALE]", "connecting": "[CONNECTING]", "disconnected": "[DISCONNECTED]",
                 "auth_required": "[AUTH REQUIRED]"}.get(self.connection_state, "[LIVE]")
        t = Text(no_wrap=True, overflow="crop")
        head = f"STRATA  Connect   {self.info['url']}"
        t.append(head, style=f"bold {ACCENT}")
        t.append(" " * max(2, width - len(head) - len(state)) + state + "\n", style=f"bold {ACCENT}")
        nav = []
        for i, name in enumerate(VIEWS):
            nav.append(f"{i + 1} {name.title()}")
        nav.append("[7 Connect]")
        t.append("  ".join(nav) + "\n")
        t.append(g.rule * min(width, 100) + "\n", style=MUTE)
        key = self.info.get("key") or ""
        shown = key if self.show_key else (key[:10] + g.mask * 4 if key else "off")
        t.append("Connect an app", style=f"bold {ACCENT}")
        t.append(f"     URL {self.info['url']}/v1    model {self.info.get('model') or 'strata'}    key {shown}  ")
        t.append("(k hide)" if self.show_key else "(k show)", style=MUTE)
        t.append("\n")
        for u in self.info.get("urls", [])[1:]:
            t.append(f"     also reachable at {u}/v1\n", style=MUTE)
        t.append("\n   " + "App".ljust(20) + "Works with".ljust(22) + "Status\n", style="bold")
        for i, h in enumerate(harness.HARNESSES):
            sel = i == self.connect_index % len(harness.HARNESSES)
            res = self.check_results.get(h.id)
            status = ("tested " + g.tick) if h.tested else "not tested yet"
            style = ""
            if res == "running":
                status, style = "checking…", WARN
            elif isinstance(res, list):
                okn = sum(1 for c in res if c.ok)
                status, style = (f"works {g.tick} ({okn}/{len(res)} checks)" if okn == len(res) else f"problem ({okn}/{len(res)} checks)"), \
                                (ACCENT if okn == len(res) else BAD)
            row = ("  " + (g.cursor + " " if sel else "  ")) + h.title.ljust(20) + h.speaks.ljust(22) + status
            t.append(row.ljust(width - 1) + "\n", style=(f"bold {ACCENT} on #1f2a27" if sel else style))
        h = self._harness()
        out = self._rendered()
        key = self.info.get("key") or ""
        height = max(12, self.screen.size.height)
        used = t.plain.count("\n") + 2
        room = max(3, height - used - 3)
        t.append("\n")
        paths = ", ".join(f["path"] for f in out["files"]) or ("built into the Strata app" if h.built_in else "")
        t.append(f"{h.title} setup", style=f"bold {ACCENT}")
        t.append(f"   {paths}\n", style=MUTE)
        room -= 2
        import textwrap
        limit_lines: list[str] = []
        for n in out["limits"]:
            wrapped = textwrap.wrap(n, max(30, width - 6)) or [n]
            limit_lines += ["  - " + wrapped[0]] + ["    " + w for w in wrapped[1:]]
        code: list[str] = []
        for f in out["files"][:1]:
            body = f["text"] if self.show_key else harness.mask(f["text"], key)
            code += ["  " + ln for ln in body.splitlines()]
        for k, v in out["env"].items():
            code.append(f"  {k}={v if self.show_key else harness.mask(v, key)}")
        res = self.check_results.get(h.id)
        res_lines = ([f"  {g.tick if c.ok else 'x'} {c.name}: {c.detail}" for c in res] if isinstance(res, list) else [])
        want_code = min(len(code), max(3, room - min(len(limit_lines), 6) - len(res_lines) - 1))
        for ln in code[:want_code]:
            t.append(ln + "\n")
        if len(code) > want_code:
            t.append(f"  ... {len(code) - want_code} more lines (c copies all)\n", style=MUTE)
        left = max(0, room - want_code - (1 if len(code) > want_code else 0) - len(res_lines))
        if limit_lines:
            t.append("Good to know\n", style=f"bold {ACCENT}")
            for ln in limit_lines[:max(2, left)]:
                t.append(ln + "\n", style=WARN if ln.lstrip().startswith("- WARNING") else MUTE)
        if isinstance(res, list):
            for c, ln in zip(res, res_lines):
                t.append(ln + "\n", style=ACCENT if c.ok else BAD)
        return t
