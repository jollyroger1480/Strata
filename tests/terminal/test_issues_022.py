"""Small fixes from the 0.2.2 issue sweep, one test class per issue number."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class Issue3298MtpDepth(unittest.TestCase):
    """Qwen3.6 passes `--mtp 2` (a draft depth); it is not a folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_doctor_does_not_look_for_a_folder_named_2(self):
        from strata_control import doctor
        engine = self.root / "engine"
        engine.write_text("x", encoding="utf-8")
        engine.chmod(0o700)
        tok = self.root / "tok"
        tok.mkdir()
        for n in ("vocab.json", "merges.txt", "token_type.json"):
            (tok / n).write_text("{}", encoding="utf-8")
        model = self.root / "m.gguf"
        model.write_bytes(b"x")
        cfg = self.root / "run.json"
        cfg.write_text(json.dumps({"exe": str(engine), "args": ["--native", str(model), "--mtp", "2", "--vision"],
                                   "cwd": str(self.root), "tokenizer": str(tok), "model_name": "q36", "port": 8095}),
                       encoding="utf-8")
        report = doctor.inspect(cfg)
        self.assertNotIn("mtp", [c["name"] for c in report["components"]])
        self.assertEqual([c["name"] for c in report["components"] if c["status"] not in ("ok",)
                          and c.get("required") and c["name"] in ("native", "mtp", "pack")], [])

    def test_doctor_still_checks_an_mtp_folder(self):
        from strata_control import doctor
        engine = self.root / "engine"
        engine.write_text("x", encoding="utf-8")
        engine.chmod(0o700)
        tok = self.root / "tok"
        tok.mkdir()
        for n in ("vocab.json", "merges.txt", "token_type.json"):
            (tok / n).write_text("{}", encoding="utf-8")
        cfg = self.root / "run.json"
        cfg.write_text(json.dumps({"exe": str(engine), "args": ["--mtp", str(self.root / "gone")],
                                   "cwd": str(self.root), "tokenizer": str(tok), "model_name": "x", "port": 8095}),
                       encoding="utf-8")
        report = doctor.inspect(cfg)
        mtp = [c for c in report["components"] if c["name"] == "mtp"]
        self.assertEqual(len(mtp), 1)
        self.assertEqual(mtp[0]["status"], "missing")

    def test_launch_files_present_ignores_a_depth_number(self):
        from strata_cli.launch.runner import files_present
        model = self.root / "m.gguf"
        model.write_bytes(b"x")
        self.assertEqual(files_present({"args": ["--native", str(model), "--mtp", "2"]}), [])
        self.assertEqual(files_present({"args": ["--mtp", str(self.root / "gone")]}), [str(self.root / "gone")])


class Issue3403MacBuildTools(unittest.TestCase):
    """The Mac setup compiles without setup.cmake_build(), so cmake and ninja must be installed for it too."""

    def test_missing_build_tools_are_installed_before_the_mac_compile(self):
        from unittest import mock
        import setup
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metal"))
        import setup_mac as M
        installed = {"cmake": None, "ninja": None}

        def find(name):
            return installed[name]

        def pip(packages, what):
            self.assertTrue(all(p.split("==")[0] in ("cmake", "ninja") for p in packages), packages)
            for p in packages:
                installed[p.split("==")[0]] = "/venv/bin/" + p.split("==")[0]

        with mock.patch.object(setup, "find_tool", side_effect=find), mock.patch.object(setup, "pip_install", side_effect=pip) as pi:
            self.assertEqual(setup.ensure_build_tools(), ("/venv/bin/cmake", "/venv/bin/ninja"))
            self.assertEqual(pi.call_count, 1)
            setup.ensure_build_tools()                       # present now: nothing more is installed
            self.assertEqual(pi.call_count, 1)

    def test_both_mac_builds_ask_for_the_tools_first(self):
        import inspect
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "metal"))
        import setup_mac as M
        for fn in (M.build_engine, M.build_vision):
            self.assertIn("S.ensure_build_tools()", inspect.getsource(fn), fn.__name__)


class Issue3188BenchStartsItsOwnServer(unittest.TestCase):
    """tools/strata_bench.py named `port` in the start-a-server branch without ever setting it."""

    def test_the_start_branch_runs_with_the_config_port(self):
        from unittest import mock
        from tools import strata_bench as sb
        import io
        from contextlib import redirect_stdout
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / "run.json"
            cfg.write_text(json.dumps({"port": 18999, "args": []}), encoding="utf-8")
            started = []
            with mock.patch.object(sb, "is_strata", return_value=False), \
                    mock.patch.object(sb, "try_json", return_value=None), \
                    mock.patch.object(sb, "start_server", side_effect=lambda c, p, g, log: started.append(p) or object()), \
                    mock.patch.object(sb, "wait_ready", return_value=False), \
                    mock.patch.object(sb, "stop_server"):
                out = io.StringIO()
                with redirect_stdout(out), mock.patch("sys.stderr", io.StringIO()):
                    code = sb.main(["--config", str(cfg)])
                self.assertEqual(code, 1)
                self.assertEqual(started, [18999])
                self.assertIn("on port 18999", out.getvalue())
                started.clear()
                with redirect_stdout(io.StringIO()), mock.patch("sys.stderr", io.StringIO()):
                    sb.main(["--config", str(cfg), "--port", "18777"])
                self.assertEqual(started, [18777])


@unittest.skipUnless(sys.platform == "win32", "SDDL aliases exist on Windows only")
class Issue3302AdministratorAlias(unittest.TestCase):
    """SDDL prints the built-in Administrator (RID 500) as `LA`; the owner-only checks must still accept it."""

    def setUp(self):
        from unittest import mock
        from strata_control import private_files as pf
        self.pf = pf
        self.mock = mock
        # the built-in Administrator's string SID on this machine: resolve the alias the same way the fix does
        import ctypes
        from ctypes import wintypes
        pointer = ctypes.c_void_p()
        pf._advapi.ConvertStringSidToSidW("LA", ctypes.byref(pointer))
        text = wintypes.LPWSTR()
        pf._advapi.ConvertSidToStringSidW(pointer.value, ctypes.byref(text))
        self.admin_sid = text.value
        self.assertTrue(self.admin_sid.endswith("-500"))

    def test_alias_matches_the_account_it_names(self):
        self.assertTrue(self.pf._same_sid("LA", self.admin_sid))
        self.assertTrue(self.pf._same_sid(self.admin_sid, self.admin_sid))
        self.assertFalse(self.pf._same_sid("LA", "S-1-5-21-1-2-3-1001"))
        self.assertFalse(self.pf._same_sid("WD", self.admin_sid))
        self.assertFalse(self.pf._same_sid("S-1-5-21-1-2-3-1001", self.admin_sid))

    def test_private_and_unshared_hold_for_the_built_in_administrator(self):
        pf = self.pf
        with self.mock.patch.object(pf, "current_user_sid", return_value=self.admin_sid), \
                self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;LA)"):
            self.assertTrue(pf.is_private("x"))
            self.assertTrue(pf.is_unshared("x"))
        with self.mock.patch.object(pf, "current_user_sid", return_value=self.admin_sid), \
                self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;S-1-5-21-1-2-3-1001)"):
            self.assertFalse(pf.is_private("x"))
            self.assertFalse(pf.is_unshared("x"))
        with self.mock.patch.object(pf, "current_user_sid", return_value=self.admin_sid), \
                self.mock.patch.object(pf, "dacl_sddl", return_value="D:PAI(A;OICI;FA;;;LA)(A;OICI;FR;;;WD)"):
            self.assertFalse(pf.is_private("x"))
            self.assertFalse(pf.is_unshared("x"))


if __name__ == "__main__":
    unittest.main()
