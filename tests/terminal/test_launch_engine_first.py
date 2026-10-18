"""`strata launch` settles the engine before it touches the model: with no usable engine and none to get, it refuses at once, with the
plain sentence `strata update` gives and one next step, and no model byte is downloaded (the 0.1.45.x tester round downloaded 19 of
37.6 GB first and then failed with the generic line)."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import LaunchError, NO_ENGINE_NEXT, Runner
from strata_control import engine_component as ec

ROOT = Path(__file__).resolve().parents[2]

WRONG_FAMILY = ec.arch_notice(("gfx1100", "gfx1101", "gfx1102"), ["gfx1201"])
USABLE = {"usable": True, "engine": {"version": "0.1.45", "backend": "hip"}, "notice": None}
UNUSABLE = {"usable": False, "engine": None, "notice": "No engine for this PC was found in /x: the component folder holds no engine package. "
                                                       "The app never downloads an engine while it starts or serves."}


class FakeProc:
    def __init__(self, events: list[dict], code: int = 0):
        self.stdout = iter(json.dumps(e) + "\n" for e in events)
        self._code = code

    def wait(self) -> int:
        return self._code


def update_events(status: str, note: str = "", code: int = 0) -> list[dict]:
    res = {"component": "engine", "status": status, "changed": status == "updated", "installed_before": None}
    if note:
        res["note"] = note
    return [{"event": "phase", "phase": "feed", "message": "Reading the signed update feed"},
            {"event": "result", "results": [res], "exit_code": code}]


def runner(spawned: list) -> Runner:
    r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=mock.Mock(), emit=lambda e: None)
    r.check_cancel = lambda: None
    return r


class EngineBeforeModel(unittest.TestCase):
    def run_ensure(self, statuses: list[dict], events: list[dict], code: int = 0):
        """ensure_engine() with a fake `strata update` and a sequence of engine reports. Returns (spawn count, the error or None)."""
        spawns: list = []
        r = runner(spawns)

        def spawn(argv, **kw):
            spawns.append(argv)
            return FakeProc(events, code)

        err = None
        with mock.patch.object(ec, "status", side_effect=statuses), mock.patch.object(Runner, "_spawn", side_effect=lambda argv, **kw: spawn(argv, **kw)):
            try:
                r.ensure_engine()
            except LaunchError as e:
                err = e
        return spawns, err

    def test_engine_present_runs_nothing(self):
        spawns, err = self.run_ensure([USABLE], [])
        self.assertIsNone(err)
        self.assertEqual(spawns, [])

    def test_no_engine_and_the_feed_has_one(self):
        spawns, err = self.run_ensure([UNUSABLE, USABLE], update_events("updated"))
        self.assertIsNone(err)
        self.assertEqual(len(spawns), 1)
        self.assertIn("update", spawns[0])

    def test_no_engine_and_the_feed_has_none(self):
        note = "The feed has no engine package for this PC."
        spawns, err = self.run_ensure([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", note))
        self.assertIsNotNone(err)
        self.assertEqual(err.code, 5)
        self.assertEqual(err.text, note)
        self.assertEqual(err.next, NO_ENGINE_NEXT)
        self.assertNotIn("The app never downloads", str(err))          # not the generic line

    def test_wrong_family_says_the_plain_line(self):
        spawns, err = self.run_ensure([UNUSABLE, UNUSABLE], update_events("none-for-this-pc", WRONG_FAMILY))
        self.assertIsNotNone(err)
        self.assertEqual(err.code, 5)
        self.assertEqual(err.text, WRONG_FAMILY)
        self.assertIn("This build of Strata is for AMD RX 7000 cards. This computer has an AMD RX 9000 card (gfx1201).", err.text)
        self.assertIn("Ask for the RX 9000 build.", err.text)
        self.assertIn("Nothing was downloaded", err.next)

    def test_installed_wrong_family_package_uses_its_plain_notice(self):
        report = {"usable": False, "engine": None, "notice": WRONG_FAMILY}
        spawns, err = self.run_ensure([report, report], update_events("none-for-this-pc"))
        self.assertEqual(err.text, WRONG_FAMILY)

    def test_the_model_is_never_touched_when_the_engine_is_refused(self):
        r = runner([])
        with mock.patch.object(ec, "status", return_value=UNUSABLE), \
                mock.patch.object(Runner, "_spawn", return_value=FakeProc(update_events("none-for-this-pc", WRONG_FAMILY))), \
                mock.patch.object(Runner, "ensure_model", side_effect=AssertionError("model step reached")) as model:
            with self.assertRaises(LaunchError) as ctx:
                r.run()
        self.assertEqual(ctx.exception.text, WRONG_FAMILY)
        model.assert_not_called()

    def test_a_failed_update_still_reports_the_old_message(self):
        spawns, err = self.run_ensure([UNUSABLE], [{"event": "error", "error": {"message": "cannot reach the feed"}}], code=6)
        self.assertEqual(err.code, 6)
        self.assertIn("cannot reach the feed", err.text)


class SetupControlStep(unittest.TestCase):
    """setup.py in the app's mode (and under `strata launch`): the engine step comes before the download step."""

    def load_setup(self):
        """A private copy of setup.py read in the app's mode; the shared `setup` module (other tests) is left as it is."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_control_copy", ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "1"}):
            spec.loader.exec_module(module)
        return module

    def step(self, exc: Exception, env: dict) -> tuple[object, str]:
        setup = self.load_setup()
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), mock.patch.object(ec, "resolve", side_effect=exc), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("llama")), \
                mock.patch.object(setup, "say", lambda *a, **k: out.write(" ".join(map(str, a)) + "\n")), \
                mock.patch.object(setup, "warn", lambda *a, **k: out.write("WARN " + " ".join(map(str, a)) + "\n")), \
                mock.patch.object(setup, "ok", lambda *a, **k: None):
            try:
                return setup.control_engine_step({}, "none"), out.getvalue()
            except SystemExit as e:
                return e, out.getvalue()

    def test_wrong_family_stops_before_the_download(self):
        exc = ec.EngineMissing("nothing matched", Path("/c"), notice=WRONG_FAMILY)
        res, out = self.step(exc, {})
        self.assertIsInstance(res, SystemExit)
        self.assertIn("Ask for the RX 9000 build.", out)
        self.assertIn("nothing was downloaded", out)

    def test_no_engine_stops_before_the_download_for_launch(self):
        res, out = self.step(ec.EngineMissing("the component folder holds no engine package", Path("/c")), {"STRATA_NEED_ENGINE": "1"})
        self.assertIsInstance(res, SystemExit)

    def test_the_app_without_an_engine_keeps_its_order(self):
        env = {k: v for k, v in os.environ.items() if k != "STRATA_NEED_ENGINE"}
        with mock.patch.dict(os.environ, env, clear=True):
            res, out = self.step(ec.EngineMissing("the component folder holds no engine package", Path("/c")), {})
        self.assertNotIsInstance(res, SystemExit)
        self.assertIn("The model is downloaded and prepared now", out)


if __name__ == "__main__":
    unittest.main()
