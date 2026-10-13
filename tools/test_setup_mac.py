"""Tests for setup on a Mac (metal/setup_mac.py and setup.py's hand-over): the Mac check, the pack recommended by
unified memory, what a bigger choice risks, the engine arguments with the Metal defaults and the config. Runs on any
PC: sysctl / sw_vers / xcode-select are mocked. No GPU, no network.

    python -m unittest tools.test_setup_mac
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "metal"))
import setup  # noqa: E402
import setup_mac as M  # noqa: E402


def fake_run(values: dict, clt_rc=0):
    def run(cmd, **kw):
        if cmd[0] == "sysctl":
            return subprocess.CompletedProcess(cmd, 0, values.get(cmd[2], ""), "")
        if cmd[0] == "sw_vers":
            return subprocess.CompletedProcess(cmd, 0, values.get("macos", ""), "")
        if cmd[0] == "xcode-select":
            return subprocess.CompletedProcess(cmd, clt_rc, "/Library/Developer/CommandLineTools" if clt_rc == 0 else "",
                                               "")
        raise AssertionError(cmd)
    return run


M5 = {"machdep.cpu.brand_string": "Apple M5 Max", "hw.memsize": str(128 * 2 ** 30), "macos": "27.2",
      "hw.perflevel0.physicalcpu": "12"}


class MacCheck(unittest.TestCase):
    def test_reads_chip_memory_macos_and_tools(self):
        with mock.patch.object(M.platform, "machine", return_value="arm64"):
            info = M.mac_info(fake_run(M5))
        self.assertEqual(info["chip"], "Apple M5 Max")
        self.assertAlmostEqual(info["ram_gb"], 128.0)
        self.assertEqual(info["macos"], "27.2")
        self.assertTrue(info["arm64"])
        self.assertTrue(info["clt"])
        self.assertEqual(M.mac_problems(info), [])

    def test_intel_old_macos_and_missing_tools_are_named(self):
        with mock.patch.object(M.platform, "machine", return_value="x86_64"):
            info = M.mac_info(fake_run({**M5, "macos": "14.6", "machdep.cpu.brand_string": "Intel Core i9"}, clt_rc=2))
        p = " | ".join(M.mac_problems(info))
        self.assertIn("Apple Silicon", p)
        self.assertIn("macOS 15", p)
        self.assertIn("xcode-select --install", p)


class Recommendation(unittest.TestCase):
    def test_by_memory(self):
        self.assertEqual(M.recommend_model(128)[0], "IQ3_S")
        self.assertEqual(M.recommend_model(96)[0], "IQ3_XXS")
        self.assertEqual(M.recommend_model(64)[0], "Q2_0")
        rec, notes = M.recommend_model(32)
        self.assertEqual(rec, "Q2_0")
        self.assertTrue(notes and "stream" in notes[0] and "~14 tok/s" in notes[0])
        self.assertTrue(M.streams("Q2_0", 32) and M.streams("Q2_0", 48) and not M.streams("Q2_0", 64))

    def test_a_bigger_choice_is_kept_with_a_note(self):
        notes = M.model_note("IQ3_XXS", 64)
        self.assertTrue(any("stream from the SSD" in n for n in notes))
        self.assertEqual(M.model_note("Q2_0", 128), [])
        self.assertEqual(M.model_note("IQ3_S", 128), [])

    def test_context(self):
        self.assertEqual(M.recommend_context(64), 32768)
        self.assertEqual(M.recommend_context(128), 65536)


class Config(unittest.TestCase):
    def test_engine_args_carry_the_metal_defaults(self):
        args = M.engine_args(Path("/p/packs/q2_0"), [Path("/m/a-00001.gguf"), Path("/m/a-00002.gguf")],
                             Path("/d/mtp/rt"), 32768)
        self.assertEqual(args[args.index("--spec") + 1], "2")
        self.assertEqual(args[args.index("--native") + 1], str(Path("/m/a-00001.gguf")))
        self.assertEqual(args[args.index("--ple-gguf") + 1], str(Path("/m/a-00002.gguf")))
        self.assertEqual(args[args.index("--mtp") + 1], str(Path("/d/mtp/rt")))
        self.assertIn("--expert-profile", args)
        self.assertNotIn("--hc-q8", args)                         # Q8_0 HC is the engine's own default on Metal
        st = M.engine_args(Path("/p"), [Path("/a1.gguf"), Path("/a2.gguf")], Path("/d/mtp/rt"), 32768, stream=True)
        self.assertNotIn("--mtp", st)                             # streaming: one token per step, no draft layer
        self.assertEqual(st[st.index("--suffix-draft") + 1], "0")

    def test_config_is_a_model_config_with_the_metal_backend(self):
        cfg = M.mac_config(Path("/r/build-metal/strata"), ["--pack", "x"], Path("/p"), "qwen3.8-flash-next-q2_0",
                           "Q2_0", 8080)
        self.assertEqual(cfg["backend"], "metal")
        self.assertEqual(cfg["tokenizer"], str(Path("/p") / "tokenizer"))
        self.assertTrue(cfg["exe"] and isinstance(cfg["args"], list))


class Vision(unittest.TestCase):
    def test_the_projector_is_pinned_and_the_encoder_builds_with_metal(self):
        self.assertIn("mmproj-Qwen3.8-Flash-Next-BF16.gguf", M.VISION_SHA256)
        src = (ROOT / "metal" / "setup_mac.py").read_text(encoding="utf-8")
        self.assertIn('"-DSTRATA_VISION_CUDA=OFF"', src)
        self.assertIn('cfg["args"] += ["--vision"]', src)


class ExpertsFile(unittest.TestCase):
    """The streaming Mac's <model>.experts: written only with room on the disk, else a warning and the GGUF."""

    def run_it(self, free_gb, done=False):
        import tempfile
        sys.path.insert(0, str(ROOT / "metal" / "tools"))
        import stream_experts
        d = Path(tempfile.mkdtemp())
        native = d / "m.gguf"
        native.write_bytes(b"")
        if done:
            (d / "m.gguf.experts").write_bytes(b"")
            (d / "m.gguf.experts.done").write_text("x")
        with mock.patch.object(stream_experts, "size_bytes", return_value=40e9), \
                mock.patch.object(M.S, "free_gb", return_value=free_gb), \
                mock.patch.object(M.S, "run") as run, mock.patch.object(M.S, "warn") as warn, \
                mock.patch.object(M.S, "mark") as mark, mock.patch.object(M.S, "say"), mock.patch.object(M.S, "ok"):
            M.experts_file(native)
        return run, warn, mark

    def test_fits_needs_the_margin(self):
        self.assertFalse(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB - 0.1))
        self.assertTrue(M.experts_file_fits(40, 40 + M.EXPERTS_MARGIN_GB))

    def test_written_and_verified_with_room(self):
        run, warn, mark = self.run_it(100)
        self.assertIn("--verify", run.call_args[0][0])
        self.assertTrue(str(run.call_args[0][0][1]).endswith("stream_experts.py"))
        mark.assert_called_once()
        warn.assert_not_called()

    def test_too_little_disk_warns_and_goes_on(self):
        run, warn, mark = self.run_it(45)
        run.assert_not_called()
        mark.assert_not_called()
        self.assertIn("not written", warn.call_args[0][0])

    def test_done_is_not_written_again(self):
        run, warn, mark = self.run_it(100, done=True)
        run.assert_not_called()


class HandOver(unittest.TestCase):
    def test_darwin_hands_the_run_to_setup_mac(self):
        with mock.patch.object(setup.sys, "platform", "darwin"), \
                mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--model", "Q2_0", "--yes"]):
            self.assertEqual(setup.main(), 0)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("metal") / "setup_mac.py")))
        self.assertEqual(cmd[2:], ["--model", "Q2_0", "--yes"])

    def test_other_platforms_never_reach_it(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn('if sys.platform == "darwin":', src)
        with mock.patch.object(setup, "mac_setup", side_effect=AssertionError("called")), \
                mock.patch.object(setup.sys, "platform", "linux"), \
                mock.patch.object(setup.sys, "argv", ["setup.py", "--inspect", "x"]), \
                mock.patch.object(setup.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(SystemExit):
                setup.main()

    def test_setup_sh_fetches_a_pinned_python_on_macos_only(self):
        sh = (ROOT / "setup.sh").read_text(encoding="utf-8")
        self.assertIn('[ "$(uname -s)" = Darwin ]', sh)
        self.assertIn("83dcb04777c798bd95f3687acf9ae84b09860272ee6bdfb05be8bcf9447f75b8", sh)


if __name__ == "__main__":
    unittest.main()
