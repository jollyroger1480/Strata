"""The desktop app's installer: tools/strata_mcp.py and setup.py run on Control's Python in STRATA_CONTROL_INSTALL mode.

What the owner asked for: a user installs the app, picks a model in it and downloads and prepares it, like any normal user.
What must stay true: a start or a serve never acquires anything (tests/terminal/test_no_acquisition.py), and the install
itself downloads model files only - no pip, no git, no compiler, no engine fetch, only the bundled preparation tools.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_setup(home: Path, name: str = "setup_control_mode"):
    """setup.py imported as the app's installer: the mode is read when the module loads."""
    env = {"STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(home)}
    with mock.patch.dict(os.environ, env):
        spec = importlib.util.spec_from_file_location(name, ROOT / "setup.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ControlModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name) / "install"
        self.home.mkdir()
        self.S = load_setup(self.home)

    def test_the_folder_everything_is_written_to_is_the_install_folder_not_the_bundle(self):
        S = self.S
        self.assertTrue(S.CONTROL)
        self.assertEqual(S.home(), self.home.resolve())
        self.assertEqual(S.ROOT, ROOT)
        # without the mode nothing moves: HOME is the source folder, as it always was
        spec = importlib.util.spec_from_file_location("setup_plain_mode", ROOT / "setup.py")
        plain = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "0"}):
            spec.loader.exec_module(plain)
        self.assertFalse(plain.CONTROL)
        self.assertEqual(plain.home(), plain.ROOT)

    def test_only_this_python_running_a_bundled_tool_may_run(self):
        S = self.S
        tool = ROOT / "tools" / "pack_index.py"
        for refused in (
            [sys.executable, "-m", "pip", "install", "numpy"],
            ["pip", "install", "numpy"],
            ["git", "clone", "https://example.invalid/x.git"],
            ["cmake", "--build", "."],
            ["winget", "install", "Nvidia.CUDA"],
            [sys.executable, str(self.home / "evil.py")],                 # a script outside the bundled tools
            [sys.executable, str(ROOT / "setup.py"), "--build"],
            [sys.executable],
        ):
            with self.subTest(refused=refused), mock.patch.object(S.subprocess, "run", side_effect=AssertionError("ran")):
                with self.assertRaises(S.acquisition.AcquisitionRefused):
                    S.run(refused)
        done = subprocess.CompletedProcess([], 0)
        with mock.patch.object(S.subprocess, "run", return_value=done) as runner:
            S.run([sys.executable, str(tool), "--help"], check=False)
        runner.assert_called_once()

    def test_no_package_is_installed(self):
        S = self.S
        with mock.patch.object(S, "run", side_effect=AssertionError("pip")), \
                mock.patch.object(S, "control_missing_packages", return_value=[]):
            S.pip_install(["numpy==1"], "the packages")      # the app ships them; nothing runs
        with mock.patch.object(S, "control_missing_packages", return_value=["numpy"]), \
                self.assertRaises(SystemExit):
            S.pip_install(["numpy==1"], "the packages")

    def test_the_engine_is_the_component_and_a_missing_one_does_not_stop_the_download(self):
        S = self.S
        llama = self.home / "third_party" / "llama.cpp"
        (llama / "ggml").mkdir(parents=True)
        (llama / "ggml" / "CMakeLists.txt").write_text("", encoding="utf-8")
        (llama / "gguf-py").mkdir()
        empty = Path(self.tmp.name) / "components"
        empty.mkdir()
        gpu = {"index": 0, "vram_gb": 12.0}
        with mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": str(empty)}), \
                mock.patch.object(S, "download", side_effect=AssertionError("fetched")), \
                mock.patch.object(S, "build_engine", side_effect=AssertionError("built")), \
                mock.patch.object(S, "get_prebuilt", side_effect=AssertionError("fetched an engine")):
            found, eng, meta, libs, chosen, vision = S.engine_step(mock.Mock(build=False, prebuilt=None), gpu, False, [], 13, "gpu")
        self.assertEqual(found.resolve(), llama.resolve())
        self.assertEqual(libs, [])
        self.assertEqual(vision, "none")                                   # images are not in the first app release
        self.assertEqual(meta["source"], "local")
        self.assertEqual(S.engine_exe(eng), "component:strata-engine")

    def test_the_run_config_goes_to_the_install_folder_and_the_data_folder_sits_beside_it(self):
        S = self.S
        self.assertEqual(S.installed_configs(), [])
        (self.home / "strata-q2_0.json").write_text(json.dumps({"exe": "component:strata-engine", "args": ["--native", "x.gguf"],
                                                                "tokenizer": "t"}), encoding="utf-8")
        with mock.patch.object(S, "model_config", return_value=True):
            self.assertEqual([p.name for p in S.installed_configs()], ["strata-q2_0.json"])
        with mock.patch.dict(os.environ, {"APPDATA": self.tmp.name, "XDG_CONFIG_HOME": self.tmp.name}):
            data, _ = S.data_folder_in_place()
        self.assertEqual(data, self.home.resolve().parent / "Strata-data")

    def test_the_bundle_lists_everything_the_install_imports(self):
        sys.path.insert(0, str(ROOT / "apps" / "desktop" / "scripts"))
        try:
            spec = importlib.util.spec_from_file_location("stage_control_for_test", ROOT / "apps" / "desktop" / "scripts" / "stage-control.py")
            stage = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(stage)
        finally:
            sys.path.pop(0)
        tools = stage.local_imports(ROOT, stage.INSTALLER_TOOLS)
        for need in ("strata_mcp", "iq_pack", "strata_pack", "canonical_xcheck", "pack_layer", "gguf_reader", "gguf_writer", "_paths"):
            self.assertIn(need, tools)
        out = Path(self.tmp.name) / "control"
        out.mkdir()
        record = stage.add_installer(out)
        for name in ("setup.py", "ref/load.py", "tools/strata_mcp.py", "tools/iq_pack.py", "data/draft_vocab.bin"):
            self.assertTrue((out / name).is_file(), name)
        self.assertIn("tools/strata_pack.py", record["files"])
        # every module setup.py's preparation steps run exists in the bundle
        for tool in ("iq_pack", "strata_pack", "pack_index", "strata_tokenizer", "mtp_fetch", "mtp_pack", "mtp_rt"):
            self.assertTrue((out / "tools" / f"{tool}.py").is_file(), tool)


class InstallerServerTests(unittest.TestCase):
    """tools/strata_mcp.py with its source (setup.py, serve/) apart from the install folder, as the app runs it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "install"
        self.root.mkdir()
        spec = importlib.util.spec_from_file_location("strata_mcp_for_test", ROOT / "tools" / "strata_mcp.py")
        self.mcp = importlib.util.module_from_spec(spec)
        with mock.patch.dict(os.environ, {"STRATA_CONTROL_INSTALL": "1", "STRATA_INSTALL_DIR": str(self.root)}):
            spec.loader.exec_module(self.mcp)

    def test_the_catalog_comes_from_setup_py_in_the_bundle(self):
        s = self.mcp.Strata(self.root, ROOT)
        self.assertEqual(s.root, self.root.resolve())
        self.assertEqual(s.src, ROOT)
        models, families, contexts, source = s.tables()
        self.assertEqual(source, "setup.py")
        self.assertIn("qwen", families)
        self.assertIn("IQ3_XXS", models)
        self.assertEqual(s.python_override, sys.executable)               # Control's own Python runs the install

    def test_the_install_job_runs_on_control_python_without_a_venv_and_may_fetch_only_for_itself(self):
        s = self.mcp.Strata(self.root, ROOT)
        spec_path = self.root / "job.json"
        spec = {"job": "j1", "root": str(self.root), "src": str(ROOT), "control": True, "python": sys.executable,
                "base_python": sys.executable, "create_venv": False, "args": ["--yes", "--no-start"],
                "result": str(self.root / "result.json")}
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        seen = {}

        class FakePopen:
            pid = 4242

            def __init__(self, cmd, **kw):
                seen["cmd"], seen["kw"] = cmd, kw

            def wait(self):
                return 0

        with mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1", "PYTHONPATH": "x", "PYTHONHOME": "y"}), \
                mock.patch.object(self.mcp.subprocess, "Popen", FakePopen), \
                mock.patch.object(self.mcp, "proc_identity", return_value=None):
            self.assertEqual(self.mcp.install_job(str(spec_path)), 0)
            self.assertEqual(os.environ["STRATA_NO_ACQUIRE"], "1", "this process keeps the switch")
        self.assertEqual(seen["cmd"][:2], [sys.executable, str(ROOT / "setup.py")])
        env = seen["kw"]["env"]
        self.assertEqual(env["STRATA_CONTROL_INSTALL"], "1")
        self.assertEqual(env["STRATA_INSTALL_DIR"], str(self.root))
        self.assertNotIn("STRATA_NO_ACQUIRE", env, "the explicit install job may fetch the chosen model's files")
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertFalse((self.root / ".venv").exists(), "no virtual environment, no pip")
        self.assertIsNotNone(s)

    def test_a_two_file_model_has_one_combined_progress(self):
        """The bar went 70 GB -> 0 -> up again because each file reported its own total. One total for the whole model now."""
        folder = self.root / "Strata-data" / "models" / "IQ3_S"
        folder.mkdir(parents=True)
        job = {"folder": str(folder), "total_gb": 0.004}
        step5 = {"step": "5/7: downloading Qwen3.8-Flash-Next IQ3_S"}
        seen = []
        # file 1 downloading, then complete, then file 2 starts: the combined figure only goes up
        for name, size in (("a.gguf.part", 400_000), ("a.gguf.part", 1_500_000)):
            (folder / name).write_bytes(b"x" * size)
            seen.append(self.mcp.combined_progress(dict(step5), job)["download"])
        (folder / "a.gguf.part").rename(folder / "a.gguf")
        (folder / "b.gguf.part").write_bytes(b"x" * 100_000)
        seen.append(self.mcp.combined_progress(dict(step5), job)["download"])
        (folder / "b.gguf.part").write_bytes(b"x" * 1_000_000)
        seen.append(self.mcp.combined_progress(dict(step5), job)["download"])
        done = [d["done_gb"] for d in seen]
        self.assertEqual(done, sorted(done), done)
        self.assertEqual(seen[-1]["total_gb"], 0.004)
        self.assertAlmostEqual(seen[-1]["remaining_gb"], 0.004 - seen[-1]["done_gb"], places=2)
        self.assertTrue(all(d["file"] == "model files" for d in seen))
        # not a download any more once setup is preparing the files
        self.assertNotIn("download", self.mcp.combined_progress({"step": "6/7: preparing the model for Strata", "download": {"x": 1}}, job))
        # a resumed install counts what is already on disk
        self.assertGreater(self.mcp.combined_progress(dict(step5), job)["download"]["done_gb"], 0)

    @unittest.skipUnless(os.name == "nt" or sys.platform.startswith("linux"), "a PC with a GPU probe")
    def test_a_real_install_reaches_the_model_download_with_nothing_else_acquired(self):
        """setup.py in the app's mode, as the install job runs it: steps 1-4 pass (PC check, packages bundled, engine component
        optional, llama.cpp source already present) and step 5 asks for the model file. The model host is a folder that does
        not exist, so nothing is downloaded; pip, git, cmake and winget shims that record any call must stay unused."""
        import setup as plain
        if not plain.gpus():
            self.skipTest("no NVIDIA GPU here: setup stops at step 1")
        home = Path(self.tmp.name) / "home"
        shims = Path(self.tmp.name) / "shims"
        for p in (home / "third_party" / "llama.cpp" / "ggml", home / "third_party" / "llama.cpp" / "gguf-py", shims):
            p.mkdir(parents=True)
        (home / "third_party" / "llama.cpp" / "ggml" / "CMakeLists.txt").write_text("", encoding="utf-8")
        log = Path(self.tmp.name) / "shim.log"
        for tool in ("pip", "pip3", "git", "cmake", "ninja", "winget", "curl", "wget", "cl", "make"):
            if os.name == "nt":
                (shims / f"{tool}.cmd").write_text(f"@echo {tool} %* >> \"{log}\"\r\n", encoding="utf-8")
            else:
                (shims / tool).write_text(f"#!/bin/sh\necho {tool} \"$@\" >> '{log}'\n", encoding="utf-8")
                (shims / tool).chmod(0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("STRATA_NO_ACQUIRE", "PYTHONPATH", "VIRTUAL_ENV")}
        env.update(STRATA_CONTROL_INSTALL="1", STRATA_INSTALL_DIR=str(home), APPDATA=str(Path(self.tmp.name) / "appdata"),
                   XDG_CONFIG_HOME=str(Path(self.tmp.name) / "xdg"), STRATA_COMPONENTS_DIR=str(Path(self.tmp.name) / "components"),
                   HF_ENDPOINT=(Path(self.tmp.name) / "no-such-hf").as_uri(), PATH=str(shims) + os.pathsep + env.get("PATH", ""),
                   PYTHONIOENCODING="utf-8")
        done = subprocess.run([sys.executable, str(ROOT / "setup.py"), "--yes", "--no-start", "--family", "qwen", "--model", "Q2_0",
                               "--context", "32768", "--vision", "no", "--experimental-speed-projection", "off"],
                              capture_output=True, text=True, env=env, cwd=str(ROOT), stdin=subprocess.DEVNULL, timeout=300)
        out = done.stdout + done.stderr
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("bundled with the app", out)
        self.assertIn("Step 5: downloading", out)
        self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", out)
        self.assertFalse(log.exists(), "something ran pip, git, cmake or a package manager: " + (log.read_text() if log.exists() else ""))


if __name__ == "__main__":
    unittest.main()
