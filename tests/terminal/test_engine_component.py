"""Engine resolution from a local component folder: which package is chosen, why one is refused, and that nothing is
ever fetched. Every folder here is a throwaway; no GPU tool is run (the backend is always given or forced)."""

import contextlib
import hashlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from strata_cli import cli
from strata_control import acquisition, components, engine_component as ec

WINDOWS = os.name == "nt"
HERE = components.host_platform()
EXE = "strata.exe" if WINDOWS else "strata"


def make_package(root: Path, *, version="0.1.45", platform=None, backend="cuda", archs=(), min_runtime=None,
                 extra=None, entry=None) -> Path:
    """A fake engine package the way packaging/package_engine.py lays one out."""
    platform = platform or HERE
    entry = entry or EXE
    folder = root / "engine" / f"{version}-{platform}-{backend}"
    folder.mkdir(parents=True)
    payload = {entry: b"MZ-not-a-real-engine-" + version.encode(), "cudart64_13.dll" if WINDOWS else "libcudart.so.13":
              b"runtime-library-bytes" * 50, "BUILD.json": json.dumps({"version": version}).encode()}
    payload.update(extra or {})
    files = []
    for name, data in payload.items():
        (folder / name).write_bytes(data)
        files.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    if not WINDOWS:
        (folder / entry).chmod(0o755)
    manifest = {"schema_version": 1, "id": "strata-engine", "version": version, "platform": platform,
                "backend": backend, "entrypoint": entry, "files": files}
    if archs:
        manifest["cuda_archs"] = list(archs)
    if min_runtime:
        manifest["min_runtime"] = min_runtime
    (folder / ec.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def rewrite(folder: Path, **changes):
    path = folder / ec.MANIFEST
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


class NoNetwork:
    """Any attempt to reach the network or start an installer fails the test that is running."""

    def __enter__(self):
        def refuse(*a, **k):
            raise AssertionError("an engine lookup tried to reach the network")
        self.patches = [mock.patch.object(urllib.request, "urlopen", refuse),
                        mock.patch.object(socket, "create_connection", refuse),
                        mock.patch.object(socket.socket, "connect", refuse)]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in self.patches:
            item.stop()


class Resolution(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)

    def test_the_package_for_the_backend_is_chosen(self):
        make_package(self.root, backend="cuda")
        make_package(self.root, backend="hip", version="0.1.46")
        chosen = ec.resolve(self.root, backend="cuda", runtime="0.1.45")
        self.assertEqual((chosen.backend, chosen.version), ("cuda", "0.1.45"))
        self.assertEqual(chosen.exe, self.root / "engine" / f"0.1.45-{HERE}-cuda" / EXE)
        self.assertTrue(chosen.exe.is_file())

    def test_the_newest_compatible_version_wins(self):
        make_package(self.root, version="0.1.45")
        make_package(self.root, version="0.1.45.2")
        make_package(self.root, version="0.1.9")                 # 0.1.9 < 0.1.45 numerically, not as text
        self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").version, "0.1.45.2")

    def test_nothing_installed_is_a_notice_that_says_nothing_is_fetched(self):
        with NoNetwork():
            with self.assertRaises(ec.EngineMissing) as caught:
                ec.resolve(self.root, backend="cuda")
        text = str(caught.exception)
        self.assertIn("never downloads an engine", text)
        self.assertIn("strata install", text)
        self.assertIsInstance(caught.exception, acquisition.AcquisitionRefused)
        self.assertEqual(caught.exception.exit_code, 5)
        self.assertFalse((self.root / "engine").exists(), "looking must not create the folder")

    def test_a_missing_component_folder_is_the_same_notice(self):
        with self.assertRaises(ec.EngineMissing):
            ec.resolve(self.root / "does-not-exist", backend="cuda")

    def test_another_platform_is_refused_with_the_reason(self):
        other = "linux-x86_64" if WINDOWS else "windows-x86_64"
        make_package(self.root, platform=other)
        with self.assertRaises(ec.EngineMissing) as caught:
            ec.resolve(self.root, backend="cuda")
        self.assertIn(f"is for {other}", str(caught.exception))

    def test_another_backend_is_refused_with_the_reason(self):
        make_package(self.root, backend="hip")
        with self.assertRaises(ec.EngineMissing) as caught:
            ec.resolve(self.root, backend="cuda")
        self.assertIn("is a hip engine, this PC needs cuda", str(caught.exception))

    def test_an_unqualified_combination_is_not_attempted(self):
        if HERE == "macos-arm64":
            self.skipTest("metal is supported here")
        make_package(self.root, backend="metal")                 # metal is macOS arm64 only
        with self.assertRaises(ec.EngineMissing) as caught:
            ec.resolve(self.root, backend="metal", target=HERE)
        self.assertIn("not a supported combination", str(caught.exception))

    def test_runtime_too_old_is_refused(self):
        make_package(self.root, min_runtime="0.2.0")
        with self.assertRaises(ec.EngineMissing) as caught:
            ec.resolve(self.root, backend="cuda", runtime="0.1.45")
        self.assertIn("needs control 0.2.0", str(caught.exception))
        self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="0.2.0").version, "0.1.45")

    def test_cuda_archs_must_cover_the_gpu(self):
        make_package(self.root, archs=("sm_86",))
        with mock.patch.object(ec, "cuda_archs_present", return_value=["sm_120"]):
            with self.assertRaises(ec.EngineMissing) as caught:
                ec.resolve(self.root, backend="cuda", runtime="1")
        self.assertIn("sm_86", str(caught.exception))
        with mock.patch.object(ec, "cuda_archs_present", return_value=["sm_86"]):
            self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").version, "0.1.45")
        with mock.patch.object(ec, "cuda_archs_present", return_value=[]):     # cannot tell: not a refusal
            self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").version, "0.1.45")


class Integrity(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)
        self.folder = make_package(self.root)

    def test_a_truncated_library_is_refused(self):
        lib = next(p for p in self.folder.iterdir() if "cudart" in p.name)
        lib.write_bytes(lib.read_bytes()[:-3])
        with self.assertRaisesRegex(ec.EngineMissing, "bytes, the manifest says"):
            ec.resolve(self.root, backend="cuda", runtime="1")

    def test_a_changed_entrypoint_is_refused_at_once(self):
        exe = self.folder / EXE
        exe.write_bytes(b"X" * exe.stat().st_size)               # same size, other bytes
        with self.assertRaisesRegex(ec.EngineMissing, "SHA-256"):
            ec.resolve(self.root, backend="cuda", runtime="1")

    def test_a_changed_library_is_found_by_the_full_check_only(self):
        lib = next(p for p in self.folder.iterdir() if "cudart" in p.name)
        lib.write_bytes(b"Y" * lib.stat().st_size)
        self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").version, "0.1.45")
        with self.assertRaisesRegex(ec.EngineMissing, "SHA-256"):
            ec.resolve(self.root, backend="cuda", runtime="1", full=True)

    def test_a_missing_file_is_refused(self):
        next(p for p in self.folder.iterdir() if "cudart" in p.name).unlink()
        with self.assertRaisesRegex(ec.EngineMissing, "is missing"):
            ec.resolve(self.root, backend="cuda", runtime="1")

    def test_manifest_paths_cannot_leave_the_folder(self):
        data = json.loads((self.folder / ec.MANIFEST).read_text())
        for bad in ("../outside.bin", "/etc/passwd", "C:\\Windows\\win.ini", "sub/../../x"):
            with self.subTest(path=bad):
                broken = dict(data, files=data["files"] + [{"path": bad, "sha256": "0" * 64, "bytes": 1}])
                (self.folder / ec.MANIFEST).write_text(json.dumps(broken), encoding="utf-8")
                with self.assertRaises(ec.ComponentProblem):
                    ec.read_manifest(self.folder)

    def test_an_entrypoint_outside_the_file_list_is_refused(self):
        rewrite(self.folder, entrypoint="other.exe")
        with self.assertRaisesRegex(ec.ComponentProblem, "not in the file list"):
            ec.read_manifest(self.folder)

    def test_a_broken_manifest_does_not_hide_a_good_package(self):
        broken = self.root / "engine" / "0.9-broken"
        broken.mkdir()
        (broken / ec.MANIFEST).write_text("{not json", encoding="utf-8")
        chosen = ec.resolve(self.root, backend="cuda", runtime="1")
        self.assertEqual(chosen.version, "0.1.45")
        report = ec.status(self.root, backend="cuda")
        self.assertTrue(report["usable"])
        self.assertEqual(len(report["problems"]), 1)

    @unittest.skipIf(WINDOWS, "symlinks need a privilege on Windows")
    def test_a_symlinked_file_is_refused(self):
        lib = next(p for p in self.folder.iterdir() if "cudart" in p.name)
        target = self.root / "elsewhere.bin"
        target.write_bytes(lib.read_bytes())
        lib.unlink()
        lib.symlink_to(target)
        with self.assertRaisesRegex(ec.EngineMissing, "not a regular file"):
            ec.resolve(self.root, backend="cuda", runtime="1")

    @unittest.skipIf(WINDOWS, "NTFS has no execute bit")
    def test_a_non_executable_entrypoint_is_refused(self):
        (self.folder / EXE).chmod(0o644)
        with self.assertRaisesRegex(ec.EngineMissing, "not executable"):
            ec.resolve(self.root, backend="cuda", runtime="1")


class Where(unittest.TestCase):
    def test_environment_wins_then_the_per_user_folder(self):
        self.assertEqual(ec.components_dir({"STRATA_COMPONENTS_DIR": "/x/y"}), Path("/x/y"))
        win = ec.components_dir({"LOCALAPPDATA": "C:\\L"}, windows=True)
        self.assertEqual(win.name, "components")
        self.assertEqual(win.parent.name, "Strata")
        posix = ec.components_dir({"XDG_DATA_HOME": "/d"}, windows=False)
        self.assertEqual(posix.as_posix(), "/d/strata/components")

    def test_backend_override_is_validated(self):
        self.assertEqual(ec.detect_backend({"STRATA_ENGINE_BACKEND": "HIP"}), ("hip", "environment"))
        with self.assertRaises(ec.ComponentProblem):
            ec.detect_backend({"STRATA_ENGINE_BACKEND": "vulkan"})

    def test_detection_runs_only_read_only_tools(self):
        calls = []

        def fake_run(command, **kw):
            calls.append(Path(command[0]).stem.lower())
            return subprocess.CompletedProcess(command, 0, stdout="NVIDIA GeForce RTX 3060\n", stderr="")
        with mock.patch.object(ec.shutil, "which", lambda name: name), \
                mock.patch.object(ec.subprocess, "run", fake_run), \
                mock.patch.object(ec, "host_platform", return_value="linux-x86_64"):
            self.assertEqual(ec.detect_backend({}), ("cuda", "nvidia-smi"))
        self.assertEqual(calls, ["nvidia-smi"])


class NeverAcquires(unittest.TestCase):
    """The lookup, the status report and the CLI work with the network and process creation unavailable."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_status_and_cli_with_the_network_down(self):
        make_package(self.root)
        with NoNetwork(), mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            report = ec.status(self.root, backend="cuda")
            code, out, _ = self.run_cli("components", "engine", "--components-dir", str(self.root),
                                        "--backend", "cuda", "--output", "json")
        self.assertTrue(report["usable"])
        self.assertFalse(report["acquires"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertEqual(Path(data["engine"]["exe"]).name, EXE)

    def test_cli_missing_engine_is_exit_5_with_the_notice(self):
        with NoNetwork():
            code, out, _ = self.run_cli("components", "engine", "--components-dir", str(self.root),
                                        "--backend", "cuda", "--output", "json")
        self.assertEqual(code, 5)
        data = json.loads(out)
        self.assertFalse(data["ok"])
        self.assertIn("never downloads an engine", data["notice"])

    def test_a_driver_too_old_for_every_engine_gets_one_plain_line_naming_the_version(self):
        a = make_package(self.root, version="0.1.45", backend="cuda")
        rewrite(a, min_driver={"nvidia": "580.65"})
        with mock.patch.object(ec, "nvidia_driver_version", return_value="550.120"):
            with self.assertRaises(ec.EngineMissing) as caught:
                ec.resolve(self.root, backend="cuda", runtime="1")
            note = ec.status(self.root, backend="cuda")["notice"]
        self.assertIn("580.65", str(caught.exception), "the technical reason stays for the log")
        self.assertIn("(550.120)", note)
        self.assertIn("driver 580 or newer", note)
        self.assertNotIn("component", note)
        # two builds: the line names the lower requirement; a driver that runs one of them is no problem
        folder = self.root / "engine" / "0.1.45-twelve"
        folder.mkdir(parents=True)
        for item in a.iterdir():
            (folder / item.name).write_bytes(item.read_bytes())
        (folder / "strata").chmod(0o755) if not WINDOWS else None
        rewrite(folder, min_driver={"nvidia": "575.51"})
        with mock.patch.object(ec, "nvidia_driver_version", return_value="550.120"):
            self.assertIn("driver 575 or newer", ec.status(self.root, backend="cuda")["notice"])
        with mock.patch.object(ec, "nvidia_driver_version", return_value="576.02"):
            self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").root.name, "0.1.45-twelve")
        with mock.patch.object(ec, "nvidia_driver_version", return_value="580.178.04"):
            self.assertEqual(ec.resolve(self.root, backend="cuda", runtime="1").root.name, a.name, "the newer-driver build wins")

    def test_the_component_fetcher_refuses_where_acquiring_is_forbidden(self):
        store = components.ComponentStore(self.root / "store")
        artifact = components.Artifact(name="engine.tar.gz", url="https://example.invalid/e.tgz",
                                       sha256="0" * 64, bytes=10)
        with NoNetwork(), mock.patch.dict(os.environ, {"STRATA_NO_ACQUIRE": "1"}):
            with self.assertRaises(acquisition.AcquisitionRefused):
                store._fetch(artifact)

    def test_main_guard_turns_the_notice_into_exit_5(self):
        def start():
            ec.resolve(self.root, backend="cuda")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = acquisition.main_guard(start)
        self.assertEqual(code, 5)
        self.assertIn("No engine for this PC", err.getvalue())


if __name__ == "__main__":
    unittest.main()
