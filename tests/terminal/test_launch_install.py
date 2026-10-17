"""The one-line installers and the local test feed: syntax, the placeholder guard, the feed builder, and (POSIX) a real install.sh run against a feed."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))


class Scripts(unittest.TestCase):
    def test_install_sh_syntax_and_guard(self):
        sh = shutil.which("sh") or shutil.which("bash")
        if not sh:
            self.skipTest("no sh")
        r = subprocess.run([sh, "-n", str(ROOT / "packaging" / "install.sh")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (ROOT / "packaging" / "install.sh").read_text(encoding="utf-8")
        self.assertEqual(text.count("__FEED__"), 1)                       # the baked address replaces only the default; the guard survives
        self.assertIn("Installed. Run: ", text)
        self.assertNotIn("\r", text)

    def test_install_ps1_parses_and_guard(self):
        text = (ROOT / "packaging" / "install.ps1").read_text(encoding="utf-8")
        self.assertEqual(text.count("__FEED__"), 1)
        self.assertIn("Installed. Run: strata launch", text)
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if not ps:
            self.skipTest("no PowerShell")
        cmd = ("$e=$null; $t=$null; [System.Management.Automation.Language.Parser]::ParseFile('%s',[ref]$t,[ref]$e) | Out-Null; "
               "if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 } else { 'ok' }" % str(ROOT / "packaging" / "install.ps1").replace("'", "''"))
        r = subprocess.run([ps, "-NoProfile", "-Command", cmd], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_a_script_without_an_address_says_what_to_do(self):
        sh = shutil.which("sh") or shutil.which("bash")
        if not sh or os.name == "nt":
            self.skipTest("POSIX sh only")
        env = {k: v for k, v in os.environ.items() if k != "STRATA_FEED"}
        r = subprocess.run([sh, str(ROOT / "packaging" / "install.sh")], capture_output=True, text=True, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Set STRATA_FEED", r.stderr)


class Feed(unittest.TestCase):
    def make_bundle(self, work: Path) -> Path:
        """A tiny terminal archive that is a working `strata` (the repo's own source with the system Python)."""
        top = work / "bundle" / "strata-terminal-9.9.9-linux-x86_64"
        top.mkdir(parents=True)
        for name in ("strata.py", "strata_cli", "strata_control", "serve"):
            src = ROOT / name
            (shutil.copytree(src, top / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")) if src.is_dir() else shutil.copy2(src, top / name))
        (top / "bin").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "bin" / "strata", top / "bin" / "strata")
        (top / "bin" / "strata").chmod(0o755)
        (top / "VERSION").write_text("9.9.9\n", encoding="utf-8")
        archive = work / "strata-terminal-9.9.9-linux-x86_64.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(top, arcname=top.name)
        return archive

    def test_make_local_feed_and_a_real_install(self):
        import make_local_feed
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = self.make_bundle(work)
            feed = work / "feed"
            rc = make_local_feed.main(["--out", str(feed), "--terminal", str(archive), "--keys", str(work / "keys"), "--url", "http://127.0.0.1:1"])
            self.assertEqual(rc, 0)
            for name in ("strata-feed.json", "strata-feed.json.sig", "install-manifest.txt", "install.sh", "install.ps1", "test-trust.json"):
                self.assertTrue((feed / name).is_file(), name)
            line = (feed / "install-manifest.txt").read_text().split()
            self.assertEqual(line[0], "linux-x86_64")
            self.assertEqual(line[1], "9.9.9")
            control = feed / line[2]
            self.assertEqual(hashlib.sha256(control.read_bytes()).hexdigest(), line[3])
            self.assertEqual(control.stat().st_size, int(line[4]))
            sh_text = (feed / "install.sh").read_text()
            self.assertIn("http://127.0.0.1:1", sh_text)
            self.assertIn('"__FE""ED__"', sh_text)                        # the guard was not rewritten
            self.assertIn("Installed. Run:", (feed / "install.ps1").read_text())
            doc = json.loads((feed / "strata-feed.json").read_text())
            self.assertEqual(doc["control"]["packages"][0]["sha256"], line[3])
            self.assertNotIn("seed", (feed / "test-trust.json").read_text())      # only the public key is published
            if os.name == "nt" or not (shutil.which("sh") and shutil.which("curl") and shutil.which("tar")):
                self.skipTest("the real install.sh run needs POSIX sh, curl and tar")
            handler = partial(SimpleHTTPRequestHandler, directory=str(feed))
            handler.log_message = lambda *a, **k: None                                   # type: ignore[attr-defined]
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                url = f"http://127.0.0.1:{server.server_address[1]}"
                home = work / "home"
                home.mkdir()
                env = {**os.environ, "HOME": str(home), "STRATA_FEED": url, "STRATA_INSTALL_TEST_TRUST": "1", "STRATA_PYTHON": sys.executable,
                       "PATH": os.environ["PATH"], "STRATA_NO_ACQUIRE": ""}
                for k in ("XDG_DATA_HOME", "STRATA_UPDATE_TRUSTED_KEYS"):
                    env.pop(k, None)
                r = subprocess.run(["sh", str(feed / "install.sh")], capture_output=True, text=True, env=env, timeout=240)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                self.assertIn("Checksum ok", r.stdout)
                self.assertIn("Signature ok", r.stdout)
                self.assertIn("Installed. Run:", r.stdout)
                self.assertTrue((home / ".local" / "bin" / "strata").exists())
                v = subprocess.run([str(home / ".local" / "bin" / "strata"), "--version"], capture_output=True, text=True, env=env, timeout=60)
                self.assertEqual(v.returncode, 0, v.stderr)
                # a damaged download is refused and nothing is installed
                data = (feed / line[2]).read_bytes()
                (feed / line[2]).write_bytes(data[:-10] + b"x" * 10)
                home2 = work / "home2"
                home2.mkdir()
                r = subprocess.run(["sh", str(feed / "install.sh")], capture_output=True, text=True, env={**env, "HOME": str(home2)}, timeout=240)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("does not match the release's checksum", r.stderr)
                self.assertFalse((home2 / ".local" / "share" / "strata" / "terminal" / "9.9.9").exists())
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
