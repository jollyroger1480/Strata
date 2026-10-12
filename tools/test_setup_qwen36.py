"""Tests for setup.py's Qwen3.6-35B-A3B entries (NEW-MODELS-PLAN.md T0): a family setup knows and downloads (pinned
revision, size and SHA-256), which the engine does not run yet.  Mocked downloads - nothing is fetched.

    python -m unittest tools.test_setup_qwen36
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")
REPO = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"


def args(**kw):
    base = dict(family="qwen36", model=None, check=False, yes=True, vision=None, models_dir=None)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Qwen36Entries(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.PREVIEW_FAMILIES["qwen36"]
        self.assertEqual(setup.HF_REVISIONS[REPO], "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d")
        for key in ("hf", "mmproj_hf"):
            self.assertRegex(fam[key], SHA)
            self.assertIn(REPO, fam[key])
        self.assertEqual(fam["arch"], "qwen35moe")
        self.assertEqual(list(fam["models"]), ["UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"])
        for m in fam["models"]:
            size, sha = fam["sha256"][fam["file"].format(q=m)]
            self.assertGreater(size, 20e9)
            self.assertRegex(sha, r"^[0-9a-f]{64}$")
        self.assertEqual(fam["sha256"]["Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"][0], 22853663008)
        self.assertIn(fam["mmproj"], fam["sha256"])

    def test_not_a_regular_family(self):
        # the menu, the installed configs and the packs never see it: it is not in FAMILIES
        self.assertNotIn("qwen36", setup.FAMILIES)
        for m, d in setup.MODELS.items():
            self.assertNotIn("qwen36", d.get("families", ()))


class PreviewFamily(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.calls = []
        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha: self.calls.append(("sha", s.name, size, sha))),
                   mock.patch.object(setup, "ram_gb", lambda: 64.0),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_downloads_the_pinned_file_and_checks_it(self):
        rc, out = run(setup.preview_family, args(model="UD-Q4_K_XL", models_dir=self.tmp.name))
        self.assertEqual(rc, 0)
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual(len(get), 1)
        self.assertEqual(get[0][1], setup.hf(REPO) + "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        self.assertEqual(get[0][2], Path(self.tmp.name) / "qwen36-UD-Q4_K_XL" / "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        self.assertIn(("sha", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", 22853663008,
                       "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"), self.calls)
        self.assertIn("cannot run it yet", out)
        self.assertIn("does not run Qwen3.6-35B-A3B yet", out)

    def test_vision_adds_the_encoder(self):
        rc, _ = run(setup.preview_family, args(model="Q8_0", vision="yes", models_dir=self.tmp.name))
        self.assertEqual(rc, 0)
        names = [c[2].name for c in self.calls if c[0] == "get"]
        self.assertEqual(names, ["Qwen3.6-35B-A3B-Q8_0.gguf", "mmproj-F16.gguf"])

    def test_check_lists_without_downloading(self):
        rc, out = run(setup.preview_family, args(check=True, models_dir=self.tmp.name))
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls, [])
        self.assertIn("UD-Q6_K_XL", out)
        self.assertIn("32.6 GB", out)

    def test_a_flash_next_size_is_refused(self):
        rc, out = run(setup.preview_family, args(model="IQ3_XXS", models_dir=self.tmp.name))
        self.assertEqual(rc, 1)
        self.assertIn("has no IQ3_XXS", out)
        self.assertEqual(self.calls, [])

    def test_too_little_disk(self):
        with mock.patch.object(setup, "free_gb", lambda p: 10.0):
            rc, out = run(setup.preview_family, args(model="Q8_0", models_dir=self.tmp.name))
        self.assertEqual(rc, 1)
        self.assertIn("not enough free disk space", out)

    def test_a_qwen36_size_without_the_family(self):
        err = io.StringIO()
        with mock.patch.object(sys, "argv", ["setup.py", "--model", "Q8_0"]), contextlib.redirect_stderr(err):
            rc, _ = run(setup.main)
        self.assertEqual(rc, 2)
        self.assertIn("--family qwen36", err.getvalue())


if __name__ == "__main__":
    unittest.main()
