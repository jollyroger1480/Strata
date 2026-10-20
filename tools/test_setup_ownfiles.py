"""setup.py and the person's own model files (--gguf-dir): setup checks them, it never deletes them (0.2.2).

    python -m unittest tools.test_setup_ownfiles
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


def run(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            fn(*args, **kw)
            code = 0
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


class OwnFileChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.file = Path(self.tmp.name) / "model-00001-of-00002.gguf"
        self.file.write_bytes(b"abcdefgh" * 100)
        self.sha = hashlib.sha256(self.file.read_bytes()).hexdigest()

    def test_a_downloaded_file_of_the_wrong_size_asks_to_delete_it(self):
        code, text = run(setup.verify_sha256, self.file, 801, self.sha)
        self.assertEqual(code, 1)
        self.assertIn("delete it and run setup again", text)

    def test_an_own_file_of_the_wrong_size_is_left_and_the_text_says_so(self):
        code, text = run(setup.verify_sha256, self.file, 801, self.sha, own=True)
        self.assertEqual(code, 1)
        self.assertIn("your file was left as it is", text)
        self.assertTrue(self.file.exists())

    def test_an_own_file_with_another_hash_is_not_deleted(self):
        code, text = run(setup.verify_sha256, self.file, 800, "0" * 64, own=True)
        self.assertEqual(code, 1)
        self.assertIn("does not match the published file", text)
        self.assertTrue(self.file.exists())

    def test_a_downloaded_file_with_another_hash_is_deleted_as_before(self):
        code, _text = run(setup.verify_sha256, self.file, 800, "0" * 64)
        self.assertEqual(code, 1)
        self.assertFalse(self.file.exists())

    def test_a_matching_own_file_passes(self):
        code, _text = run(setup.verify_sha256, self.file, 800, self.sha, own=True)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
