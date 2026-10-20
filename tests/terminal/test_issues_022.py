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


if __name__ == "__main__":
    unittest.main()
