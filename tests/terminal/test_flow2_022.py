"""0.2.2 flow walk, second round (phaseA-tests/flow-findings-022.md): what `strata launch` says and does with your own model files, the
plan, the download figure, the long silent step and a tuning result that already exists."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.catalog import Catalog
from strata_cli.launch.runner import LaunchError, Runner
from tests.terminal.launch_support import HW_12, Home, run_cli
from tests.terminal.test_own_files import IQ3XXS, Q2, gguf_bytes


class Base(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-flow2-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def files(self, folder: Path, names: list[str], kind: str) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        for n in names:
            (folder / n).write_bytes(gguf_bytes([kind]))
        return folder

    def runner(self, **kw) -> Runner:
        return Runner(S.LaunchSettings(**kw), catalog=Catalog(hardware=HW_12))


class MissingOwnFiles(Base):
    """Finding 2: a --model-path that is not there, or holds no model, is said in plain words before anything starts."""

    def test_a_folder_that_does_not_exist_is_named_and_nothing_is_promised(self):
        missing = self.base / "models"
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(missing)])
        self.assertEqual(code, 2)
        self.assertIn(f"This folder does not exist: {missing}", err)
        self.assertNotIn("Qwen", err)
        self.assertNotIn("continues where it stopped", err)
        self.assertNotIn("could not be installed", err)

    def test_a_missing_file_says_file(self):
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(self.base / "x-00001-of-00002.gguf")])
        self.assertEqual(code, 2)
        self.assertIn("This file does not exist", err)

    def test_an_empty_folder_says_there_are_no_gguf_files(self):
        empty = self.base / "empty"
        empty.mkdir()
        code, out, err = run_cli(["launch", "--plan", "--model", "q2_0", "--model-path", str(empty)])
        self.assertEqual(code, 2)
        self.assertIn("no .gguf model files", err)
        self.assertIn("...-00001-of-00002.gguf", err)

    def test_machine_output_carries_the_same_words(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model-path", str(self.base / "gone")])
        self.assertEqual(code, 2)
        self.assertIn("This folder does not exist", json.loads(out)["error"]["message"])

    def test_the_engine_is_not_touched_for_a_wrong_path(self):
        r = self.runner(model_path=str(self.base / "gone"))
        with mock.patch.object(Runner, "ensure_engine", side_effect=AssertionError("engine step must not start")):
            with self.assertRaises(LaunchError) as ctx:
                r.run()
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("does not exist", ctx.exception.text)

    def test_the_failure_hint_of_own_files_promises_no_download(self):
        r = self.runner(model_path=str(self.files(self.base / "m", Q2, "Q2_0")), family="qwen", model="Q2_0")
        proc = mock.Mock()
        proc.stdout = iter(["=== Step 5: Getting the model files ===", "[X] missing /m/x.gguf"])
        proc.wait.return_value = 1
        with mock.patch.object(Runner, "_spawn", return_value=proc), mock.patch.object(Runner, "adopt_prepared_files", return_value=None):
            with self.assertRaises(LaunchError) as ctx:
                r.ensure_model()
        self.assertNotIn("download", ctx.exception.next)
        self.assertIn("finished steps are skipped", ctx.exception.next)

    def test_the_screens_say_the_same(self):
        from strata_control import models
        text, nxt = models.own_path_problem(str(self.base / "nope"))
        self.assertIn("This folder does not exist", text)
        self.assertIn("Nothing was downloaded", nxt)
        self.assertIsNone(models.own_path_problem(str(self.files(self.base / "ok", IQ3XXS, "IQ3_XXS"))))

    def test_a_folder_one_level_above_the_files_is_not_called_empty(self):
        self.files(self.base / "repo" / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        from strata_control import models
        self.assertIsNone(models.own_path_problem(str(self.base / "repo")))


class LinkedShards(Base):
    """Finding 3: shards that are links are accepted with no --model, as with --model, and a problem is said once."""

    def links(self, names: list[str], real: bool = True) -> Path:
        blobs = self.base / "blobs"
        snap = self.base / "snap"
        blobs.mkdir(exist_ok=True)
        snap.mkdir(exist_ok=True)
        for i, n in enumerate(names):
            target = blobs / f"sha{i}"
            if real:
                target.write_bytes(gguf_bytes(["Q2_0"]))
            try:
                os.symlink(target, snap / n)
            except (OSError, NotImplementedError):
                self.skipTest("this account cannot create symbolic links")
        return snap

    def test_linked_shards_without_a_model_name_are_used(self):
        snap = self.links(Q2)
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model-path", str(snap)])
        self.assertEqual(code, 0, err or out)
        s = json.loads(out)["plan"]["settings"]
        self.assertEqual((s["family"], s["model"]), ("qwen", "Q2_0"))
        self.assertEqual(Path(s["model_path"]), snap)                       # the link names stay (the shard names are what setup reads)

    def test_linked_shards_with_the_model_named_give_the_same_folder(self):
        snap = self.links(Q2)
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--model-path", str(snap)])
        self.assertEqual(code, 0, err or out)
        self.assertEqual(Path(json.loads(out)["plan"]["settings"]["model_path"]), snap)

    def test_links_to_nothing_are_one_sentence_not_one_per_file(self):
        snap = self.links(Q2, real=False)
        code, out, err = run_cli(["launch", "--plan", "--model-path", str(snap)])
        self.assertEqual(code, 5)
        self.assertEqual(err.count("shortcut"), 1, err)
        self.assertIn(Q2[0], err)
        self.assertIn(Q2[1], err)

    def test_the_same_problem_for_many_files_is_said_once(self):
        from strata_control import models
        issues = [f"m-0000{i}-of-00005.gguf: GGUF metadata is truncated" for i in range(1, 6)] + ["incomplete_or_invalid_shard_set"]
        words = models.problem_words(issues)
        self.assertEqual(words.count("the file is cut short"), 1)
        self.assertIn("m-00001-of-00005.gguf, m-00002-of-00005.gguf and 3 more files", words)
        self.assertEqual(models.problem_words(["a.gguf: GGUF metadata is truncated", "b.gguf: GGUF metadata is truncated"]),
                         "a.gguf and b.gguf: the file is cut short (a download or copy that did not finish)")
        self.assertEqual(models.problem_words(["multiple_model_groups", "unrecognized_or_unsupported_model_variant"]),
                         "the folder holds more than one model")
        many = [f"unsupported_or_missing_architecture", "incomplete_or_invalid_shard_set", "multiple_model_groups",
                "unsupported_or_incomplete_tokenizer_metadata", "missing_or_unexpected_expert_quantization"]
        self.assertTrue(models.problem_words(many).endswith("and 2 more problems"))



if __name__ == "__main__":
    unittest.main()
