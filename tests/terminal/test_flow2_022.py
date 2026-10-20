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



class PlanWords(Base):
    """Findings 1 and 7: --plan shows the real model, the picture setting, a copyable line, and what would be downloaded."""

    def plan_text(self, *args):
        code, out, err = run_cli(["launch", "--plan", *args])
        self.assertEqual(code, 0, err or out)
        return out

    def test_own_files_show_the_model_they_are(self):
        folder = self.files(self.base / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        out = self.plan_text("--model-path", str(folder))
        self.assertIn("Plan for 'IQ3_XXS", out)
        self.assertNotIn("model default", out)
        self.assertNotIn("None", out)
        self.assertIn("Qwen3.8-Flash-Next IQ3_XXS", out)
        self.assertIn(f"your files in {folder}", out)
        self.assertRegex(out, r"pictures (on|off)")
        self.assertIn("--model iq3_xxs", out.split("same as")[1])         # the copyable line names the model too
        self.assertIn(f'--model-path "{folder}"', out.split("same as")[1])

    def test_a_named_catalog_model_shows_its_size_and_the_download(self):
        out = self.plan_text("--model", "q2_0")
        self.assertIn("Qwen3.8-Flash-Next Q2_0, 66 GB", out)
        self.assertRegex(out, r"download\s+About 7\d\.\d GB to download: the model \(66\.4 GB\), the draft layer \(5\.2 GB\)")
        self.assertIn("The engine for your graphics card is downloaded first", out)

    def test_nothing_is_downloaded_when_everything_is_here(self):
        folder = self.files(self.base / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        data = self.base / "Strata" 
        (data / "models" / "x").mkdir(parents=True)
        (data / "mtp" / "rt").mkdir(parents=True)
        (data / "mtp" / "rt" / "experts.bin").write_bytes(b"x")
        (data / "models" / "mmproj-Qwen3.8-Flash-Next-BF16.gguf").write_bytes(b"x")
        # the files sit under <data>/models/<tag>: the draft layer and the picture reader are found beside them
        moved = data / "models" / "iq3_xxs"
        folder.rename(moved)
        out = self.plan_text("--model-path", str(moved))
        self.assertIn("download  Nothing to download: your model files are used where they are.", out)

    def test_own_files_with_no_draft_layer_say_what_is_fetched(self):
        folder = self.files(self.base / "IQ3_XXS", IQ3XXS, "IQ3_XXS")
        with mock.patch.object(Runner, "existing_data_dir", return_value=None),                 mock.patch.object(Catalog, "data_dir", return_value=self.base / "emptydata"):
            out = self.plan_text("--model-path", str(folder), "--vision", "off")
        self.assertIn("About 5.2 GB to download: the draft layer (5.2 GB). Your model files are used where they are.", out)

    def test_json_plan_carries_the_numbers(self):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", "--model", "q2_0", "--mtp", "off", "--vision", "off"])
        self.assertEqual(code, 0, err)
        plan = json.loads(out)["plan"]
        self.assertEqual(plan["model"]["title"], "Qwen3.8-Flash-Next Q2_0")
        self.assertEqual(plan["model"]["model_gb"], 66.4)
        self.assertEqual(plan["download"]["gb"], 66.4)
        self.assertEqual([p["kind"] for p in plan["download"]["parts"]], ["weights"])



if __name__ == "__main__":
    unittest.main()
