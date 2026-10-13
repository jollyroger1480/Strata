"""Tests for setup.py's model menu (0.1.44): the family order, NVFP4 hidden on AMD and Intel, the NVFP4 line's card note
(RTX 50 native FP4, older RTX slower than UD-IQ4_XS), the Unsloth sizes in order with images on, and the prebuilt
install removing a stale strata_kq library.

    python -m unittest tools.test_setup_menu
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class FamilyOrder(unittest.TestCase):
    def test_the_menu_order_on_nvidia(self):
        fams = setup.menu_families(False) + list(setup.OWN_PATH_FAMILIES)
        self.assertEqual(fams, ["qwen", "swift", "coder", "unsloth", "nvfp4", "qwen36"])

    def test_the_default_stays_the_first_family(self):
        self.assertEqual(setup.menu_families(False)[0], "qwen")
        self.assertIn("GSQ-RCO", setup.FAMILIES["qwen"]["by"])

    def test_nvfp4_is_not_listed_on_amd_and_intel(self):
        fams = setup.menu_families(True)
        self.assertNotIn("nvfp4", fams)
        self.assertEqual(fams, ["qwen", "swift", "coder", "unsloth"])
        self.assertIn("nvfp4", setup.FAMILIES)             # --family nvfp4 still exists, and says why not

    def test_the_unsloth_line_names_the_alternative_on_amd_and_intel(self):
        self.assertIn("UD-IQ4_XS", setup.family_menu_note("unsloth", True, None))
        self.assertEqual(setup.family_menu_note("unsloth", False, {"arch": 86}), "")
        self.assertEqual(setup.family_menu_note("coder", True, None), "")


class Nvfp4Note(unittest.TestCase):
    def test_rtx_50_has_native_fp4(self):
        line = setup.family_menu_note("nvfp4", False, {"arch": 120, "name": "RTX 5070"})
        self.assertIn("native FP4 on this card", line)

    def test_older_rtx_runs_but_is_slower_than_ud_iq4_xs(self):
        for arch in (75, 86, 89):
            line = setup.family_menu_note("nvfp4", False, {"arch": arch})
            self.assertIn("runs, but UD-IQ4_XS is faster on this card", line, arch)
            self.assertNotIn("native FP4", line)

    def test_the_line_gives_the_download_and_the_ram_need(self):
        line = setup.family_menu_note("nvfp4", False, {"arch": 120})
        d = setup.MODELS["NVFP4"]
        self.assertIn(f"{d['download_gb']:.0f} GB download", line)
        self.assertIn(f"~{d['ram_gb']} GB of RAM", line)


class UnslothSizes(unittest.TestCase):
    def test_the_sizes_in_menu_order(self):
        names = [m for m in setup.MODELS if "unsloth" in setup.MODELS[m].get("families", ())]
        self.assertEqual(names, ["UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"])

    def test_nothing_is_experimental_and_images_are_on(self):
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            d = setup.MODELS[m]
            self.assertFalse(d.get("experimental"), m)
            self.assertFalse(d.get("nvidia_only"), m)
            self.assertIs(d.get("vision"), True, m)
            self.assertIn("download_gb", d)
            self.assertIn("ram_gb", d)
        self.assertNotIn("experimental", setup.MODELS["UD-Q4_K_XL"]["about"].lower())
        self.assertTrue(setup.FAMILIES["unsloth"]["vision"])

    def test_the_family_line_lists_every_size(self):
        about = setup.FAMILIES["unsloth"]["about"]
        for m in ("UD-IQ4_XS", "UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "Q8_0"):
            self.assertIn(m, about)


class EnginePlaceholders(unittest.TestCase):
    def test_every_engine_minimum_is_this_release(self):
        self.assertEqual(setup.NVFP4_ENGINE, (0, 1, 44))
        self.assertEqual(setup.MIN_ENGINE, (0, 1, 44))
        for m in ("UD-Q5_K_XL", "UD-Q6_K_XL", "NVFP4", "NVFP4-NVIDIA"):
            self.assertEqual(setup.MODELS[m]["engine"], (0, 1, 44), m)
        self.assertIn("0.1.44", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8").split("project(strata VERSION", 1)[1][:12])


class KqLibrary(unittest.TestCase):
    def test_a_zip_without_strata_kq_does_not_leave_an_old_one(self):
        with tempfile.TemporaryDirectory() as d:
            eng, tmp = Path(d) / "engine", Path(d) / "new"
            eng.mkdir()
            tmp.mkdir()
            (eng / setup.EXE).write_bytes(b"old")
            (eng / setup.KQ_LIB).write_bytes(b"old kq")
            (tmp / setup.EXE).write_bytes(b"new")
            setup.install_unpacked(tmp, eng)
            self.assertEqual((eng / setup.EXE).read_bytes(), b"new")
            self.assertFalse((eng / setup.KQ_LIB).exists())
            self.assertEqual((eng / setup.PREVIOUS_ENGINE / setup.KQ_LIB).read_bytes(), b"old kq")   # a rollback gets it back

    def test_a_zip_with_strata_kq_replaces_it(self):
        with tempfile.TemporaryDirectory() as d:
            eng, tmp = Path(d) / "engine", Path(d) / "new"
            eng.mkdir()
            tmp.mkdir()
            (eng / setup.KQ_LIB).write_bytes(b"old kq")
            (tmp / setup.EXE).write_bytes(b"new")
            (tmp / setup.KQ_LIB).write_bytes(b"new kq")
            setup.install_unpacked(tmp, eng)
            self.assertEqual((eng / setup.KQ_LIB).read_bytes(), b"new kq")


if __name__ == "__main__":
    unittest.main()
