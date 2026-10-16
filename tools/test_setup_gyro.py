"""The Gyro entries of setup's catalog (agentionai/Qwen3.8-Flash-Next-Gyro-GGUF): sizes, pins, the VRAM + RAM memory rule,
never a default, and the MCP server's recommendation naming them only where nothing else fits.  Offline."""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp as M  # noqa: E402


def hardware(ram, vram):
    return {"ram_gb": ram, "gpus": [{"usable": True, "vram_gb": vram, "vendor": "nvidia", "index": 0, "name": "RTX"}],
            "cpu": {"avx2": True}}


class GyroCatalog(unittest.TestCase):
    def test_the_two_sizes_and_their_numbers(self):
        self.assertEqual(setup.FAMILIES["gyro"]["file"].format(q="Gyro-S"), "Qwen3.8-Flash-Next-Gyro-S.gguf")
        for model, name, download, memory in (("Gyro-S", "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf", 58.5, 27.6),
                                              ("Gyro-M", "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf", 92.0, 35.0)):
            d = setup.MODELS[model]
            self.assertEqual(setup.model_file(setup.FAMILIES["gyro"], model, 1), name)
            self.assertEqual((d["download_gb"], d["ram_gb"]), (download, memory))
            size, sha = setup.GYRO_FILES[name]
            self.assertAlmostEqual(size / 1e9, download, delta=0.05)
            self.assertEqual(len(sha), 64)
            self.assertEqual(setup.model_shards(setup.FAMILIES["gyro"], model), 1)
            self.assertIs(d.get("vision"), False)
            self.assertTrue(d["mem_total"])
            self.assertEqual(d["families"], ("gyro",))

    def test_pinned_revision_and_license(self):
        repo = "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF"
        self.assertEqual(len(setup.HF_REVISIONS[repo]), 40)
        self.assertIn(setup.HF_REVISIONS[repo], setup.FAMILIES["gyro"]["hf"])
        self.assertIn("Qwen Community License 1.0", setup.FAMILIES["gyro"]["license"])

    def test_never_the_default_family(self):
        self.assertEqual(setup.menu_families(False)[0], "qwen")
        self.assertNotIn("gyro", setup.RECOMMENDED_SIZES)

    def test_memory_is_vram_plus_ram(self):
        self.assertEqual(setup.mem_have("Gyro-S", 16.0, 12.0), 28.0)
        self.assertEqual(setup.mem_have("Q2_0", 16.0, 12.0), 16.0)
        names = ["Gyro-S", "Gyro-M"]
        self.assertEqual(setup.recommended_size("gyro", names, 24.0, 12.0), "Gyro-M")     # 36 GB
        self.assertEqual(setup.recommended_size("gyro", names, 16.0, 12.0), "Gyro-S")     # 28 GB
        self.assertEqual(setup.recommended_size("gyro", names, 8.0, 8.0), "Gyro-S")       # nothing fits: the first

    def test_mcp_table_matches(self):
        for m in ("Gyro-S", "Gyro-M"):
            for k in ("download_gb", "ram_gb", "arena_gb", "mem_total", "vision"):
                self.assertEqual(setup.MODELS[m][k], M.FALLBACK_MODELS[m][k], (m, k))
        self.assertEqual(setup.FAMILIES["gyro"]["tag"], M.FALLBACK_FAMILIES["gyro"]["tag"])

    def test_mcp_recommends_gyro_only_where_nothing_else_fits(self):
        s = M.Strata(ROOT)
        for ram, vram, family in ((64.0, 12.0, "qwen"), (48.0, 12.0, "qwen"), (32.0, 12.0, "coder")):
            self.assertEqual(s.recommend(hardware(ram, vram))["family"], family, (ram, vram))
        got = s.recommend(hardware(20.0, 12.0))                  # 32 GB in total, the Coder's 32 GB of RAM missing
        self.assertEqual((got["family"], got["model"]), ("gyro", "Gyro-S"))
        got = s.recommend(hardware(24.0, 16.0))                  # 40 GB in total: the Coder fits in the low-RAM mode, so it wins
        self.assertEqual((got["family"], got["model"]), ("coder", "IQ1_M"))
        self.assertIsNone(s.recommend(hardware(12.0, 8.0))["family"])
        self.assertIsNone(s.recommend(hardware(16.0, 11.0))["family"])   # 27 GB: below Gyro-S's 27.6 + room

    def test_mcp_table_verdict_counts_vram(self):
        s = M.Strata(ROOT)
        fam = next(f for f in s.model_table(hardware(16.0, 16.0)) if f["family"] == "gyro")
        by = {x["model"]: x for x in fam["sizes"]}
        self.assertEqual(by["Gyro-S"]["on_this_pc"], "fits")
        self.assertEqual(by["Gyro-M"]["on_this_pc"], "tight")
        self.assertFalse(by["Gyro-S"]["images"])


class GyroEngineVersionTest(unittest.TestCase):
    def test_gyro_needs_the_split_release_engine(self):
        self.assertEqual(setup.GYRO_ENGINE, (0, 1, 45))
        sizes = [m for m in setup.MODELS.values() if "gyro" in str(m.get("file", "")).lower()]
        self.assertEqual(len(sizes), 2)
        self.assertTrue(all(m["engine"] == (0, 1, 45) for m in sizes))


if __name__ == "__main__":
    unittest.main()
