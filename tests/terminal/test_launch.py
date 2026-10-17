"""`strata launch` without screens: the settings rules, the catalog, presets, parameters, access, connect, and every flag."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from strata_cli.launch import access, harness, params, settings as S, store
from strata_cli.launch.catalog import Catalog, CatalogUnavailable, fit_on, verdict
from strata_cli.launch.runner import Result, Runner, derive_config, desc_matches, tune_words
from tests.terminal.launch_support import HW_12, HW_24, HW_2X16, Home, catalog, run_cli


class SettingsRules(unittest.TestCase):
    def test_parse_context(self):
        self.assertEqual(S.parse_context("128k"), 131072)
        self.assertEqual(S.parse_context("128K"), 131072)
        self.assertEqual(S.parse_context("32 k"), 32768)
        self.assertEqual(S.parse_context("65536"), 65536)
        with self.assertRaises(ValueError):
            S.parse_context("lots")
        with self.assertRaises(ValueError):
            S.parse_context("100")

    def test_fit_parallel_follows_the_spec(self):
        self.assertEqual([S.fit_parallel(v) for v in (8, 12, 16, 20, 24, 32, 48)], [1, 1, 1, 2, 3, 4, 4])
        self.assertEqual(S.fit_parallel(None), 1)

    def test_words_and_labels(self):
        self.assertEqual(S.context_label(131072), "128K")
        self.assertEqual(S.words_for(131072), "about 98,000 words")

    def test_parallel_warning_is_a_warning_not_an_error(self):
        notes = S.check_parallel(S.LaunchSettings(parallel=4), vram_gb=12)
        self.assertEqual([n.level for n in notes], ["warn"])
        self.assertIn("still start", notes[0].line())
        self.assertEqual(S.check_parallel(S.LaunchSettings(parallel=9), vram_gb=12)[0].level, "error")
        self.assertEqual(S.check_parallel(S.LaunchSettings(parallel=2), vram_gb=24), [])

    def test_access_notes(self):
        self.assertEqual(S.check_access(S.LaunchSettings(access="local")), [])
        lan = S.check_access(S.LaunchSettings(access="lan"))
        self.assertEqual([n.level for n in lan], ["warn"])
        self.assertIn("API key", lan[0].line())
        bad = S.check_access(S.LaunchSettings(access="lann"))
        self.assertEqual(bad[0].level, "error")
        self.assertIn("lan", bad[0].line())
        self.assertEqual(S.check_port("abc").level, "error")
        self.assertEqual(S.check_port(70000).level, "error")
        self.assertEqual(S.check_port(80).level, "warn")
        self.assertIsNone(S.check_port(8080))

    def test_split_setting(self):
        self.assertEqual(S.split_setting("prefill=65536"), ("--prefill", "65536"))
        self.assertEqual(S.split_setting("--kv-grow"), ("--kv-grow", ""))
        self.assertEqual(S.split_setting("STRATA_PIN_BY_NEED=1"), ("STRATA_PIN_BY_NEED", "1"))
        self.assertEqual(S.split_setting("expert_cache=auto"), ("--expert-cache", "auto"))
        with self.assertRaises(ValueError):
            S.split_setting("no spaces allowed=1")

    def test_flags_text_round_trips_to_the_same_choices(self):
        s = S.LaunchSettings(family="qwen", model="IQ3_XXS", context=131072, parallel=2, tune="quick", access="lan", port=9000,
                             extra={"--prefill": "65536", "STRATA_PIN_BY_NEED": "1"})
        text = S.flags_text(s)
        for part in ("--model iq3_xxs", "--context 128k", "--parallel 2", "--tune quick", "--access lan", "--port 9000",
                     "--set prefill=65536", "--set STRATA_PIN_BY_NEED=1"):
            self.assertIn(part, text)
        self.assertIn('--preset "Fast"', S.flags_text(s, preset="Fast"))

    def test_slug(self):
        self.assertEqual(S.slug("IQ3_XXS 128K - Full speed"), "iq3_xxs-128k-full-speed")
        self.assertTrue(S.slug("123 go")[0].isalpha())

    def test_roundtrip_dict(self):
        s = S.LaunchSettings(name="a", model="Q2_0", family="qwen", context=32768, extra={"--spec": "3"}, tuned_preset="abc")
        again = S.LaunchSettings.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(again, s)
        self.assertEqual(s.to_dict(with_key=False)["api_key"], None)
        self.assertEqual(S.LaunchSettings(api_key="sk-x").to_dict(with_key=False)["api_key"], "set")


class CatalogRules(unittest.TestCase):
    def test_one_source_of_truth_and_fit_numbers(self):
        with Home():
            cat = catalog(HW_12)
            rows = cat.rows()
            ids = {(r.family, r.model) for r in rows}
            self.assertIn(("qwen", "IQ3_XXS"), ids)
            models, families, contexts, source = cat.tables()
            self.assertEqual(source, "setup.py")                       # the same tables setup.py and the app read
            for r in rows:
                self.assertAlmostEqual(r.download_gb, models[r.model]["download_gb"])
            rec = [r for r in rows if r.recommended]
            self.assertEqual([(r.family, r.model) for r in rec], [("qwen", "IQ3_XXS")])

    def test_fit_on_matches_native_fit_ts(self):
        # apps/host/src/lib/native-fit.ts: memory = experts + 1; GPU holds experts up to vram - 5; RAM holds up to ram - 10; the rest spills
        f = fit_on({"experts_gb": 42.9, "ram_needed_gb": 60}, 64.0, 12.0)
        self.assertEqual((f["memoryGb"], f["vramUseGb"], f["spillGb"]), (43.9, 12.0, 0.0))
        self.assertAlmostEqual(f["ramUseGb"], 35.9, places=1)
        spill = fit_on({"experts_gb": 98.4, "ram_needed_gb": 108}, 64.0, 12.0)
        self.assertGreater(spill["spillGb"], 30)
        self.assertFalse(fit_on({"experts_gb": 10}, None, 12)["known"])
        gyro = fit_on({"mem_total": True, "ram_needed_gb": 27.6}, 64, 12)
        self.assertEqual(gyro["memoryGb"], 27.6)

    def test_verdict_words_never_block(self):
        self.assertEqual(verdict("fits")[0], "fits")
        self.assertEqual(verdict("does not fit")[0], "no")
        self.assertIn("can still try", verdict("does not fit")[1])
        self.assertEqual(verdict("tight")[0], "tight")
        self.assertEqual(verdict("fits in the low-RAM mode (slower)")[0], "fits")

    def test_resolve_names_and_closest(self):
        with Home():
            cat = catalog()
            self.assertEqual(cat.resolve("iq3_xxs"), ("qwen", "IQ3_XXS"))
            self.assertEqual(cat.resolve("IQ1_M"), ("coder", "IQ1_M"))
            self.assertEqual(cat.resolve("iq3_xxs", "swift"), ("swift", "IQ3_XXS"))
            with self.assertRaises(ValueError) as e:
                cat.resolve("iq3xxs")
            self.assertIn("Did you mean 'IQ3_XXS'", str(e.exception))
            with self.assertRaises(ValueError) as e:
                cat.resolve("q2_0", "swiift")
            self.assertIn("swift", str(e.exception))
            with self.assertRaises(ValueError) as e:
                cat.resolve("IQ1_M", "swift")
            self.assertIn("has no size", str(e.exception))

    def test_small_computer_gets_warnings_not_blocks(self):
        with Home():
            rows = catalog({**HW_12, "ram_gb": 32.0}).rows()
            big = [r for r in rows if r.model == "IQ3_S" and r.family == "qwen"][0]
            self.assertIn(big.verdict, ("no", "tight"))

    def test_kv_figures(self):
        with Home():
            cat = catalog()
            self.assertAlmostEqual(cat.kv_gb(131072, "int8"), 1.799, places=2)
            self.assertLess(cat.kv_gb(131072, "q4_0"), cat.kv_gb(131072, "int8"))
            self.assertEqual(cat.max_context("qwen", "IQ3_XXS", HW_12), 32768)
            self.assertEqual(cat.max_context("qwen", "IQ3_XXS", HW_24), 131072)

    def test_missing_installer_files_are_said_in_words(self):
        with Home():
            with self.assertRaises(CatalogUnavailable) as e:
                Catalog(src="/nonexistent-folder", hardware=HW_12)
            self.assertIn("one-line installer", str(e.exception))


class PresetStore(unittest.TestCase):
    def test_save_load_list_delete(self):
        with Home():
            s = S.LaunchSettings(name="IQ3_XXS 128K", family="qwen", model="IQ3_XXS", context=131072, api_key="sk-secret")
            store.save(s)
            store.save(S.LaunchSettings(name="Coder 64K", family="coder", model="IQ1_M", context=65536))
            names = [p.name for p in store.list_presets()]
            self.assertEqual(sorted(names), ["Coder 64K", "IQ3_XXS 128K"])
            back = store.load("IQ3_XXS 128K")
            self.assertEqual((back.model, back.context, back.api_key), ("IQ3_XXS", 131072, "sk-secret"))
            store.touch_used("IQ3_XXS 128K")
            self.assertTrue(store.load("IQ3_XXS 128K").last_used)
            self.assertEqual(store.list_presets()[0].name, "IQ3_XXS 128K")          # most recently used first
            store.delete("Coder 64K")
            self.assertEqual([p.name for p in store.list_presets()], ["IQ3_XXS 128K"])
            with self.assertRaises(store.PresetError) as e:
                store.load("IQ3 XXS 128")
            self.assertIn("Did you mean", str(e.exception))
            with self.assertRaises(store.PresetError):
                store.delete("nope")

    def test_key_is_not_world_readable_on_posix(self):
        if os.name == "nt":
            self.skipTest("POSIX modes")
        with Home():
            store.save(S.LaunchSettings(name="k", api_key="sk-secret"))
            path = next(store.preset_dir().glob("*.json"))
            self.assertEqual(path.stat().st_mode & 0o077, 0)


class Advanced(unittest.TestCase):
    HELP = """strata generate --pack DIR [options]

  --pack DIR           the pack directory (default pack/full)
  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes, half the
                       VRAM; default fp16 until the gate accepts int8
  --kv q4_0            4-bit K/V after a Hadamard rotation
  --kv-grow            the K/V takes VRAM only for the cells the requests reach
  --native-rope        experimental pinned text-only CUDA rotary arithmetic
  --prefill N          prompt chunk size (default 2048)
"""

    def test_parse_help(self):
        ps = {p.name: p for p in params.parse_help(self.HELP)}
        self.assertEqual(set(ps), {"--pack", "--kv", "--kv-grow", "--native-rope", "--prefill"})
        self.assertEqual(ps["--pack"].metavar, "DIR")
        self.assertEqual(ps["--pack"].help_default, "pack/full")
        self.assertEqual(ps["--kv"].values, ["fp16", "int8"])
        self.assertIn("4-bit", ps["--kv"].description)               # a flag listed twice keeps both texts
        self.assertTrue(ps["--native-rope"].experimental)
        self.assertFalse(ps["--prefill"].experimental)
        self.assertEqual(ps["--kv-grow"].metavar, "")
        self.assertLessEqual(len(ps["--prefill"].short), 70)

    def test_rows_show_default_tuned_and_yours(self):
        rows = {r.name: r for r in params.build_rows(self.HELP, ["--kv", "int8", "--prefill", "auto"], ["--kv", "int8", "--prefill", "auto:32768"],
                                                     {}, {}, {"--prefill": "65536"})}
        self.assertEqual((rows["--prefill"].default, rows["--prefill"].tuned, rows["--prefill"].yours), ("auto", "auto:32768", "65536"))
        self.assertEqual(rows["--kv"].yours, "")
        self.assertIn("--native-rope", rows)
        names = list(rows)
        self.assertEqual(names[:2], ["--kv", "--prefill"])             # what this launch uses comes first

    def test_unknown_names_warn_and_are_kept(self):
        rows = params.build_rows(self.HELP, [], None, {}, {}, {"--prefil": "1"})
        r = next(x for x in rows if x.name == "--prefil")
        self.assertFalse(r.known)
        warn = params.check_name("--prefil", ["--prefill", "--kv"])
        self.assertIn("Did you mean --prefill", warn)
        self.assertIsNone(params.check_name("--prefill", ["--prefill"]))
        self.assertIsNone(params.check_name("--prefil", None))        # no list to check against: no warning

    def test_extra_becomes_argv_and_env(self):
        a, e = S.as_argv_changes({"--prefill": "65536", "--kv-grow": "", "--spec": "off", "STRATA_PIN_BY_NEED": "1"}, frozenset({"--kv-grow"}))
        self.assertEqual(a, {"--prefill": "65536", "--kv-grow": True, "--spec": None})
        self.assertEqual(e, {"STRATA_PIN_BY_NEED": "1"})


class RunConfig(unittest.TestCase):
    BASE = {"exe": "component:strata-engine", "args": ["--native", "x.gguf", "--max-context", "32768", "--kv", "int8", "--prefill", "auto"],
            "port": 8095, "model_name": "m", "env": {}}

    def test_derive_applies_access_key_and_advanced(self):
        s = S.LaunchSettings(access="lan", port=9001, api_key="sk-k", extra={"--prefill": "65536", "--kv-grow": "", "STRATA_PIN_BY_NEED": "1"})
        cfg = derive_config(self.BASE, s, bool_flags=frozenset({"--kv-grow"}))
        self.assertEqual((cfg["host"], cfg["port"], cfg["api_key"]), ("0.0.0.0", 9001, "sk-k"))
        args = cfg["args"]
        self.assertEqual(args[args.index("--prefill") + 1], "65536")
        self.assertIn("--kv-grow", args)
        self.assertEqual(cfg["env"], {"STRATA_PIN_BY_NEED": "1"})
        self.assertEqual(self.BASE["args"][-1], "auto")               # the installed config is never edited

    def test_local_and_cloudflare_stay_on_loopback_without_a_key(self):
        for kind in ("local", "cloudflare"):
            cfg = derive_config(self.BASE, S.LaunchSettings(access=kind), bool_flags=frozenset())
            self.assertEqual(cfg["host"], "127.0.0.1")
            self.assertNotIn("api_key", cfg)

    def test_installed_config_matching(self):
        d = {"context": 32768, "kv": "int8", "images": "off"}
        self.assertTrue(desc_matches(d, S.LaunchSettings(context=32768)))
        self.assertFalse(desc_matches(d, S.LaunchSettings(context=65536)))
        self.assertFalse(desc_matches(d, S.LaunchSettings(context=32768, vision="on")))
        self.assertFalse(desc_matches({**d, "images": "cpu"}, S.LaunchSettings(context=32768, vision="off")))
        self.assertFalse(desc_matches(d, S.LaunchSettings(context=32768, kv="q4_0")))

    def test_tune_words(self):
        self.assertEqual(tune_words({"outcome": "improved", "gain_pct": 4.2})[0], "faster")
        self.assertEqual(tune_words({"outcome": "default_confirmed"})[0], "already fastest")
        self.assertEqual(tune_words({"outcome": "inconclusive", "reason_code": "noisy"})[0], "busy PC")
        self.assertEqual(tune_words({"outcome": "base_failed", "message": "Not tuned: no memory"})[0], "failed")
        self.assertEqual(tune_words(None)[0], "failed")
        self.assertEqual(tune_words({"outcome": "cancelled"})[0], "stopped")


class AccessRules(unittest.TestCase):
    def test_hosts_and_keys(self):
        self.assertEqual(access.host_for("local"), "127.0.0.1")
        self.assertEqual(access.host_for("cloudflare"), "127.0.0.1")
        self.assertEqual(access.host_for("lan"), "0.0.0.0")
        self.assertTrue(access.new_key().startswith("sk-strata-"))
        self.assertNotIn(access.new_key(), access.mask_key(access.new_key()))
        self.assertEqual(access.mask_key(None), "off")

    def test_private_ranges(self):
        for ip in ("10.1.2.3", "192.168.88.21", "172.16.0.9", "100.101.1.1"):
            self.assertTrue(access._is_private(ip), ip)
        for ip in ("8.8.8.8", "172.32.0.1", "1.2.3.4"):
            self.assertFalse(access._is_private(ip), ip)

    def test_missing_helpers_say_how_to_install(self):
        with mock.patch.object(access, "find_cloudflared", return_value=None):
            msg = access.missing_helper("cloudflare")
            self.assertIn("cloudflared is not installed", msg)
            self.assertIn("quick tunnel", msg)
        with mock.patch.object(access, "find_tailscale", return_value=None):
            self.assertIn("Tailscale is not installed", access.missing_helper("tailscale"))
        with mock.patch.object(access, "tailscale_status", return_value={"installed": True, "running": False, "ip": None, "name": None}):
            self.assertIn("tailscale up", access.missing_helper("tailscale"))
        self.assertIsNone(access.missing_helper("local"))

    def test_options_and_urls(self):
        with mock.patch.object(access, "lan_ips", return_value=["192.168.1.5"]), \
                mock.patch.object(access, "find_cloudflared", return_value="/bin/cloudflared"), \
                mock.patch.object(access, "tailscale_status", return_value={"installed": True, "running": True, "ip": "100.64.0.1", "name": "pc.tail.ts.net"}):
            opts = {o.kind: o for o in access.options(8080)}
            self.assertEqual(opts["lan"].url, "http://192.168.1.5:8080")
            self.assertTrue(opts["cloudflare"].ready)
            self.assertEqual(opts["tailscale"].url, "http://pc.tail.ts.net:8080")
            urls = access.urls_for(S.LaunchSettings(access="lan", port=8080))
            self.assertEqual(urls, ["http://127.0.0.1:8080", "http://192.168.1.5:8080"])
            self.assertEqual(access.urls_for(S.LaunchSettings(access="cloudflare", port=8080), tunnel_url="https://a.trycloudflare.com")[0],
                             "https://a.trycloudflare.com")

    def test_helper_output_filter_drops_login_lines(self):
        self.assertEqual(access.filter_line("Please open this URL to login: https://dash.cloudflare.com/argotunnel?token=abc"), "")
        self.assertEqual(access.filter_line("Your quick Tunnel has been created: https://x.trycloudflare.com"),
                         "Your quick Tunnel has been created: https://x.trycloudflare.com")
        self.assertEqual(access.TUNNEL_URL.search("| https://calm-river-1234.trycloudflare.com |").group(0),
                         "https://calm-river-1234.trycloudflare.com")


class FlagsCommand(unittest.TestCase):
    """Every `strata launch` flag, run through the real parser with the computer faked and nothing installed."""

    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)
        p = mock.patch.object(Catalog, "hardware", lambda self, fresh=False: dict(HW_12, disk_free_gb=500.0))
        p.start()
        self.addCleanup(p.stop)

    def plan(self, *args):
        code, out, err = run_cli(["launch", "--plan", "--output", "json", *args])
        self.assertEqual(code, 0, err or out)
        return json.loads(out)["plan"]

    def test_help_lists_everything_with_examples(self):
        code, out, err = run_cli(["launch", "--help"])
        self.assertEqual(code, 0)
        for flag in ("--preset", "--model", "--family", "--model-path", "--data-dir", "--context", "--kv", "--vision", "--parallel", "--gpu",
                     "--tune", "--tune-allow-small", "--set", "--access", "--port", "--api-key", "--api-key-env", "--new-key", "--save-preset",
                     "--no-tui", "--plan", "--feed", "--list-presets", "--delete-preset", "--status", "--stop", "--show-key", "examples:"):
            self.assertIn(flag, out)
        code, out, _ = run_cli(["--help"])
        self.assertIn("launch", out)
        self.assertIn("connect", out)

    def test_model_context_parallel_vision_tune_access_port(self):
        p = self.plan("--model", "iq3_xxs", "--context", "128k", "--parallel", "2", "--vision", "off", "--tune", "quick", "--access", "lan",
                      "--port", "9090")
        s = p["settings"]
        self.assertEqual((s["family"], s["model"], s["context"], s["parallel"], s["vision"], s["tune"], s["access"], s["port"]),
                         ("qwen", "IQ3_XXS", 131072, 2, "off", "quick", "lan", 9090))
        self.assertEqual(s["api_key"], "set")                               # shared beyond this computer: a key was made
        self.assertTrue(any("2 requests" in w and "12 GB" in w for w in p["warnings"]))

    def test_defaults_come_from_the_recommendation(self):
        p = self.plan()
        s = p["settings"]
        self.assertEqual((s["family"], s["model"], s["context"], s["access"], s["tune"]), ("qwen", "IQ3_XXS", 32768, "local", "skip"))
        self.assertTrue(any("picked" in w for w in p["warnings"]))

    def test_family_and_kv_and_gpu_and_data_dir_and_set(self):
        p = self.plan("--model", "iq3_xxs", "--family", "swift", "--kv", "q4_0", "--gpu", "0,1", "--data-dir", "/data/x",
                      "--set", "prefill=65536", "--set", "STRATA_PIN_BY_NEED=1", "--tune-allow-small")
        s = p["settings"]
        self.assertEqual((s["family"], s["kv"], s["gpus"], s["data_dir"], s["tune_allow_small"]), ("swift", "q4_0", [0, 1], "/data/x", True))
        self.assertEqual(s["extra"], {"--prefill": "65536", "STRATA_PIN_BY_NEED": "1"})
        self.assertIn("--set prefill=65536", p["flags"])

    def test_api_key_flags(self):
        p = self.plan("--model", "q2_0", "--access", "lan", "--api-key", "sk-mine")
        self.assertEqual(p["settings"]["api_key"], "set")
        self.assertFalse(any("new API key was made" in w for w in p["warnings"]))
        with mock.patch.dict(os.environ, {"MY_KEY": "sk-env"}):
            p = self.plan("--model", "q2_0", "--api-key-env", "MY_KEY")
            self.assertEqual(p["settings"]["api_key"], "set")
        code, out, err = run_cli(["launch", "--plan", "--api-key-env", "NOT_SET_ANYWHERE"])
        self.assertEqual(code, 2)
        self.assertIn("NOT_SET_ANYWHERE", err)
        p = self.plan("--model", "q2_0", "--new-key")
        self.assertEqual(p["settings"]["api_key"], "set")

    def test_model_path_is_used_instead_of_a_catalog_model(self):
        p = self.plan("--model-path", "/data/models/IQ3_XXS")
        s = p["settings"]
        self.assertEqual(s["model_path"], str(Path("/data/models/IQ3_XXS")))
        self.assertIsNone(s["model"])

    def test_unknown_model_names_the_closest_one(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "iq3xxs"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean 'IQ3_XXS'", err)
        self.assertNotIn("Traceback", err)

    def test_unknown_flag_names_the_closest_one(self):
        code, out, err = run_cli(["launch", "--modle", "iq3_xxs"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean '--model'", err)

    def test_bad_values_are_plain_sentences(self):
        for argv, word in ((["--tune", "quik"], "quick"), (["--access", "lann"], "lan"), (["--context", "huge"], "context"),
                           (["--port", "99999"], "1 to 65535"), (["--parallel", "12"], "1 to 8"), (["--vision", "maybe"], "off")):
            code, out, err = run_cli(["launch", "--plan", *argv])
            self.assertEqual(code, 2, argv)
            self.assertIn(word, err, argv)

    def test_preset_with_flags_on_top(self):
        store.save(S.LaunchSettings(name="Office", family="qwen", model="IQ3_XXS", context=65536, parallel=2, access="local"))
        p = self.plan("--preset", "Office", "--context", "128k")
        s = p["settings"]
        self.assertEqual((s["name"], s["context"], s["parallel"]), ("Office", 131072, 2))
        code, out, err = run_cli(["launch", "--plan", "--preset", "Offce"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean 'Office'", err)

    def test_vision_is_turned_off_for_a_model_without_pictures(self):
        p = self.plan("--model", "gyro-s", "--vision", "on")
        self.assertEqual(p["settings"]["vision"], "off")
        self.assertTrue(any("cannot read pictures" in w for w in p["warnings"]))

    def test_does_not_fit_is_a_warning_with_the_reason(self):
        p = self.plan("--model", "nvfp4")
        self.assertTrue(any("does not fit" in w for w in p["warnings"]))

    def test_human_plan_changes_nothing(self):
        code, out, err = run_cli(["launch", "--plan", "--model", "q2_0", "--save-preset", "X"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing was changed.", out)
        self.assertEqual(store.list_presets(), [])

    def test_run_without_screens_prints_the_address_and_saves_the_preset(self):
        fake = Result(profile="launch-office", urls=["http://127.0.0.1:8080", "http://192.168.1.5:8080"], key="sk-strata-abcdefghijkl",
                      tune_word=None, notices=[], config="/x")
        with mock.patch.object(Runner, "run", return_value=fake):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--access", "lan", "--save-preset", "Office"])
        self.assertEqual(code, 0, err)
        self.assertIn("Strata is serving.", out)
        self.assertIn("http://127.0.0.1:8080/v1", out)
        self.assertIn("http://192.168.1.5:8080/v1", out)
        self.assertNotIn("sk-strata-abcdefghijkl", out)                      # the key is masked
        self.assertIn("strata connect", out)
        self.assertEqual([p.name for p in store.list_presets()], ["Office"])
        code, out, err = run_cli(["launch", "--list-presets"])
        self.assertIn("Office", out)
        code, out, _ = run_cli(["launch", "--show-key", "--preset", "Office"])
        self.assertEqual(code, 0)                                              # the preset's own key
        self.assertTrue(out.strip().startswith("sk-strata-"))

    def test_launch_error_is_one_plain_sentence(self):
        from strata_cli.launch.runner import LaunchError
        with mock.patch.object(Runner, "run", side_effect=LaunchError("The engine could not be installed.", "Check your internet.", 6)):
            code, out, err = run_cli(["launch", "--no-tui", "--model", "q2_0"])
        self.assertEqual(code, 6)
        self.assertIn("The engine could not be installed. Check your internet.", err)
        self.assertNotIn("Traceback", err)

    def test_list_delete_status_stop_without_anything_running(self):
        store.save(S.LaunchSettings(name="A", family="qwen", model="Q2_0", context=32768))
        code, out, _ = run_cli(["launch", "--list-presets"])
        self.assertIn("A", out)
        code, out, _ = run_cli(["launch", "--delete-preset", "A"])
        self.assertEqual(code, 0)
        self.assertIn("Models and tuning results were not touched", out)
        code, out, _ = run_cli(["launch", "--list-presets"])
        self.assertIn("No saved presets", out)
        code, out, _ = run_cli(["launch", "--status"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing from `strata launch` is running.", out)
        code, out, _ = run_cli(["launch", "--stop"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing", out)
        code, out, err = run_cli(["launch", "--delete-preset", "gone"])
        self.assertEqual(code, 2)
        self.assertIn("no saved preset", err)

    def test_feed_flag_reaches_the_runner(self):
        seen = {}

        def fake(self, *a, **k):
            seen["feed"] = self.feed
            return Result("launch-x", ["http://127.0.0.1:8080"], None, None, [], "/x")

        with mock.patch.object(Runner, "run", fake):
            code, _, err = run_cli(["launch", "--no-tui", "--model", "q2_0", "--feed", "http://127.0.0.1:9/feed"])
        self.assertEqual(code, 0, err)
        self.assertEqual(seen["feed"], "http://127.0.0.1:9/feed")

    def test_output_json_for_scripts(self):
        fake = Result("launch-x", ["http://127.0.0.1:8080"], "sk-k", "faster", ["a note"], "/x")
        with mock.patch.object(Runner, "run", return_value=fake):
            code, out, err = run_cli(["launch", "--no-tui", "--output", "json", "--model", "q2_0"])
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["urls"], ["http://127.0.0.1:8080"])
        self.assertEqual(data["api_key"], "set")                                # never the key itself
        self.assertEqual(data["tuned"], "faster")


class ConnectCommand(unittest.TestCase):
    def setUp(self):
        self.home = Home().__enter__()
        self.addCleanup(self.home.__exit__, None, None, None)

    def test_list_and_unknown_names(self):
        code, out, _ = run_cli(["connect"])
        self.assertEqual(code, 0)
        for h in ("opencode", "codex", "claude-code", "omp", "pi", "hermes", "openclaw", "deepseek-harness"):
            self.assertIn(h, out)
        code, out, err = run_cli(["connect", "opncode"])
        self.assertEqual(code, 2)
        self.assertIn("Did you mean 'opencode'", err)
        code, out, _ = run_cli(["connect", "--output", "json"])
        self.assertEqual(len(json.loads(out)["harnesses"]), 8)

    def test_prints_the_config_with_a_masked_key(self):
        code, out, err = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8123", "--key", "sk-strata-abcdefghij1234", "--model", "m1"])
        self.assertEqual(code, 0, err)
        self.assertIn("http://127.0.0.1:8123/v1", out)
        self.assertIn("opencode.json", out)
        self.assertIn("limit", out)
        self.assertNotIn("abcdefghij1234", out)
        code, out, _ = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8123", "--key", "sk-strata-abcdefghij1234", "--show-key"])
        self.assertIn("sk-strata-abcdefghij1234", out)

    def test_claude_code_uses_the_anthropic_variables_without_a_login(self):
        code, out, _ = run_cli(["connect", "claude-code", "--url", "http://127.0.0.1:8080", "--key", "sk-x1234567890abc", "--show-key"])
        self.assertIn("ANTHROPIC_BASE_URL", out)
        self.assertIn("ANTHROPIC_AUTH_TOKEN", out)
        self.assertNotIn("/v1", out.split("ANTHROPIC_BASE_URL")[1].splitlines()[0])      # the Anthropic client adds /v1 itself

    def test_write_keeps_a_backup_and_merges(self):
        target = Path(os.environ["HOME"]) / ".config" / "opencode" / "opencode.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"theme": "dark", "provider": {"other": {"a": 1}}}))
        code, out, err = run_cli(["connect", "opencode", "--url", "http://127.0.0.1:8080", "--key", "sk-x1234567890abc", "--write", "--yes"])
        self.assertEqual(code, 0, err)
        data = json.loads(target.read_text())
        self.assertEqual(data["theme"], "dark")
        self.assertIn("other", data["provider"])
        self.assertIn("strata", data["provider"])
        backups = list(target.parent.glob("opencode.json.strata-bak-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text())["theme"], "dark")
        self.assertIn("backup", out)

    def test_write_asks_first_without_yes(self):
        code, out, err = run_cli(["connect", "opencode", "--write"])
        self.assertEqual(code, 2)
        self.assertIn("asks first", err)

    def test_built_in_harness_has_nothing_to_write(self):
        code, out, _ = run_cli(["connect", "deepseek-harness", "--write", "--yes"])
        self.assertEqual(code, 0)
        self.assertIn("Nothing to write", out)

    def render(self, name, **kw):
        h = harness.find(name)
        t = harness.target(kw.get("url", "http://127.0.0.1:18900"), kw.get("key"), "qwen3.8-flash-next-q2_0", kw.get("context", 32768))
        return harness.render(h, t)

    def test_all_eight_are_tested_configs(self):
        self.assertEqual([h.id for h in harness.HARNESSES],
                         ["opencode", "codex", "claude-code", "omp", "pi", "hermes", "openclaw", "deepseek-harness"])
        for h in harness.HARNESSES:
            self.assertTrue(h.tested, h.id)
            self.assertIn("tested", h.status)

    def test_opencode_has_limit_input_and_the_15k_compaction_buffer(self):
        cfg = json.loads(self.render("opencode")["files"][0]["text"])
        m = cfg["provider"]["strata"]["models"]["qwen3.8-flash-next-q2_0"]
        self.assertEqual(m["limit"], {"context": 32768, "input": 28672, "output": 4096})
        self.assertEqual(cfg["compaction"]["buffer"], 13672)                  # 28672 - 15000
        self.assertEqual(cfg["provider"]["strata"]["options"]["baseURL"], "http://127.0.0.1:18900/v1")
        self.assertEqual(cfg["provider"]["strata"]["options"]["apiKey"], "strata-local")

    def test_codex_uses_responses_and_the_key_variable(self):
        out = self.render("codex", key="sk-real")
        text = out["files"][0]["text"]
        self.assertIn('wire_api = "responses"', text)
        self.assertIn("model_auto_compact_token_limit = 15000", text)
        self.assertIn('base_url = "http://127.0.0.1:18900/v1"', text)
        self.assertEqual(out["env"], {"STRATA_API_KEY": "sk-real"})

    def test_claude_code_sets_window_and_output_cap_and_has_no_login(self):
        out = self.render("claude-code")
        env = out["env"]
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:18900")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "32768")
        self.assertEqual(env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "4096")
        self.assertEqual(env["CLAUDE_CODE_SIMPLE"], "1")                      # the full prompt is 15.6K tokens
        self.assertEqual(json.loads(out["files"][0]["text"])["env"], env)
        big = self.render("claude-code", context=131072)["env"]
        self.assertEqual(big["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "131072")
        self.assertNotIn("CLAUDE_CODE_SIMPLE", big)
        self.assertTrue(any("CLAUDE_CODE_MAX_OUTPUT_TOKENS" in n for n in out["limits"]))

    def test_pi_omp_hermes_openclaw_numbers(self):
        pi = self.render("pi")["files"]
        self.assertEqual(json.loads(pi[1]["text"])["compaction"]["reserveTokens"], 17768)       # 32768 - 15000
        self.assertEqual(json.loads(pi[0]["text"])["providers"]["strata"]["models"][0]["contextWindow"], 32768)
        omp = self.render("omp")["files"]
        self.assertIn("thresholdTokens: 15000", omp[1]["text"])
        self.assertIn("contextWindow: 32768", omp[0]["text"])
        hermes = self.render("hermes")
        self.assertIn("provider: lmstudio", hermes["files"][0]["text"])
        self.assertIn("context_length: 65536", hermes["files"][0]["text"])
        self.assertTrue(any("WARNING" in n and "64K" in n for n in hermes["limits"]))
        self.assertFalse(any("WARNING" in n for n in self.render("hermes", context=131072)["limits"]))
        oc = json.loads(self.render("openclaw")["files"][0]["text"])
        self.assertEqual(oc["models"]["providers"]["strata"]["models"][0]["contextTokens"], 23192)
        self.assertTrue(any("Node 24.16" in n for n in self.render("openclaw")["limits"]))

    def test_limits_are_in_plain_words_for_every_app(self):
        for h in harness.HARNESSES:
            self.assertTrue(self.render(h.id)["limits"], h.id)

    def test_deepseek_harness_is_the_app(self):
        out = self.render("deepseek-harness")
        self.assertEqual(out["files"], [])
        self.assertIn("Strata app", " ".join(out["limits"]))

    def test_context_flag_reaches_the_config(self):
        code, out, _ = run_cli(["connect", "pi", "--url", "http://127.0.0.1:8080", "--context", "64k", "--show-key"])
        self.assertIn('"contextWindow": 65536', out)
        self.assertIn('"reserveTokens": 50536', out)


if __name__ == "__main__":
    unittest.main()
