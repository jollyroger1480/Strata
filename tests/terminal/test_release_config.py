"""The release app configuration: the engine travels inside every installer, the updater signs, the Mac minimum matches the Metal engine."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

TAURI = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "src-tauri"


def load(name: str) -> dict:
    return json.loads((TAURI / name).read_text(encoding="utf-8-sig"))


class ReleaseConfig(unittest.TestCase):
    def test_every_installer_bundles_the_engine(self):
        self.assertEqual(load("tauri.conf.json")["bundle"]["resources"]["resources/engine-bundle"], "engine-bundle")
        self.assertEqual(load("tauri.tester.conf.json")["bundle"]["resources"]["resources/engine-bundle"], "engine-bundle")   # the same place

    def test_the_release_overlay_makes_signed_updater_artifacts(self):
        self.assertIs(load("tauri.release.conf.json")["bundle"]["createUpdaterArtifacts"], True)
        self.assertIs(load("tauri.conf.json")["bundle"]["createUpdaterArtifacts"], False)       # a development build needs no private key

    def test_the_mac_app_needs_the_macos_the_metal_engine_needs(self):
        self.assertEqual(load("tauri.conf.json")["bundle"]["macOS"]["minimumSystemVersion"], "15.0")

    def test_the_updater_points_at_github_releases(self):
        up = load("tauri.conf.json")["plugins"]["updater"]
        self.assertEqual(up["endpoints"], ["https://github.com/Niko1221/Strata/releases/latest/download/latest.json"])
        self.assertNotIn("REPLACE", up["pubkey"])


if __name__ == "__main__":
    unittest.main()
