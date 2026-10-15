"""Owner-only files: 0600/0700 on POSIX, a one-entry protected ACL for the current user on NTFS."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from strata_control import private_files as pf  # noqa: E402


class PrivateFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_a_file_becomes_private_and_stays_readable_by_us(self):
        path = self.root / "key.json"
        path.write_text('{"apiKey": "x"}')
        pf.restrict_to_current_user(path)
        self.assertTrue(pf.is_private(path))
        self.assertEqual(path.read_text(), '{"apiKey": "x"}')
        path.write_text("{}")  # the owner can still replace it

    def test_a_directory_becomes_private_and_its_children_inherit(self):
        folder = self.root / "app-data"
        folder.mkdir()
        pf.restrict_to_current_user(folder)
        self.assertTrue(pf.is_private(folder))
        child = folder / "child.json"
        child.write_text("{}")
        if pf.WINDOWS:
            # the child inherits the directory's single entry rather than the profile's
            sddl = pf.dacl_sddl(child)
            self.assertIn(pf.current_user_sid(), sddl)
            self.assertNotIn("S-1-1-0", sddl)
            self.assertNotIn(";;;SY)", sddl)
            self.assertNotIn(";;;BA)", sddl)
        else:
            self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o700)

    @unittest.skipIf(pf.WINDOWS, "POSIX mode bits")
    def test_posix_modes_are_exact(self):
        path = self.root / "key.json"
        path.write_text("{}")
        os.chmod(path, 0o666)
        pf.restrict_to_current_user(path)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        os.chmod(path, 0o644)
        self.assertFalse(pf.is_private(path))

    @unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
    def test_the_effective_acl_is_one_protected_entry_for_the_current_user(self):
        path = self.root / "key.json"
        path.write_text("{}")
        self.assertFalse(pf.is_private(path), "a fresh file inherits SYSTEM/Administrators entries")
        pf.restrict_to_current_user(path)
        sddl = pf.dacl_sddl(path)
        self.assertEqual(sddl, f"D:PAI(A;;FA;;;{pf.current_user_sid()})")
        # nobody else: no Everyone, Users, Authenticated Users, SYSTEM or Administrators
        for sid in ("WD", "BU", "AU", "SY", "BA", "S-1-1-0", "S-1-5-32-545"):
            self.assertNotIn(f";;;{sid})", sddl)

    def test_a_file_made_in_the_profile_is_unshared_but_not_strictly_private(self):
        path = self.root / "inherited.json"
        path.write_text("{}")
        self.assertTrue(pf.is_unshared(path))
        if pf.WINDOWS:
            self.assertFalse(pf.is_private(path))   # SYSTEM and Administrators still inherit

    @unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
    def test_everyone_is_noticed_by_the_relaxed_check_too(self):
        import subprocess
        path = self.root / "shared.json"
        path.write_text("{}")
        subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:R"], check=True, capture_output=True)
        self.assertFalse(pf.is_unshared(path))

    @unittest.skipUnless(pf.WINDOWS, "NTFS ACLs")
    def test_widening_the_acl_is_noticed(self):
        path = self.root / "key.json"
        path.write_text("{}")
        pf.restrict_to_current_user(path)
        import subprocess
        subprocess.run(["icacls", str(path), "/grant", "*S-1-1-0:R"], check=True, capture_output=True)
        self.assertFalse(pf.is_private(path))


if __name__ == "__main__":
    unittest.main()
