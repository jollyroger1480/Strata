"""The app reports its version as 0.2.0-beta; the update check must compare it, not choke on it."""
import unittest

from strata_control import feed as feedmod


class AppPrereleaseVersionTests(unittest.TestCase):
    def test_a_pre_release_sorts_below_the_same_number_and_above_the_older_one(self):
        key = feedmod.app_version_key
        self.assertLess(key("0.2.0-beta"), key("0.2.0"))
        self.assertGreater(key("0.2.0-beta"), key("0.1.46"))
        self.assertLess(key("0.2.0-beta"), key("0.2.0.1"))
        self.assertEqual(key("0.2.0"), key("0.2.0"))

    def test_the_feeds_own_versions_stay_strict(self):
        with self.assertRaises(feedmod.FeedError):
            feedmod.version_key("0.2.0-beta")
        with self.assertRaises(feedmod.FeedError):
            feedmod.app_version_key("0.2.0-be ta")


if __name__ == "__main__":
    unittest.main()
