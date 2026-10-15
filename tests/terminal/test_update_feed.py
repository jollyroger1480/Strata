"""The signed feed and its checks: a bad signature, a bad hash and a downgrade are all refused, and the choice of engine package
follows the hardware. No network and no engine here (the end-to-end flow is test_updater.py)."""

import base64
import datetime as dt
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import feed as feedmod, signing, transfer
from tests.terminal import update_fixture as fx

SEED1 = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
PUB1 = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
SIG1 = bytes.fromhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
SEED2 = bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb")
PUB2 = bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c")
SIG2 = bytes.fromhex("92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00")


class Ed25519(unittest.TestCase):
    def test_rfc8032_vectors(self):
        self.assertEqual(signing.public_from_seed(SEED1), PUB1)
        self.assertEqual(signing.sign_raw(SEED1, b""), SIG1)
        self.assertTrue(signing.verify_raw(PUB1, b"", SIG1))
        self.assertEqual(signing.public_from_seed(SEED2), PUB2)
        self.assertEqual(signing.sign_raw(SEED2, b"\x72"), SIG2)
        self.assertTrue(signing.verify_raw(PUB2, b"\x72", SIG2))

    def test_tampering_is_refused(self):
        self.assertFalse(signing.verify_raw(PUB2, b"\x73", SIG2))
        self.assertFalse(signing.verify_raw(PUB1, b"\x72", SIG2))
        bad = bytearray(SIG2)
        bad[10] ^= 1
        self.assertFalse(signing.verify_raw(PUB2, b"\x72", bytes(bad)))
        self.assertFalse(signing.verify_raw(PUB2, b"\x72", SIG2[:63]))
        high_s = SIG2[:32] + (int.from_bytes(SIG2[32:], "little") + signing._L).to_bytes(32, "little")
        self.assertFalse(signing.verify_raw(PUB2, b"\x72", high_s), "a non-canonical s must be refused")

    def test_documents_and_key_ids(self):
        key = signing.generate_key()
        trusted = fx.trusted_of(key)
        signature = signing.sign_bytes(key, b"hello")
        self.assertEqual(signing.verify_bytes(trusted, b"hello", signature), key["key_id"])
        with self.assertRaises(signing.SignatureError):
            signing.verify_bytes(trusted, b"hellO", signature)
        other = signing.generate_key()
        with self.assertRaisesRegex(signing.SignatureError, "does not trust"):
            signing.verify_bytes(fx.trusted_of(other), b"hello", signature)
        entry = signing.public_part(key)
        entry["key_id"] = "0" * 16
        with self.assertRaises(signing.SignatureError):
            signing.parse_public(entry)

    def test_minisign_app_signature(self):
        public, secret = fx.minisign_pair()
        blob = os.urandom(5000)
        sig = fx.minisign_sign(blob, secret)
        signing.minisign_verify(public, sig, [blob[:1000], blob[1000:]])
        with self.assertRaises(signing.SignatureError):
            signing.minisign_verify(public, sig, [blob + b"x"])
        other_public, _ = fx.minisign_pair()
        with self.assertRaisesRegex(signing.SignatureError, "different key"):
            signing.minisign_verify(other_public, sig, [blob])
        text = base64.b64decode(sig).decode().replace("timestamp:1", "timestamp:2")
        with self.assertRaisesRegex(signing.SignatureError, "comment was changed"):
            signing.minisign_verify(public, base64.b64encode(text.encode()).decode(), [blob])


def entry(version="0.1.46", backend="cuda", variant="cuda13", archs=("sm_86", "compute_89"), driver="580.65", **extra):
    base = {"version": version, "platform": "linux-x86_64", "backend": backend, "variant": variant, "archs": list(archs),
            "min_driver": {"nvidia": driver} if backend == "cuda" else {}, "min_runtime": None, "build_id": f"b-{version}-{variant}",
            "format": "tar.gz", "url": f"engine/e-{version}-{variant}.tar.gz", "size": 1000, "sha256": "a" * 64,
            "manifest_sha256": "b" * 64, "installed_bytes": 2000, "notes": ""}
    base.update(extra)
    return base


def document(**over):
    doc = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": 5,
           "generated_at": "2026-10-15T00:00:00Z", "expires_at": None, "engine": {"packages": [entry()]}}
    doc.update(over)
    return doc


def signed(doc, key):
    payload = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    return payload, json.dumps(signing.sign_bytes(key, payload)).encode()


class FeedChecks(unittest.TestCase):
    def setUp(self):
        self.key = signing.generate_key()
        self.trusted = fx.trusted_of(self.key)

    def load(self, payload, sig, **guards):
        return feedmod.verify_and_parse(payload, sig, source="/feeds/strata-feed.json", trusted=self.trusted, **guards)

    def test_a_good_feed_loads(self):
        feed = self.load(*signed(document(), self.key))
        self.assertEqual((feed.sequence, feed.channel, feed.key_id), (5, "stable", self.key["key_id"]))
        self.assertEqual(feed.engine[0].label, "0.1.46 linux-x86_64-cuda-cuda13")

    def test_a_bad_signature_is_refused(self):
        payload, sig = signed(document(), self.key)
        forged = json.loads(sig)
        raw = bytearray(base64.b64decode(forged["value"]))
        raw[3] ^= 0x40
        forged["value"] = base64.b64encode(bytes(raw)).decode()
        with self.assertRaisesRegex(feedmod.FeedError, "not trusted"):
            self.load(payload, json.dumps(forged).encode())

    def test_a_feed_changed_after_signing_is_refused(self):
        payload, sig = signed(document(), self.key)
        with self.assertRaisesRegex(feedmod.FeedError, "does not match the signed content"):
            self.load(payload.replace(b'"sequence": 5', b'"sequence": 6'), sig)

    def test_a_feed_signed_by_an_unknown_key_is_refused(self):
        payload, sig = signed(document(), signing.generate_key())
        with self.assertRaisesRegex(feedmod.FeedError, "does not trust"):
            self.load(payload, sig)

    def test_no_trusted_keys_means_no_update(self):
        payload, sig = signed(document(), self.key)
        with self.assertRaisesRegex(feedmod.FeedError, "trusts no update key"):
            feedmod.verify_and_parse(payload, sig, source="x/strata-feed.json", trusted={})

    def test_an_older_feed_is_a_downgrade(self):
        payload, sig = signed(document(sequence=4), self.key)
        with self.assertRaises(feedmod.FeedDowngrade):
            self.load(payload, sig, last_sequence=5)
        self.assertEqual(self.load(*signed(document(sequence=5), self.key), last_sequence=5).sequence, 5)
        self.assertEqual(self.load(*signed(document(sequence=9), self.key), last_sequence=5).sequence, 9)

    def test_an_expired_feed_and_another_channel_are_refused(self):
        payload, sig = signed(document(expires_at="2026-01-01T00:00:00Z"), self.key)
        with self.assertRaisesRegex(feedmod.FeedError, "expired"):
            self.load(payload, sig, now=dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc))
        with self.assertRaisesRegex(feedmod.FeedError, "channel"):
            self.load(*signed(document(channel="beta"), self.key), channel="stable")

    def test_malformed_entries_are_errors_not_defaults(self):
        for change in ({"sha256": "nothex"}, {"size": 0}, {"format": "rar"}, {"platform": "plan9"}, {"backend": "tpu"},
                       {"version": "latest"}, {"archs": []}, {"manifest_sha256": None}, {"size": True}):
            with self.subTest(change=change):
                with self.assertRaises(feedmod.FeedError):
                    self.load(*signed(document(engine={"packages": [entry(**change)]}), self.key))
        with self.assertRaisesRegex(feedmod.FeedError, "twice"):
            self.load(*signed(document(engine={"packages": [entry(), entry()]}), self.key))
        with self.assertRaises(feedmod.FeedError):
            self.load(b"not json", signed(document(), self.key)[1])

    def test_versions_are_strict_and_ordered(self):
        self.assertLess(feedmod.version_key("0.1.9"), feedmod.version_key("0.1.10"))
        self.assertLess(feedmod.version_key("0.1.45"), feedmod.version_key("0.1.45.1"))
        with self.assertRaises(feedmod.FeedError):
            feedmod.version_key("0.1.x")


class Selection(unittest.TestCase):
    HW = {"platform": "linux-x86_64", "backend": "cuda", "runtime": "0.1.45", "nvidia_driver": "615.71", "cuda_archs": ["sm_86"], "gfx": []}

    def pick(self, packages, **hw):
        parsed = [feedmod._engine_package(p) for p in packages]
        return feedmod.select_engine(parsed, {**self.HW, **hw})

    def test_cuda_13_when_the_driver_is_new_enough_else_12(self):
        both = [entry(variant="cuda12", driver="575.51", archs=("sm_60", "sm_86", "compute_89")), entry(variant="cuda13")]
        self.assertEqual(self.pick(both)[0].variant, "cuda13")
        old = self.pick(both, nvidia_driver="576.02")
        self.assertEqual(old[0].variant, "cuda12")
        self.assertTrue(any("580.65" in r for r in old[1]))
        self.assertIsNone(self.pick(both, nvidia_driver="500.00")[0])

    def test_the_newest_version_wins_and_the_card_must_be_covered(self):
        newer = [entry("0.1.45"), entry("0.1.46")]
        self.assertEqual(self.pick(newer)[0].version, "0.1.46")
        pascal = self.pick([entry(archs=("sm_75", "sm_86", "compute_86"))], cuda_archs=["sm_61"])
        self.assertIsNone(pascal[0])
        self.assertIn("this PC's GPU is sm_61", pascal[1][0])
        self.assertIsNotNone(self.pick([entry(archs=("sm_75", "compute_75"))], cuda_archs=["sm_120"])[0], "PTX covers a newer card")

    def test_hip_by_gfx_sycl_and_metal_by_platform(self):
        hip = [entry(backend="hip", variant="", archs=("gfx1200", "gfx1201"))]
        self.assertIsNotNone(self.pick(hip, backend="hip", gfx=["gfx1201"])[0])
        self.assertIsNone(self.pick(hip, backend="hip", gfx=["gfx1100"])[0])
        self.assertIsNotNone(self.pick(hip, backend="hip", gfx=[])[0], "an unreadable gfx is not a refusal")
        self.assertIsNone(self.pick(hip, backend="cuda")[0], "a HIP package is not offered to an NVIDIA PC")
        sycl = [entry(backend="sycl", variant="", archs=("intel-arc",))]
        self.assertIsNotNone(self.pick(sycl, backend="sycl")[0])
        metal = dict(entry(backend="metal", variant="", archs=("apple-gpu",)), platform="macos-arm64")
        self.assertIsNone(self.pick([metal])[0], "wrong platform")
        self.assertIsNotNone(self.pick([metal], platform="macos-arm64", backend="metal")[0])

    def test_min_runtime(self):
        got = self.pick([entry(min_runtime="0.1.50")])
        self.assertIsNone(got[0])
        self.assertIn("needs control 0.1.50", got[1][0])


class References(unittest.TestCase):
    def test_references_stay_inside_the_feed(self):
        base = "http://127.0.0.1:9/feed/"
        self.assertEqual(transfer.resolve_ref(base, "engine/a.tar.gz"), "http://127.0.0.1:9/feed/engine/a.tar.gz")
        for bad in ("../x", "/etc/passwd", "a/../../b", "file:///etc/passwd", "ftp://example.com/x", "a\\b", "http://example.com/x"):
            with self.subTest(bad=bad), self.assertRaises(transfer.TransferError):
                transfer.resolve_ref(base, bad)
        self.assertEqual(transfer.resolve_ref("https://example.com/f/", "https://cdn.example.com/x"), "https://cdn.example.com/x")

    def test_a_local_feed_cannot_name_a_remote_address(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(transfer.TransferError):
                transfer.resolve_ref(folder, "https://evil.example.com/x")
            with self.assertRaises(transfer.TransferError):
                transfer.resolve_ref(folder, "../outside")
            self.assertTrue(transfer.resolve_ref(folder, "engine/a").startswith(str(Path(folder).resolve())))


class Downloads(unittest.TestCase):
    def test_a_partial_file_resumes_and_a_bad_hash_is_deleted(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            data = os.urandom(300_000)
            source = Path(folder) / "a.bin"
            source.write_bytes(data)
            partial = Path(folder) / "x.part"
            partial.write_bytes(data[:100_000])
            seen = []
            done = transfer.download(str(source), partial, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                     progress=lambda d, t: seen.append(d))
            self.assertEqual(done.read_bytes(), data)
            self.assertEqual(seen[0] > 100_000, True, "it continued from the partial file instead of starting over")
            partial.write_bytes(data[:50_000])
            with self.assertRaises(transfer.ChecksumMismatch):
                transfer.download(str(source), partial, size=len(data), sha256="0" * 64)
            self.assertFalse(partial.exists(), "bytes that fail the signed hash are not kept")
            partial.write_bytes(b"x" * (len(data) + 5))
            transfer.download(str(source), partial, size=len(data), sha256=hashlib.sha256(data).hexdigest())
            self.assertEqual(partial.read_bytes(), data, "an oversized stale partial is discarded, not appended to")


if __name__ == "__main__":
    unittest.main()
