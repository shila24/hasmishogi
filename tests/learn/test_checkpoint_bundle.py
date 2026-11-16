"""Self-contained checkpoint bundles (standard library only; no torch needed)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hasami_learn.checkpoint_bundle import BundleError, bundle_checkpoint, engine_dependency, file_hashes


def make_checkpoint(directory: Path, weights: bytes, hint: dict | None = None, extra: dict | None = None) -> str:
    directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(weights)
    sha = hashlib.sha256(weights).hexdigest()
    manifest = {"weights_sha256": sha, "inference_hint": hint or {"mode": "policy"}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for name, text in (extra or {}).items():
        (directory / name).write_text(text)
    return sha


class CheckpointBundleTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hasami-bundle-"))
        self.addCleanup(__import__("shutil").rmtree, self.root, True)

    def practical(self, engine_sha=None):
        engine_sha_actual = make_checkpoint(self.root / "engine-run/iter_000010", b"engine weights")
        make_checkpoint(self.root / "practical-run/iter_000003", b"practical weights",
                        {"mode": "practical", "levels": 6, "engine_checkpoint": "../../engine-run/iter_000010",
                         "engine_sha256": engine_sha or engine_sha_actual}, {"levels.json": "{}"})
        return self.root / "practical-run/iter_000003", engine_sha_actual

    def test_plain_checkpoint_copies_every_file(self):
        make_checkpoint(self.root / "a", b"w", extra={"notes.txt": "x"})
        hashes = bundle_checkpoint(self.root / "a", self.root / "b")
        self.assertEqual(set(hashes), {"manifest.json", "weights.pt", "notes.txt"})
        self.assertEqual(hashes, file_hashes(self.root / "b"))

    def test_engine_dependency_is_materialised_and_the_hint_rewritten(self):
        source, engine_sha = self.practical()
        hashes = bundle_checkpoint(source, self.root / "export")
        self.assertEqual(set(hashes), {"manifest.json", "weights.pt", "levels.json", "engine/manifest.json",
                                       "engine/weights.pt"})
        hint = json.loads((self.root / "export/manifest.json").read_text())["inference_hint"]
        self.assertEqual(hint["engine_checkpoint"], "engine")
        self.assertEqual(hint["engine_sha256"], engine_sha)
        self.assertEqual(engine_dependency(self.root / "export"), (self.root / "export/engine").resolve())
        # A bundle of a bundle is the same bundle.
        self.assertEqual(bundle_checkpoint(self.root / "export", self.root / "again"), hashes)

    def test_wrong_engine_hash_is_refused_and_leaves_nothing(self):
        source, _ = self.practical(engine_sha="0" * 64)
        with self.assertRaisesRegex(BundleError, "engine_sha256"):
            bundle_checkpoint(source, self.root / "export")
        self.assertFalse((self.root / "export").exists())
        self.assertFalse((self.root / ".tmp_export").exists())

    def test_tampered_weights_links_existing_targets_and_absolute_references_are_refused(self):
        make_checkpoint(self.root / "a", b"w")
        (self.root / "a/weights.pt").write_bytes(b"tampered")
        with self.assertRaisesRegex(BundleError, "does not match the manifest"):
            bundle_checkpoint(self.root / "a", self.root / "b")
        make_checkpoint(self.root / "c", b"w")
        (self.root / "c/link").symlink_to(self.root / "c/weights.pt")
        with self.assertRaisesRegex(BundleError, "links"):
            bundle_checkpoint(self.root / "c", self.root / "d")
        make_checkpoint(self.root / "e", b"w")
        (self.root / "f").mkdir()
        with self.assertRaisesRegex(BundleError, "already exists"):
            bundle_checkpoint(self.root / "e", self.root / "f")
        make_checkpoint(self.root / "g", b"w", {"mode": "practical", "engine_checkpoint": str(self.root / "e"),
                                                "engine_sha256": "0" * 64})
        with self.assertRaisesRegex(BundleError, "relative"):
            bundle_checkpoint(self.root / "g", self.root / "h")

    def test_an_engine_with_its_own_dependency_is_refused(self):
        inner = make_checkpoint(self.root / "inner", b"inner")
        middle = make_checkpoint(self.root / "middle", b"middle", {"mode": "practical", "engine_checkpoint": "../inner",
                                                                    "engine_sha256": inner})
        make_checkpoint(self.root / "outer", b"outer", {"mode": "practical", "engine_checkpoint": "../middle",
                                                         "engine_sha256": middle})
        with self.assertRaisesRegex(BundleError, "must not depend"):
            bundle_checkpoint(self.root / "outer", self.root / "out")


if __name__ == "__main__":
    unittest.main()
