import json
import tempfile
import unittest
from pathlib import Path

from observatoire.cache import DatasetCache, build_raw_cache_context


class DatasetCacheTest(unittest.TestCase):
    def test_context_fingerprint_changes_when_replies_change(self):
        base = build_raw_cache_context(["abc123def45"], 100, False)
        with_replies = build_raw_cache_context(["abc123def45"], 100, True)

        self.assertNotEqual(base["fingerprint"], with_replies["fingerprint"])

    def test_manifest_validation_uses_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "youtube_comments_raw_anonymized.csv").write_text("video_id\nabc123def45\n", encoding="utf-8")
            (data_dir / "video_metadata.csv").write_text("video_id\nabc123def45\n", encoding="utf-8")

            cache = DatasetCache(data_dir)
            context = build_raw_cache_context(["abc123def45"], 100, False)
            cache.write_manifest(context, stats={"comments_rows": 1})

            self.assertTrue(cache.is_valid_for(context))
            manifest = json.loads(cache.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["stats"]["comments_rows"], 1)

            changed = build_raw_cache_context(["abc123def45"], 200, False)
            self.assertFalse(cache.is_valid_for(changed))


if __name__ == "__main__":
    unittest.main()

