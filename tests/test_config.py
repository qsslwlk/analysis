import json
import tempfile
import unittest
from pathlib import Path

from observatoire.config import load_frame_lexicon, load_video_config


class ConfigTest(unittest.TestCase):
    def test_load_video_config_accepts_videos_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "video_url": "https://www.youtube.com/watch?v=abc123def45",
                                "actor": "LFI",
                                "sequence": "test",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            videos = load_video_config(path)

        self.assertEqual(videos[0]["actor"], "LFI")

    def test_load_video_config_requires_video_identifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.json"
            path.write_text(json.dumps({"videos": [{"actor": "LFI"}]}), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_video_config(path)

    def test_load_frame_lexicon_validates_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.json"
            path.write_text(json.dumps({"frames": {"justice": ["salaires"]}}), encoding="utf-8")

            frames = load_frame_lexicon(path)

        self.assertEqual(frames["justice"], ["salaires"])


if __name__ == "__main__":
    unittest.main()

