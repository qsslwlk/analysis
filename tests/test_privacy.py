import unittest

from observatoire.privacy import assert_privacy_safe_columns, forbidden_columns


class PrivacyTest(unittest.TestCase):
    def test_forbidden_columns_are_detected(self):
        self.assertEqual(forbidden_columns(["video_id", "author_display_name"]), ["author_display_name"])

    def test_safe_columns_pass(self):
        assert_privacy_safe_columns(["video_id", "author_hash", "text_original"])

    def test_unsafe_columns_raise(self):
        with self.assertRaises(ValueError):
            assert_privacy_safe_columns(["author_channel_url"])


if __name__ == "__main__":
    unittest.main()

