import subprocess
import sys
import unittest


class CliTest(unittest.TestCase):
    def test_help_does_not_import_heavy_pipeline_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-m", "observatoire.cli", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("--config", result.stdout)
        self.assertIn("--extract-discourse-cards", result.stdout)
        self.assertIn("--cluster-discourse-cards", result.stdout)
        self.assertIn("--build-discourse-graph", result.stdout)


if __name__ == "__main__":
    unittest.main()
