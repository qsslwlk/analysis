import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd

if find_spec("plotly") is not None:
    from observatoire_youtube_poc import cluster_embeddings


@unittest.skipIf(find_spec("plotly") is None, "plotly is not installed in this test environment")
class SemanticClusteringTest(unittest.TestCase):
    def test_small_semantic_clusters_are_left_as_noise(self):
        comments = pd.DataFrame(
            [
                {
                    "comment_id": f"c{index}",
                    "actor": "LFI" if index % 2 else "RN",
                    "video_id": "v1",
                    "video_title": "Video",
                    "text_clean": f"Commentaire discursif numéro {index}",
                }
                for index in range(5)
            ]
        )
        embeddings = np.eye(len(comments))

        with tempfile.TemporaryDirectory() as tmp:
            comments_out, clusters_out, _ = cluster_embeddings(
                comments,
                embeddings,
                min_cluster_size=8,
                outputs_dir=tmp,
            )
            exported_clusters = pd.read_csv(Path(tmp) / "semantic_clusters.csv")

        self.assertTrue(clusters_out.empty)
        self.assertIn("size", clusters_out.columns)
        self.assertEqual(set(comments_out["semantic_cluster"]), {-1})
        self.assertIn("size", exported_clusters.columns)


if __name__ == "__main__":
    unittest.main()
