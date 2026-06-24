import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from observatoire.discourse_graph import build_discourse_graph


def _card(
    comment_id,
    frame,
    claim,
    target,
    stance,
    actor="RN",
    tone="indignation",
):
    return {
        "card_id": f"{comment_id}::discursive_card",
        "comment_id": comment_id,
        "video_id": "v1",
        "video_title": "Interview politique",
        "actor": actor,
        "sequence": "media",
        "theme_main": "médias",
        "subthemes_json": json.dumps(["journalisme"], ensure_ascii=False),
        "dominant_frame": frame,
        "secondary_frames_json": "[]",
        "stance_targets_json": json.dumps(
            [
                {
                    "target": target,
                    "stance": stance,
                    "evidence": "la journaliste essaye de le piéger",
                    "confidence": 0.9,
                }
            ],
            ensure_ascii=False,
        ),
        "central_argument": claim,
        "argument_type": "biais médiatique",
        "attack_or_objection": "",
        "emotion_tone": tone,
        "ambiguities_json": "[]",
        "representative_quotes_json": json.dumps(["la journaliste essaye de le piéger"], ensure_ascii=False),
        "confidence": 0.88,
        "discursive_summary": f"{frame} — {claim}",
        "raw_comment_preview": "Bravo, la journaliste essaye de le piéger.",
    }


class DiscourseGraphTest(unittest.TestCase):
    def test_build_discourse_graph_exports_interpretable_community(self):
        cards = pd.DataFrame(
            [
                _card(
                    "c1",
                    "biais médiatique",
                    "le traitement journalistique est perçu comme hostile",
                    "journaliste",
                    "rejet",
                ),
                _card(
                    "c2",
                    "biais médiatique",
                    "le traitement journalistique est perçu comme hostile",
                    "journaliste",
                    "rejet",
                ),
                _card(
                    "c3",
                    "pouvoir d'achat",
                    "les prix empêchent de vivre dignement",
                    "gouvernement",
                    "rejet",
                    actor="LFI",
                    tone="inquiétude",
                ),
                _card(
                    "c4",
                    "souveraineté",
                    "la France doit décider seule",
                    "France",
                    "supportive",
                    tone="fierté",
                ),
            ]
        )
        comments = pd.DataFrame(
            [
                {"comment_id": "c1", "published_at_comment": "2024-06-01T12:00:00Z", "channel_title": "Chaîne A"},
                {"comment_id": "c2", "published_at_comment": "2024-06-02T12:00:00Z", "channel_title": "Chaîne A"},
                {"comment_id": "c3", "published_at_comment": "2024-06-03T12:00:00Z", "channel_title": "Chaîne B"},
                {"comment_id": "c4", "published_at_comment": "2024-06-04T12:00:00Z", "channel_title": "Chaîne C"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = build_discourse_graph(
                cards,
                comments,
                outputs_dir=tmp,
                min_community_size=2,
                similarity_threshold=0.5,
                top_k=3,
            )
            exported_nodes = pd.read_csv(Path(tmp) / "discursive_nodes.csv")
            exported_edges = pd.read_csv(Path(tmp) / "discursive_edges.csv")
            exported_communities = pd.read_csv(Path(tmp) / "discursive_communities.csv")

        self.assertFalse(result.nodes_df.empty)
        self.assertFalse(result.edges_df.empty)
        self.assertIn("StanceTargetNode", set(exported_nodes["node_type"]))
        self.assertIn("COMMENT_EXPRESSES_STANCE_TOWARD_TARGET", set(exported_edges["edge_type"]))
        self.assertIn("COMMENT_HAS_CANONICAL_CLAIM", set(exported_edges["edge_type"]))
        self.assertEqual(exported_communities.iloc[0]["size"], 2)
        self.assertEqual(set(result.units_df.loc[result.units_df["comment_id"].isin(["c1", "c2"]), "discursive_community"]), {0})
        france_node = exported_nodes[exported_nodes["label"] == "France"].iloc[0]
        self.assertTrue(france_node["is_vague"])

    def test_build_discourse_graph_handles_empty_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = build_discourse_graph(pd.DataFrame(), pd.DataFrame(), outputs_dir=tmp)
            exported_communities = pd.read_csv(Path(tmp) / "discursive_communities.csv")

        self.assertTrue(result.communities_df.empty)
        self.assertIn("size", exported_communities.columns)


if __name__ == "__main__":
    unittest.main()
