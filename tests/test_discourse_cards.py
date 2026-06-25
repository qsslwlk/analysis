import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from observatoire.discourse_cards import extract_discursive_cards_from_comments, write_discursive_card_coverage
from observatoire.discourse_clustering import cluster_discursive_cards


class FakeDiscursiveLLMClient:
    def __init__(self):
        self.calls = 0

    def complete_json(self, system_prompt, user_prompt):
        self.calls += 1
        return {
            "discursive_card": {
                "theme_main": "Pouvoir d'achat",
                "subthemes": ["prix", "salaires"],
                "dominant_frame": "justice sociale",
                "secondary_frames": ["crédibilité économique"],
                "stance_targets": [
                    {
                        "target": "programme économique",
                        "stance": "scepticisme",
                        "evidence": "comment vous financez tout ça",
                        "confidence": 0.84,
                    }
                ],
                "central_argument": "Le commentaire demande une justification du financement.",
                "argument_type": "économique",
                "attack_or_objection": "Le financement est jugé insuffisamment expliqué.",
                "emotion_tone": "sceptique",
                "ambiguities": ["Le commentaire peut être une demande sincère ou une objection rhétorique."],
                "representative_quotes": ["comment vous financez tout ça"],
                "confidence": 0.86,
                "discursive_summary": "Scepticisme économique envers le financement du programme.",
            }
        }


class DiscourseCardsTest(unittest.TestCase):
    def test_extract_discursive_cards_keeps_grounded_structured_card(self):
        comments = pd.DataFrame(
            [
                {
                    "comment_id": "c1",
                    "video_id": "v1",
                    "video_title": "Video",
                    "actor": "LFI",
                    "sequence": "pouvoir_achat",
                    "text_clean": "C'est bien beau, mais comment vous financez tout ça ?",
                    "semantic_candidate": True,
                }
            ]
        )

        cards = extract_discursive_cards_from_comments(
            comments,
            client=FakeDiscursiveLLMClient(),
            prompt_path="prompts/extract_discursive_card.md",
        )

        self.assertEqual(len(cards), 1)
        self.assertEqual(cards.iloc[0]["card_id"], "c1::discursive_card")
        self.assertEqual(cards.iloc[0]["dominant_frame"], "justice sociale")
        self.assertIn("Scepticisme économique", cards.iloc[0]["discursive_summary"])

    def test_extract_discursive_cards_reuses_llm_cache(self):
        comments = pd.DataFrame(
            [
                {
                    "comment_id": "c1",
                    "video_id": "v1",
                    "video_title": "Video",
                    "actor": "LFI",
                    "sequence": "pouvoir_achat",
                    "text_clean": "C'est bien beau, mais comment vous financez tout ça ?",
                    "semantic_candidate": True,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "discursive_card_llm_cache.jsonl"
            first_client = FakeDiscursiveLLMClient()
            first_cards = extract_discursive_cards_from_comments(
                comments,
                client=first_client,
                prompt_path="prompts/extract_discursive_card.md",
                llm_cache_path=cache_path,
                cache_namespace="test:model",
            )
            second_client = FakeDiscursiveLLMClient()
            second_cards = extract_discursive_cards_from_comments(
                comments,
                client=second_client,
                prompt_path="prompts/extract_discursive_card.md",
                llm_cache_path=cache_path,
                cache_namespace="test:model",
            )

        self.assertEqual(first_client.calls, 1)
        self.assertEqual(second_client.calls, 0)
        self.assertEqual(len(first_cards), 1)
        self.assertEqual(len(second_cards), 1)

    def test_discursive_card_coverage_counts_candidates_without_card(self):
        comments = pd.DataFrame(
            [
                {"comment_id": "c1", "semantic_candidate": True},
                {"comment_id": "c2", "semantic_candidate": True},
                {"comment_id": "c3", "semantic_candidate": False},
            ]
        )
        cards = pd.DataFrame([{"comment_id": "c1"}])

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "coverage.csv"
            write_discursive_card_coverage(comments, cards, output)
            summary = pd.read_csv(output)

        counts = dict(zip(summary["reason"], summary["n_comments"]))
        self.assertEqual(counts["candidate_with_discursive_card"], 1)
        self.assertEqual(counts["candidate_without_discursive_card"], 1)

    def test_discursive_clustering_handles_all_noise(self):
        cards = pd.DataFrame(
            [
                {
                    "card_id": "c1::discursive_card",
                    "comment_id": "c1",
                    "actor": "LFI",
                    "dominant_frame": "justice sociale",
                    "discursive_summary": "Scepticisme économique envers le financement.",
                    "representative_quotes_json": '["comment vous financez"]',
                    "confidence": 0.8,
                },
                {
                    "card_id": "c2::discursive_card",
                    "comment_id": "c2",
                    "actor": "RN",
                    "dominant_frame": "priorité nationale",
                    "discursive_summary": "Appui à une priorité nationale sur les aides.",
                    "representative_quotes_json": '["priorité aux français"]',
                    "confidence": 0.82,
                },
            ]
        )
        embeddings = np.eye(len(cards))

        with tempfile.TemporaryDirectory() as tmp:
            cards_out, clusters_out = cluster_discursive_cards(
                cards,
                embeddings,
                min_cluster_size=6,
                outputs_dir=tmp,
            )
            exported_clusters = pd.read_csv(Path(tmp) / "discursive_clusters.csv")

        self.assertTrue(clusters_out.empty)
        self.assertIn("size", clusters_out.columns)
        self.assertEqual(set(cards_out["discursive_cluster"]), {-1})
        self.assertIn("size", exported_clusters.columns)


if __name__ == "__main__":
    unittest.main()
