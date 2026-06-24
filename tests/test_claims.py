import tempfile
import unittest
from pathlib import Path

import pandas as pd

from observatoire.claim_labeling import label_claim_clusters, write_claim_cluster_labels
from observatoire.claims import extract_claims_from_comments, write_no_claim_summary


class FakeLLMClient:
    def complete_json(self, system_prompt, user_prompt):
        return {
            "claims": [
                {
                    "claim": "Le financement du blocage des prix est jugé insuffisamment expliqué.",
                    "evidence": "le financement vous le trouvez comment",
                    "confidence": 0.86,
                    "abstraction_level": "low",
                },
                {
                    "claim": "Ce claim trop faible doit être filtré.",
                    "evidence": "faible",
                    "confidence": 0.2,
                    "abstraction_level": "low",
                },
            ]
        }


class ClaimsTest(unittest.TestCase):
    def test_extract_claims_keeps_grounded_confident_claims(self):
        comments = pd.DataFrame(
            [
                {
                    "comment_id": "c1",
                    "video_id": "v1",
                    "video_title": "Video",
                    "actor": "LFI",
                    "sequence": "pouvoir_achat",
                    "text_clean": "C'est bien beau, mais le financement vous le trouvez comment ?",
                    "semantic_candidate": True,
                }
            ]
        )

        claims = extract_claims_from_comments(
            comments,
            client=FakeLLMClient(),
            min_confidence=0.65,
            prompt_path="prompts/extract_claims.md",
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims.iloc[0]["claim_id"], "c1::1")
        self.assertIn("financement", claims.iloc[0]["claim_text"])

    def test_no_claim_summary_counts_candidate_without_claim(self):
        comments = pd.DataFrame(
            [
                {"comment_id": "c1", "semantic_candidate": True},
                {"comment_id": "c2", "semantic_candidate": True},
                {"comment_id": "c3", "semantic_candidate": False},
            ]
        )
        claims = pd.DataFrame([{"comment_id": "c1"}])

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "no_claim_summary.csv"
            write_no_claim_summary(comments, claims, output)
            summary = pd.read_csv(output)

        counts = dict(zip(summary["reason"], summary["n_comments"]))
        self.assertEqual(counts["candidate_with_claim"], 1)
        self.assertEqual(counts["candidate_no_claim"], 1)

    def test_fallback_labeling_writes_markdown(self):
        clusters = pd.DataFrame(
            [
                {
                    "claim_cluster": 0,
                    "label_auto": "financement / prix",
                    "size": 3,
                    "mean_confidence": 0.8,
                    "examples_json": '["Le financement est flou"]',
                }
            ]
        )

        labels = label_claim_clusters(clusters, client=None)

        with tempfile.TemporaryDirectory() as tmp:
            output = write_claim_cluster_labels(labels, Path(tmp) / "labels.md")
            text = output.read_text(encoding="utf-8")

        self.assertIn("financement / prix", text)


if __name__ == "__main__":
    unittest.main()

